"""
Literature Review System - Main Orchestration

This is the main system that orchestrates all modules to generate complete literature reviews.
Each component is responsible for generating its specific part, while this system handles
coordination, assembly, and incremental saving to MongoDB.
"""

import re
import concurrent.futures
from datetime import datetime
from typing import Dict, Optional, Tuple, List

from agent.topic_planning import TopicPlanningAgent
from agent.literature_collection import LiteratureCollectionAgent
from agent.review_generation import ReviewGenerationAgent
from agent.table_generation import TableGenerationAgent
from agent.overview_commentary import OverviewCommentaryAgent
from agent.citation_formatting import CitationFormattingAgent
from components.llm_call import handler
from components.config import *



class LiteratureReviewSystem:
    """
    Main Literature Review System that coordinates all agents and assembles the final review.

    Workflow:
    1. Topic Planning → Save subtopics
    2. Literature Collection → Save literature data
    3. Review Generation (Paragraphs) → Save paragraphs
    4. Table Generation → Save tables
    5. Overview & Commentary → Save overview/critique
    6. Assemble Full Text → Save complete review
    7. Translation (optional) → Save translated version
    """

    def __init__(
        self,
        user_id: str,
        task_id: str,
        kb_id: str,
        mode: Optional[int] = None,
        structure: int = 0
    ):
        """
        Initialize the Literature Review System.

        Args:
            user_id: User identifier
            task_id: Task identifier
            kb_id: Knowledge base ID for literature retrieval
            mode: Operation mode (1 for long-form)
            structure: Review structure (0 for standard, 1 for language-separated)
        """
        self.user_id = user_id
        self.task_id = task_id
        self.kb_id = kb_id
        self.mode = mode
        self.structure = structure

        # Initialize all agents
        self.topic_agent = TopicPlanningAgent()
        self.literature_agent = LiteratureCollectionAgent(kb_id, mode=mode)
        self.review_agent = ReviewGenerationAgent(mode=mode)
        self.table_agent = TableGenerationAgent()
        self.overview_agent = OverviewCommentaryAgent()
        self.citation_agent = CitationFormattingAgent()

        # MongoDB connection
        self.mongo_client = None
        self.collection = None

    def _connect_mongodb(self):
        """Establish MongoDB connection."""
        if not self.mongo_client:
            self.mongo_client = MongoClient(
                f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"
            )
            db = self.mongo_client["Newbit"]
            self.collection = db["review"]

    def _save_to_mongodb(self, data: Dict):
        """Save data to MongoDB incrementally."""
        self._connect_mongodb()
        self.collection.update_one(
            {"user_id": self.user_id, "task_id": self.task_id},
            {"$set": {**data, "update_time": datetime.now()}},
            upsert=True
        )

    def _update_state(self, state: int, additional_data: Dict = None):
        """
        Update task state.

        Args:
            state: Task state (0=failed, 1=success, 2=in_progress)
            additional_data: Additional fields to update
        """
        update_data = {"state": state}
        if additional_data:
            update_data.update(additional_data)
        self._save_to_mongodb(update_data)

    def _assemble_topic_text(
        self,
        topic: str,
        topic_index: int,
        content_types: Dict,
        paragraph_results: Dict,
        table_results: Dict
    ) -> str:
        """
        Assemble text for a single topic.

        Args:
            topic: Topic name
            topic_index: Index for numbering
            content_types: Content types for this topic
            paragraph_results: Generated paragraphs
            table_results: Generated tables

        Returns:
            Assembled topic text
        """
        chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        topic_number = chinese_numbers[topic_index] if topic_index < len(chinese_numbers) else str(topic_index + 1)
        topic_title = f"## （{topic_number}）{topic}"
        topic_sections = [topic_title]

        for content_type in content_types.keys():
            # Add paragraph
            if (topic in paragraph_results and
                content_type in paragraph_results[topic] and
                "paragraph" in paragraph_results[topic][content_type]):

                paragraph = paragraph_results[topic][content_type]["paragraph"]
                if not paragraph.startswith("Generation failed:") and \
                   not paragraph.startswith("API call failed:"):
                    topic_sections.append(paragraph)

            # Add table
            if (topic in table_results and
                content_type in table_results[topic] and
                "table" in table_results[topic][content_type]):

                table = table_results[topic][content_type]["table"]
                if not table.startswith("Generation failed:") and \
                   not table.startswith("API call failed:"):
                    topic_sections.append(table)

        return "\n\n".join(topic_sections)

    def _extract_cited_references_from_topics(
        self,
        review_data: Dict,
        topics_text: Dict
    ) -> Tuple[List[str], List[str]]:
        """
        Extract references that are actually cited in all topics.

        Args:
            review_data: Original literature data
            topics_text: Assembled topic texts

        Returns:
            Tuple of (clean_references, pdf_urls)
        """
        all_references = []
        combined_text = "\n\n".join(topics_text.values())

        # Extract citations from each topic's literature
        for topic, content_types in review_data.items():
            for content_type, data_list in content_types.items():
                if data_list:
                    refs = self.citation_agent.extract_cited_references(
                        combined_text,
                        data_list
                    )
                    all_references.extend(refs)

        # Deduplicate and sort
        unique_references = list(set(all_references))
        unique_references.sort()

        # Extract PDF URLs and clean references
        pdf_urls = []
        clean_references = []

        for ref in unique_references:
            pdf_url_match = re.search(r'\[PDF_URL_START\](.*?)\[PDF_URL_END\]', ref)
            if pdf_url_match:
                pdf_url = pdf_url_match.group(1)
                clean_ref = re.sub(r'\[PDF_URL_START\].*?\[PDF_URL_END\]', '', ref).strip()
                pdf_urls.append(pdf_url)
                clean_references.append(clean_ref)
            else:
                pdf_urls.append("")
                clean_references.append(ref)

        return clean_references, pdf_urls

    def _check_empty_literature(self, literature_data: Dict) -> bool:
        """Check if literature data is empty."""
        for subtopic, data in literature_data.items():
            for content_type, docs in data.items():
                if len(docs) > 0:
                    return False
        return True

    def _calculate_chinese_ratio(self, references: List[str]) -> float:
        """Calculate proportion of Chinese references."""
        if not references:
            return 0.0

        zh_count = 0
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')

        for ref in references:
            if chinese_pattern.search(ref):
                zh_count += 1

        return round(zh_count / len(references), 4)

    def _translate_markdown(self, markdown_text: str, target_language: str = "English") -> str:
        """Translate markdown content while preserving formatting."""
        prompt = f"""
<TASK>
You are a precision markdown translation specialist. Please translate the following markdown content into {target_language}, while meticulously preserving all original formatting.
</TASK>

<MARKDOWN_TO_TRANSLATE>
{markdown_text}
</MARKDOWN_TO_TRANSLATE>

<TRANSLATION_REQUIREMENTS>
1. Translate only the plain text content
2. Preserve all formatting markers (such as #, *, -, >)
3. Retain all line breaks, indentation, and whitespace
4. Ensure tables maintain their exact structure
</TRANSLATION_REQUIREMENTS>

<OUTPUT_INSTRUCTION>
Wrap your translation with [TRANS_START] and [TRANS_END] tags.
</OUTPUT_INSTRUCTION>

Your translated markdown:
"""

        try:
            response = handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=8000,
                temperature=0.7
            )
            if "API call failed" not in str(response):
                return self._extract_translation(response)
        except Exception:
            pass

        response = handler.call_llm(
            provider="ark",
            prompt=prompt,
            model="doubao-1-5-lite-32k-250115",
            max_tokens=8000,
            temperature=0.7
        )

        return self._extract_translation(response)

    def _extract_translation(self, response: str) -> str:
        """Extract translated content."""
        pattern = r'\[TRANS_START\]([\s\S]*)\[TRANS_END\]'
        match = re.search(pattern, response)
        return match.group(1) if match else response.replace('[TRANS_START]', '').replace('[TRANS_END]', '')

    def _parallel_translate_fields(self, review_text: Dict) -> str:
        """Translate all review fields in parallel."""
        original_keys = list(review_text["topics"].keys())
        fields = {"overview": review_text["overview"]}

        for key, value in review_text["topics"].items():
            fields[f"topics.{key}"] = value

        fields["critique"] = review_text["critique"]

        def translate_field(field_name, content):
            try:
                return field_name, self._translate_markdown(content)
            except Exception as e:
                print(f"Translation error for '{field_name}': {e}")
                return field_name, content

        translated = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(translate_field, name, content): name
                for name, content in fields.items()
            }
            for future in concurrent.futures.as_completed(futures):
                name, content = future.result()
                translated[name] = content

        result = [translated["overview"]]
        for key in original_keys:
            result.append(translated[f"topics.{key}"])

        result.append("# Summary\n\n" + translated["critique"])
        result.append("# References\n\n" + "\n\n".join(review_text["references"]))

        return "\n\n".join(result)

    def _split_by_language(self, literature_data: Dict) -> Tuple[Dict, Dict]:
        """Split literature into Chinese and English."""
        zh_lit = {}
        en_lit = {}

        for topic, types in literature_data.items():
            zh_lit[topic] = {}
            en_lit[topic] = {}

            for content_type, items in types.items():
                zh_items = [item for item in items if item.get("language") == "中文"]
                en_items = [item for item in items if item.get("language") == "英文"]

                if zh_items:
                    zh_lit[topic][content_type] = zh_items
                if en_items:
                    en_lit[topic][content_type] = en_items

        return zh_lit, en_lit

    def generate_review_structure_0(
        self,
        review_data: Dict,
        main_topic: str,
        review_text: Dict
    ):
        """
        Generate review with standard structure (no language separation).

        This method generates the review incrementally and saves each step to MongoDB.
        """
        print("\n=== Step 3: Generating paragraphs ===")
        paragraph_results = self.review_agent.generate_paragraphs_only(review_data)
        review_text["paragraph_results"] = paragraph_results
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Paragraphs saved to MongoDB")

        print("\n=== Step 4: Generating tables ===")
        table_results = self.table_agent.generate_tables_only(paragraph_results)
        review_text["table_results"] = table_results
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Tables saved to MongoDB")

        print("\n=== Step 5: Assembling topics ===")
        topics_text = {}
        for i, (topic, content_types) in enumerate(review_data.items()):
            topic_text = self._assemble_topic_text(
                topic, i, content_types,
                paragraph_results, table_results
            )
            topics_text[topic] = topic_text

        full_text = "\n\n".join(topics_text.values())
        review_text["topics"] = topics_text
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Topics assembled and saved")

        print("\n=== Step 6: Generating overview and critique ===")
        overview, critique = self.overview_agent.generate_overview_and_critique(
            full_text, main_topic
        )
        review_text["overview"] = overview
        review_text["critique"] = critique
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Overview and critique saved")

        print("\n=== Step 7: Extracting citations ===")
        references, pdf_urls = self._extract_cited_references_from_topics(
            review_data, topics_text
        )
        review_text["references"] = references
        review_text["pdf_urls"] = pdf_urls
        self._save_to_mongodb({"review_text": review_text})
        print("✓ References extracted and saved")

        print("\n=== Step 8: Assembling complete text ===")
        main_text = "\n\n".join([overview, full_text, "## 研究评述\n\n" + critique])
        complete_text = "\n\n".join([
            overview, full_text,
            "## 研究评述\n\n" + critique,
            "## 参考文献\n\n" + "\n\n".join(references)
        ]).strip()

        review_text["main_text"] = main_text
        review_text["complete_text"] = complete_text
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Complete text assembled and saved")

    def generate_review_structure_1(
        self,
        review_data: Dict,
        main_topic: str,
        review_text: Dict
    ):
        """
        Generate review with language-separated structure (Chinese/English sections).

        This method generates the review incrementally and saves each step to MongoDB.
        """
        print("\n=== Step 3: Splitting by language ===")
        zh_lit, en_lit = self._split_by_language(review_data)

        print("\n=== Step 4: Generating paragraphs (Chinese/English) ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            zh_future = executor.submit(self.review_agent.generate_paragraphs_only, zh_lit)
            en_future = executor.submit(self.review_agent.generate_paragraphs_only, en_lit)
            para_zh = zh_future.result()
            para_en = en_future.result()

        review_text["paragraph_results_zh"] = para_zh
        review_text["paragraph_results_en"] = para_en
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Paragraphs saved")

        print("\n=== Step 5: Generating tables ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            zh_future = executor.submit(self.table_agent.generate_tables_only, para_zh)
            en_future = executor.submit(self.table_agent.generate_tables_only, para_en)
            table_zh = zh_future.result()
            table_en = en_future.result()

        review_text["table_results_zh"] = table_zh
        review_text["table_results_en"] = table_en
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Tables saved")

        print("\n=== Step 6: Assembling topics ===")
        topics_zh = {}
        for i, (topic, types) in enumerate(zh_lit.items()):
            topics_zh[topic] = self._assemble_topic_text(topic, i, types, para_zh, table_zh)

        topics_en = {}
        for i, (topic, types) in enumerate(en_lit.items()):
            topics_en[topic] = self._assemble_topic_text(topic, i, types, para_en, table_en)

        full_text_zh = "\n\n".join(topics_zh.values())
        full_text_en = "\n\n".join(topics_en.values())
        combined_text = "# 一、国内相关研究\n\n" + full_text_zh + "\n\n# 二、国外相关研究\n\n" + full_text_en

        review_text["topics"] = {"topics_zh": topics_zh, "topics_en": topics_en}
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Topics assembled and saved")

        print("\n=== Step 7: Generating overview and critique ===")
        overview, critique = self.overview_agent.generate_overview_and_critique(
            combined_text, main_topic
        )
        review_text["overview"] = overview
        review_text["critique"] = critique
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Overview and critique saved")

        print("\n=== Step 8: Extracting citations ===")
        refs_zh, urls_zh = self._extract_cited_references_from_topics(zh_lit, topics_zh)
        refs_en, urls_en = self._extract_cited_references_from_topics(en_lit, topics_en)

        references = refs_zh + refs_en
        pdf_urls = urls_zh + urls_en

        review_text["references"] = references
        review_text["pdf_urls"] = pdf_urls
        self._save_to_mongodb({"review_text": review_text})
        print("✓ References extracted and saved")

        print("\n=== Step 9: Assembling complete text ===")
        main_text = "\n\n".join([overview, combined_text, "# 研究评述\n\n" + critique])
        complete_text = "\n\n".join([
            overview, combined_text,
            "# 研究评述\n\n" + critique,
            "# 参考文献\n\n" + "\n\n".join(references)
        ]).strip()

        review_text["main_text"] = main_text
        review_text["complete_text"] = complete_text
        self._save_to_mongodb({"review_text": review_text})
        print("✓ Complete text assembled and saved")

    def generate_review(
        self,
        main_topic: str,
        language: Optional[str] = None,
        chinese_weight: Optional[float] = None
    ) -> Dict:
        """
        Main entry point to generate a complete literature review.

        Args:
            main_topic: The main research topic
            language: Target language (None for Chinese, "英文" for English)
            chinese_weight: Target proportion of Chinese literature (0-1)

        Returns:
            Dictionary containing the complete review data
        """
        try:
            print(f"\n{'='*60}")
            print(f"Starting Literature Review Generation")
            print(f"Topic: {main_topic}")
            print(f"Language: {language or '中文'}")
            print(f"Chinese Weight: {chinese_weight}")
            print(f"Structure: {self.structure}")
            print(f"{'='*60}\n")

            review_text = {"main_topic": main_topic}

            # Initial state
            self._update_state(2, {
                "query": main_topic,
                "structure": self.structure,
                "mode": self.mode
            })

            # Step 1: Topic Planning
            print("=== Step 1: Topic Planning ===")
            subtopics = self.topic_agent.decompose_topic(main_topic)
            review_text["subtopics"] = subtopics
            self._save_to_mongodb({"review_text": review_text})
            print(f"✓ Generated {len(subtopics)} sub-topics and saved")

            # Step 2: Literature Collection
            print("\n=== Step 2: Literature Collection ===")
            if self.structure == 1:
                literature_data = self.literature_agent.collect_literature(
                    subtopics=subtopics,
                    language=language,
                    chinese_weight=0.5
                )
            else:
                literature_data = self.literature_agent.collect_literature(
                    subtopics=subtopics,
                    language=language,
                    chinese_weight=chinese_weight
                )

            # Check if empty
            if self._check_empty_literature(literature_data):
                review_text["complete_text"] = "在社会科学文献库中未检索到相关文献，生成失败。"
                review_text["references"] = []
                self._update_state(1, {"review_text": review_text})
                print("✗ No literature found, generation terminated")
                return review_text

            self._save_to_mongodb({"literature_data": literature_data})
            print("✓ Literature collected and saved")

            # Steps 3-8/9: Review Generation
            if self.structure == 1:
                self.generate_review_structure_1(literature_data, main_topic, review_text)
            else:
                self.generate_review_structure_0(literature_data, main_topic, review_text)

            # Step 9/10: Translation (if needed)
            if language == "英文":
                print("\n=== Translation ===")
                translated = self._parallel_translate_fields(review_text)
                review_text["complete_text"] = translated
                self._save_to_mongodb({"review_text": review_text})
                print("✓ Translation completed and saved")
            else:
                # Check Chinese ratio if specified
                if chinese_weight is not None:
                    ratio = self._calculate_chinese_ratio(review_text["references"])
                    print(f"\nChinese literature ratio: {ratio:.2%}")

                    if ratio < chinese_weight:
                        review_text["chinese_paper_check"] = \
                            f"中文文献不足{chinese_weight:.2%}，已自动补充英文文献完成综述。" \
                            f"当前中文文献比例为 {ratio:.2%}。"
                        self._save_to_mongodb({"review_text": review_text})

            # Final success state
            self._update_state(1, {"review_text": review_text})

            print(f"\n{'='*60}")
            print("Literature Review Generation Completed Successfully")
            print(f"Total references: {len(review_text.get('references', []))}")
            print(f"{'='*60}\n")

            return review_text

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR: {str(e)}")
            print(f"{'='*60}\n")

            self._update_state(0, {"error": str(e)})
            raise


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = LiteratureReviewSystem(
        user_id="test_user",
        task_id="test_task_001",
        kb_id="social_science_kb",
        mode=0,
        structure=0
    )

    # Generate review
    result = system.generate_review(
        main_topic="企业数字化转型对技术创新的影响研究",
        language=None,
        chinese_weight=None
    )

    print("Review generated successfully!")
    print(f"Subtopics: {result['subtopics']}")
    print(f"Total references: {len(result.get('references', []))}")