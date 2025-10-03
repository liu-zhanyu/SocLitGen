"""
Literature Collection Module

This module retrieves literature relevant to each sub-topic through a four-step cyclic
retrieval mechanism:

1. Planning search term strategies based on LLMs (intelligent keyword generation)
2. Hybrid retrieval coarse screening (vector + inverted index fusion)
3. LLM fine screening with Chain of Thought prompts (relevance judgment)
4. Iterative retrieval when insufficient results (dynamic query expansion)

The module breaks through limitations of traditional naive queries by adopting hybrid
retrieval architecture that fuses vector indexes with inverted indexes.
"""

import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import jieba

from components.llm_call import handler
from components.client import *


class LiteratureCollectionAgent:
    """
    Literature Collection Agent for retrieving and filtering relevant academic papers.

    Uses a sophisticated four-step retrieval mechanism with hybrid search and LLM-based
    relevance filtering to ensure comprehensive and high-quality literature collection.
    """

    def __init__(self, kb_id: str, mode: Optional[int] = None):
        """
        Initialize the Literature Collection Agent.

        Args:
            kb_id: Knowledge base ID for literature retrieval
            mode: Operation mode (1 for long-form mode requiring more documents)
        """
        self.name = "Literature Collection Agent"
        self.kb_id = kb_id
        self.mode = mode

    # ==================== Step 1: Search Term Planning ====================

    def generate_optimal_query(self, subtopic: str, content_type: str) -> str:
        """
        Step 1: Generate optimal search queries using LLM-driven keyword planning.

        This implements intelligent keyword generation technology to automatically generate
        diversified search term combinations for each sub-topic. The strategy draws from
        successful experiences in LitLLM and UR3WG frameworks.

        Args:
            subtopic: The research sub-topic
            content_type: Type of content to search for (concept/theory/findings)

        Returns:
            Optimized query string with 5-7 keywords
        """
        try:
            # Build content-specific prompt
            if content_type == "concept":
                search_focus = "核心概念和定义"
            elif content_type == "theory":
                search_focus = "理论基础和模型"
            else:  # findings
                search_focus = "研究发现和实证结果"

            prompt = f"""
我需要为文献检索生成最佳的查询词。我的目标是查找有关"{subtopic}"的{search_focus}。
请生成一个简短、精确的查询字符串，包含5-7个关键词，用空格分隔。
只返回查询字符串，不要有其他任何解释或额外文字。
"""

            # Try LLM-based query generation
            try:
                response = handler.call_llm(
                    provider="openai",
                    prompt=prompt,
                    model="gpt-4o-mini",
                    max_tokens=20
                )

                if "API call failed" not in str(response):
                    return response.strip()

            except Exception as e:
                print(f"{self.name}: LLM query generation failed - {e}")

            # Fallback to jieba segmentation
            seg_list = jieba.cut(subtopic, cut_all=False)
            segmented_query = " ".join(seg_list)
            return segmented_query

        except Exception as e:
            print(f"{self.name}: Error in query generation - {e}")
            # Final fallback with simple keyword combination
            if content_type == "concept":
                return f"{subtopic} 定义 概念 特征"
            elif content_type == "theory":
                return f"{subtopic} 理论 模型 框架"
            else:
                return f"{subtopic} 研究发现 实证研究 结论"

    # ==================== Step 2: Hybrid Retrieval ====================

    def hybrid_retrieval(
            self,
            query: str,
            chunk_type: List[str],
            top_k: int = 30,
            language: Optional[str] = None
    ) -> List[Dict]:
        """
        Step 2: Execute hybrid retrieval combining vector and inverted indexes.

        This breaks through limitations of traditional single-mode retrieval by fusing
        text matching precision with semantic association depth mining capabilities.

        Args:
            query: Search query string
            chunk_type: Types of document chunks to search
            top_k: Number of top results to return
            language: Optional language filter (中文/英文)

        Returns:
            List of document dictionaries with metadata
        """
        search_results = es_service.search_documents(
            query=query,
            kb_id=self.kb_id,
            chunk_type=chunk_type,
            top_k=top_k,
            language=language
        )

        return search_results

    # ==================== Step 3: LLM Fine Screening ====================

    def is_relevant_to_subtopic(
            self,
            title: str,
            abstract: str,
            subtopic: str
    ) -> int:
        """
        Step 3: Use LLM with Chain of Thought prompts to judge literature relevance.

        This hierarchical screening strategy employs two-step reasoning:
        1. Concept judgment - Does literature discuss the core concept?
        2. Direction judgment - Does literature align with the specific research direction?

        This precise screening ensures only truly relevant literature is collected.

        Args:
            title: Literature title
            abstract: Literature abstract
            subtopic: Target sub-topic

        Returns:
            1 if relevant, 0 if not relevant
        """
        prompt = f"""
以下文献是检索系统为"{subtopic}"这个研究议题检索到的一篇文献的<文献信息>。
请根据其中的文献标题和文献摘要判断该文献是否真正属于"{subtopic}"这个研究议题。

<文献信息>
文献标题：{title}
文献摘要：{abstract}
</文献信息>

你需要严格遵循我提供的<思维步骤>来做判断。

<思维步骤>
第一步，先做概念判断：判断提供的文献是否围绕给定研究议题中的*核心概念*展开讨论。例如，对于"企业数字化转型的效应研究"这一议题来说，这一步首先需要判断文献内容是否围绕"企业数字化转型"这个关键概念展开讨论。若是，则进入下一步判断；若不是，则直接判定为"不属于"；
第二步，再做方向判断：判断提供的文献是否在*方向*上与给定研究议题一致。例如，对于"企业数字化转型的效应研究"这一议题来说，第一步确定了一篇文献是围绕"企业数字化转型"这个关键概念展开讨论，那么这一步则需要进一步确定这篇文献是不是属于"效应研究"的范畴。若是，则代表提供的文献属于给定的研究议题，若不是，则判定为"不属于"。
</思维步骤>

判断结果只返回0或1，1表示属于，0表示不属于。不要返回任何其他内容，也无需解释理由。

判断结果："""

        try:
            # Try Zhipu AI first
            try:
                result = handler.call_llm(
                    provider="zhipuai",
                    prompt=prompt,
                    model="glm-4-air-250414",
                    max_tokens=100,
                    temperature=0.1
                )
                if "API call failed" not in str(result):
                    return 1 if "1" in result else 0

            except Exception:
                pass

            # Fallback to Doubao
            try:
                result = handler.call_llm(
                    provider="ark",
                    prompt=prompt,
                    model="doubao-1-5-lite-32k-250115",
                    max_tokens=100,
                    temperature=0.1
                )
                if "API call failed" not in str(result):
                    return 1 if "1" in result else 0

            except Exception:
                pass

            # Final fallback to OpenAI
            result = handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=100,
                temperature=0.1
            )

            return 1 if "1" in result else 0

        except Exception as e:
            print(f"{self.name}: LLM relevance judgment failed - {e}")
            # Default to keeping the document if API fails
            return 1

    # ==================== Step 4: Iterative Retrieval ====================

    def collect_single_subtopic_literature(
            self,
            subtopic: str,
            min_docs_per_category: int = 15,
            language: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Collect literature for a single sub-topic using the four-step cyclic mechanism.

        Step 4: Automatically initiates iterative retrieval when literature is insufficient.
        The mechanism innovatively integrates successfully retrieved literature titles into
        original search terms, generating enhanced query combinations.

        Args:
            subtopic: The research sub-topic
            min_docs_per_category: Minimum documents required per content type
            language: Optional language filter

        Returns:
            Dictionary mapping content types to lists of relevant documents
        """
        print(f"{self.name}: Collecting literature for sub-topic '{subtopic}'...")

        content_type_list = ["findings"]
        subtopic_results = {content_type: [] for content_type in content_type_list}

        # Global deduplication across all content types
        all_collected_docs_keys = set()

        for content_type in content_type_list:
            print(f"{self.name}: Collecting {content_type} literature for '{subtopic}'...")

            chunk_type = "summary" if content_type == "findings" else f"{content_type}_answer"

            # Step 1: Generate initial query
            query = self.generate_optimal_query(subtopic, content_type)
            all_relevant_docs = []
            used_queries = set([query])
            first_batch_results = []

            max_iterations = 3

            for iteration in range(max_iterations):
                # Step 2: Hybrid retrieval
                search_results = self.hybrid_retrieval(
                    query=query,
                    chunk_type=[chunk_type],
                    top_k=30,
                    language=language
                )

                # Save first batch for fallback
                if iteration == 0:
                    first_batch_results = search_results.copy()

                # Step 3: LLM fine screening with parallel processing
                relevant_docs_lock = threading.Lock()

                def process_document(doc):
                    """Process single document with relevance check and deduplication."""
                    title = doc.get("title", "")
                    authors = doc.get("authors", "")
                    abstract = doc.get("abstract", "")

                    doc_key = (title, authors)

                    with relevant_docs_lock:
                        # Check if already included in current results
                        doc_already_included = any(
                            existing_doc.get("title") == title
                            for existing_doc in all_relevant_docs
                        )
                        # Check if already collected across all content types
                        doc_already_collected = doc_key in all_collected_docs_keys

                    # Only process if not already included
                    if not doc_already_included and not doc_already_collected:
                        relevance_score = self.is_relevant_to_subtopic(
                            title, abstract, subtopic
                        )
                        if relevance_score == 1:
                            with relevant_docs_lock:
                                all_relevant_docs.append(doc)
                                all_collected_docs_keys.add(doc_key)
                                return True
                    return False

                # Parallel processing of documents
                with ThreadPoolExecutor(max_workers=30) as executor:
                    future_to_doc = {
                        executor.submit(process_document, doc): doc
                        for doc in search_results
                    }
                    for future in as_completed(future_to_doc):
                        try:
                            future.result()
                        except Exception as exc:
                            print(f"{self.name}: Document processing error - {exc}")

                # Check if sufficient documents collected
                if len(all_relevant_docs) >= min_docs_per_category:
                    print(f"{self.name}: Sufficient {content_type} literature collected "
                          f"({len(all_relevant_docs)} documents)")
                    break

                # Step 4: Iterative retrieval - prepare next query
                if iteration < max_iterations - 1:
                    new_query_title = ""

                    # If no relevant docs found, use first search result title
                    if len(all_relevant_docs) == 0:
                        if search_results:
                            new_query_title = search_results[0].get("title", "")
                            print(f"{self.name}: No relevant docs in iteration, "
                                  f"using first result title: {new_query_title}")
                    else:
                        new_query_title = all_relevant_docs[-1].get("title", "")

                    # Generate enhanced query by combining subtopic with literature title
                    if new_query_title:
                        new_query = f"{subtopic} {new_query_title}"
                        if new_query not in used_queries:
                            query = new_query
                            used_queries.add(query)
                            print(f"{self.name}: Collected {len(all_relevant_docs)} docs, "
                                  f"insufficient. Enhanced query: {query}")
                        else:
                            print(f"{self.name}: Cannot generate new query, "
                                  f"{len(all_relevant_docs)} docs collected")
                            break
                    else:
                        print(f"{self.name}: Cannot generate valid query, stopping search")
                        break

            # Fallback: Use first batch if still insufficient after iterations
            if len(all_relevant_docs) == 0 and first_batch_results:
                print(f"{self.name}: After {max_iterations} iterations, still insufficient. "
                      f"Using first {min_docs_per_category} from initial results")

                filtered_first_batch = []
                for doc in first_batch_results:
                    doc_key = (doc.get("title", ""), doc.get("authors", ""))
                    if doc_key not in all_collected_docs_keys:
                        filtered_first_batch.append(doc)
                        all_collected_docs_keys.add(doc_key)

                        if len(filtered_first_batch) >= min_docs_per_category:
                            break

                all_relevant_docs = filtered_first_batch[:min_docs_per_category]

            subtopic_results[content_type] = all_relevant_docs

        return subtopic_results

    # ==================== Language-Specific Collection ====================

    def collect_single_subtopic_literature_by_weight(
            self,
            subtopic: str,
            min_docs_per_category: int = 15,
            chinese_weight: float = 0.5
    ) -> Dict[str, List[Dict]]:
        """
        Collect literature with specified language distribution.

        This method ensures a target ratio of Chinese to English literature,
        collecting each language separately then combining results.

        Args:
            subtopic: The research sub-topic
            min_docs_per_category: Minimum total documents per content type
            chinese_weight: Target proportion of Chinese literature (0-1)

        Returns:
            Dictionary mapping content types to lists of documents
        """
        print(f"{self.name}: Collecting literature for sub-topic '{subtopic}'...")
        print(f"Language ratio mode: Chinese literature ratio = {chinese_weight}")

        content_type_list = ["findings"]
        subtopic_results = {content_type: [] for content_type in content_type_list}

        for content_type in content_type_list:
            print(f"{self.name}: Collecting {content_type} literature for '{subtopic}'...")
            chunk_type = "summary" if content_type == "findings" else f"{content_type}_answer"

            # Calculate target Chinese document count
            min_chinese_docs = int(min_docs_per_category * chinese_weight)
            print(f"{self.name}: Target - Chinese: {min_chinese_docs}, "
                  f"English: {min_docs_per_category - min_chinese_docs}")

            # First round: Collect Chinese literature
            chinese_docs = self._collect_docs_by_language(
                subtopic, content_type, chunk_type,
                min_chinese_docs, "中文", max_iterations=5
            )

            # Second round: Collect English literature
            remaining_docs_needed = min_docs_per_category - len(chinese_docs)
            english_docs = []
            if remaining_docs_needed > 0:
                english_docs = self._collect_docs_by_language(
                    subtopic, content_type, chunk_type,
                    remaining_docs_needed, "英文", max_iterations=5
                )

            # Combine results
            all_relevant_docs = chinese_docs + english_docs
            subtopic_results[content_type] = all_relevant_docs

            print(f"{self.name}: Collected {len(chinese_docs)} Chinese and "
                  f"{len(english_docs)} English documents, total: {len(all_relevant_docs)}")

        return subtopic_results

    def _collect_docs_by_language(
            self,
            subtopic: str,
            content_type: str,
            chunk_type: str,
            min_docs: int,
            language: str,
            max_iterations: int = 5
    ) -> List[Dict]:
        """
        Helper method to collect documents for a specific language.

        Implements the same four-step mechanism but filtered by language.
        """
        print(f"{self.name}: Collecting {language} {content_type} literature for '{subtopic}'...")

        query = self.generate_optimal_query(subtopic, content_type)
        all_relevant_docs = []
        used_queries = set([query])
        first_batch_results = []

        for iteration in range(max_iterations):
            search_results = self.hybrid_retrieval(
                query=query,
                chunk_type=[chunk_type],
                top_k=30,
                language=language
            )

            if iteration == 0:
                first_batch_results = search_results.copy()

            relevant_docs_lock = threading.Lock()

            def process_document(doc):
                title = doc.get("title", "")
                abstract = doc.get("abstract", "")

                with relevant_docs_lock:
                    doc_already_included = any(
                        existing_doc.get("title") == title
                        for existing_doc in all_relevant_docs
                    )

                if not doc_already_included:
                    relevance_score = self.is_relevant_to_subtopic(title, abstract, subtopic)
                    if relevance_score == 1:
                        with relevant_docs_lock:
                            if len(all_relevant_docs) < min_docs:
                                all_relevant_docs.append(doc)
                            return True
                return False

            with ThreadPoolExecutor(max_workers=30) as executor:
                future_to_doc = {
                    executor.submit(process_document, doc): doc
                    for doc in search_results
                }
                for future in as_completed(future_to_doc):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"{self.name}: Document processing error - {exc}")

            if len(all_relevant_docs) >= min_docs:
                print(f"{self.name}: Sufficient {language} {content_type} literature "
                      f"collected ({len(all_relevant_docs)} documents)")
                break

            # Iterative query enhancement
            if iteration < max_iterations - 1:
                new_query_title = ""

                if len(all_relevant_docs) == 0:
                    if search_results:
                        new_query_title = search_results[0].get("title", "")
                else:
                    new_query_title = all_relevant_docs[-1].get("title", "")

                if new_query_title:
                    new_query = f"{subtopic} {new_query_title}"
                    if new_query not in used_queries:
                        query = new_query
                        used_queries.add(query)
                        print(f"{self.name}: Collected {len(all_relevant_docs)} {language} docs, "
                              f"insufficient. Enhanced query: {query}")
                    else:
                        break
                else:
                    break

        # Fallback to first batch
        if len(all_relevant_docs) == 0 and first_batch_results:
            print(f"{self.name}: After {max_iterations} iterations, using first batch")
            all_relevant_docs = first_batch_results[:min_docs]

        return all_relevant_docs

    # ==================== Parallel Collection Orchestration ====================

    def collect_literature(
            self,
            subtopics: List[str],
            language: Optional[str] = None,
            chinese_weight: Optional[float] = None
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Collect literature for all sub-topics in parallel.

        This is the main entry point that orchestrates parallel literature collection
        across all sub-topics, utilizing multi-threading for efficiency.

        Args:
            subtopics: List of research sub-topics
            language: Optional language filter
            chinese_weight: Optional target proportion of Chinese literature

        Returns:
            Nested dictionary: {subtopic: {content_type: [documents]}}
        """
        print(f"{self.name}: Starting literature collection for knowledge base: {self.kb_id}")

        def process_subtopic_task(subtopic):
            """Task function for processing a single sub-topic."""
            try:
                if chinese_weight is None:
                    # Standard collection without language ratio
                    if self.mode == 1:
                        print(f"Long-form mode - Processing sub-topic: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature(
                            subtopic=subtopic,
                            min_docs_per_category=30,
                            language=language
                        )
                    else:
                        print(f"Standard mode - Processing sub-topic: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature(
                            subtopic=subtopic,
                            language=language
                        )
                else:
                    # Collection with specified language ratio
                    if self.mode == 1:
                        print(f"Long-form mode - Processing sub-topic: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature_by_weight(
                            subtopic=subtopic,
                            min_docs_per_category=30,
                            chinese_weight=chinese_weight
                        )
                    else:
                        print(f"Standard mode - Processing sub-topic: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature_by_weight(
                            subtopic=subtopic,
                            chinese_weight=chinese_weight
                        )

                print(f"Completed literature collection for sub-topic '{subtopic}'")
                return (subtopic, subtopic_result)

            except Exception as e:
                print(f"Error processing sub-topic '{subtopic}': {str(e)}")
                return (subtopic, {})

        # Parallel processing with thread pool
        results = {}
        max_workers = min(len(subtopics), 10)  # Limit to 10 threads

        print(f"Launching {len(subtopics)} literature collection tasks "
              f"with {max_workers} threads...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_subtopic = {
                executor.submit(process_subtopic_task, subtopic): subtopic
                for subtopic in subtopics
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_subtopic):
                subtopic = future_to_subtopic[future]
                try:
                    subtopic_name, subtopic_result = future.result()
                    results[subtopic_name] = subtopic_result
                except Exception as e:
                    print(f"Task execution failed for sub-topic '{subtopic}': {str(e)}")
                    results[subtopic] = {}

        print(f"All literature collection completed, processed {len(results)} sub-topics")
        return results