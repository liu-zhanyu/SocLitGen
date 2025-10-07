"""
Review Generation Module

This module generates literature reviews for each sub-topic by organizing literature
summaries into logically structured content. It adopts parallelized generation strategies
to simultaneously develop content for each sub-topic.

Key strategy: Organizes literature around competing theoretical perspectives using Chain
of Thought (CoT) reasoning to systematically categorize literature into multiple competing
categories, representing different theoretical perspectives and conceptual debates.

The module strictly retains "Author (Year)" identifiers from literature summaries to ensure
subsequent precise literature matching and citation generation.
"""

import re
import concurrent.futures
from typing import List, Dict, Any, Callable
from components.llm_call import handler


class ReviewGenerationAgent:
    """
    Review Generation Agent for creating structured academic literature reviews.

    Uses parallelized generation with Chain of Thought reasoning to categorize and
    synthesize literature into coherent thematic groups representing different
    theoretical perspectives.
    """

    def __init__(self, mode: int = None):
        """
        Initialize the Review Generation Agent.

        Args:
            mode: Operation mode (1 for long-form mode requiring extensive content)
        """
        self.name = "Review Generation Agent"
        self.mode = mode

    # ==================== APA Formatting Utilities ====================

    def transform_list_to_apa_format(self, data_list: List[Dict]) -> List[Dict]:
        """
        Transform literature data into APA citation format.

        Creates both in-text citations (Author, Year) and full reference list entries
        following APA 7th edition guidelines.

        Args:
            data_list: List of literature dictionaries with metadata

        Returns:
            List of dictionaries with APA-formatted citations and references
        """
        result_list = []

        for item in data_list:
            result_item = {}

            # Extract metadata
            authors = item.get('authors', '')
            year = item.get('year', '')
            title = item.get('title', '')
            journal = item.get('journal', '')
            content = item.get('content_with_weight', '')

            # Additional fields for full reference
            volume = item.get('vO', '')
            issue = item.get('issue', '')
            pages = item.get('page_range', '')
            pdf_url = item.get('pdf_url', '')

            # Process authors for citation
            author_list = authors.split(";")
            processed_authors = []

            for author in author_list:
                author = author.strip()
                if author:
                    # English format: Last, First -> keep only Last
                    if "," in author:
                        processed_authors.append(author.split(",")[0].strip())
                    else:
                        processed_authors.append(author)

            # Format in-text citation
            if len(processed_authors) == 0:
                formatted_author = ""
            elif len(processed_authors) == 1:
                formatted_author = processed_authors[0]
            elif len(processed_authors) == 2:
                formatted_author = f"{processed_authors[0]} & {processed_authors[1]}"
            else:  # 3+ authors: use "et al."
                formatted_author = f"{processed_authors[0]} et al."

            # Create APA in-text citation
            apa_citation = f"({formatted_author}, {year})"

            # Create content with citation
            content_with_citation = f"{content} {apa_citation}"

            # Create full APA reference
            if len(processed_authors) == 0:
                authors_for_reference = ""
            elif len(processed_authors) == 1:
                authors_for_reference = processed_authors[0]
            elif len(processed_authors) == 2:
                authors_for_reference = f"{processed_authors[0]} & {processed_authors[1]}"
            else:
                # For full references, list all authors
                authors_for_reference = ", ".join(processed_authors[:-1]) + ", & " + processed_authors[-1]

            # Build full reference
            apa_reference = f"{authors_for_reference}. ({year}). {title}. {journal}"

            if volume:
                apa_reference += f", {volume}"
                if issue:
                    apa_reference += f"({issue})"

            if pages:
                apa_reference += f", {pages}"

            apa_reference += "."
            apa_reference += f"[PDF_URL_START]{pdf_url}[PDF_URL_END]"

            # Store results
            result_item['apa_citation'] = apa_citation
            result_item['content_with_citation'] = content_with_citation
            result_item['original_content'] = content
            result_item['apa_reference'] = apa_reference

            result_list.append(result_item)

        return result_list

    # ==================== Paragraph Generation with CoT ====================

    def generate_paragraph(
            self,
            content_list: List[Dict],
            content_type: str,
            subtopic: str
    ) -> str:
        """
        Generate structured paragraph using Chain of Thought reasoning.

        This method organizes literature around competing theoretical perspectives by:
        1. Using CoT to categorize literature into thematic groups
        2. Systematically summarizing characteristics of each category
        3. Presenting research examples within each category

        Args:
            content_list: List of literature items with content
            content_type: Type of content (concept/theory/findings)
            subtopic: The research sub-topic

        Returns:
            Generated paragraph text with [CON_START] and [CON_END] markers
        """
        # Transform to APA format
        apa_formatted = self.transform_list_to_apa_format(content_list)

        if content_type == "concept":
            prompt = self._build_concept_prompt(apa_formatted, subtopic)
        elif content_type == "theory":
            prompt = self._build_theory_prompt(apa_formatted, subtopic)
        else:  # findings
            prompt = self._build_findings_prompt(apa_formatted, subtopic)

        # Generate with LLM fallback
        response = self._call_llm_for_generation(prompt)

        # Extract content between markers
        extracted_content = self._extract_content_markers(response)

        # For long-form mode with findings, expand if needed
        if self.mode == 1 and content_type == "findings" and len(response.strip()) < 2500:
            extracted_content = self._expand_findings_content(
                response.strip(), subtopic, apa_formatted
            )

        return extracted_content

    def _build_concept_prompt(self, apa_formatted: List[Dict], subtopic: str) -> str:
        """Build prompt for concept definition paragraphs."""
        combined_content = "\n\n".join(
            [str(item["content_with_citation"]).replace("\n", " ") for item in apa_formatted]
        )

        return f"""
你是一位文献综述写作专家。请基于<参考资料>，写一段关于"{subtopic}"的既有概念定义的中文段落。
在写作过程中，严格遵循<写作原则>，同时深刻参考<示例结果>中的语言风格，包括句式结构、句与句之间关系、措辞习惯。

<写作原则>

第一步，段落首句评价现有研究所用核心概念定义的多寡。
第二步，段落第二句开始呈现不同学者对于核心概念的定义。
第三步，段落最后一句或两句总结核心概念不同定义所包含的核心要点。

注意：
用[CON_START]和[CON_END]包裹结果。
保持严谨的学术风格，用一段流畅、逻辑清晰的话来阐述，请勿换行。
尽可能使用所有的文献资料，每份文献资料都来之不易。
尽可能保留文献材料原文中的细节，原文中的所有细节信息对于读者理解文献都至关重要。
参考资料中一行为一篇文献的材料，后面的（姓名，年份）为Citation，需在生成的正文中以合适的形式保留（见示例结果）。
作者姓名不用翻译。
</写作原则>

<示例结果>
[CON_START]目前，已有诸多文献对气候变化叙事这一概念做出定义。例如，Paschen & Ison (2014)提出，气候变化叙事是指以故事化方式讲述气候变化信息和政策，目的是获取各利益相关者信任并指导特定类型行动。Curran (2012)提出，气候变化叙事是指传播气候变化信息，以增进公众对气候变化政策与知识的理解。Bushell et al. (2017)提出，气候变化叙事是指以叙事传播方式讲述气候变化的未来结果，目的是促进各相关方进行战略规划以应对未来风险。Holden et al. (2021)提出，气候变化叙事是指以故事性和修辞性方式，使用场景、寓意、情节、角色等元素介绍气候变化缓解路径，以促使公众改变能源行为。Jepson (2019)提出，气候变化叙事是指讲述"环境现状、这种现状如何影响人类及人类需要做什么"的故事，从而以有意义和有说服力的方式建构环境议题。总体来看，气候变化叙事的定义总共涉及三大核心要点：一是利用叙事元素，二是围绕特定议题，三是达到特定目的。不同定义的差异主要体现在这三大核心要点上。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""

    def _build_theory_prompt(self, apa_formatted: List[Dict], subtopic: str) -> str:
        """Build prompt for theoretical perspectives paragraphs."""
        combined_content = "\n\n".join(
            [str(item["content_with_citation"]).replace("\n", " ") for item in apa_formatted]
        )

        return f"""
你是一位文献综述写作专家。请基于<参考资料>，写一段关于"{subtopic}"的既有理论视角的中文段落。
在写作过程中，严格遵循<写作原则>，同时深刻参考<示例结果>中的语言风格，包括句式结构、句与句之间关系、措辞习惯。

<写作原则>

第一步，段落首句概述现有研究所用理论视角的数量。
第二步，每个理论视角都采用两部分来论述：一是理论的内涵及观点，二是理论相关研究示例（若有多个研究用同一理论则提供多个示例）的概述。
第三步，段落最后一句或两句总结现有理论视角对于研究者的启示。

注意：
用[CON_START]和[CON_END]包裹结果。
保持严谨的学术风格，用一段流畅、逻辑清晰的话来阐述，请勿换行。
尽可能使用所有的文献资料，每份文献资料都来之不易。
尽可能保留文献材料原文中的细节，原文中的所有细节信息对于读者理解文献都至关重要。
参考资料中一行为一篇文献的材料，后面的（姓名，年份）为Citation，需在生成的正文中以合适的形式保留（见示例结果）。
作者姓名不用翻译。
</写作原则>

<示例结果>
[CON_START]既有数字政府的影响因素研究主要立足于五个理论视角。第一个理论是政策企业家理论，该理论强调特定个体在政策过程中的关键推动作用。例如，Mergel (2019)基于该理论分析和检验了机构领导人在推动政府数字化转型中的作用。第二个理论是TOE框架，该框架强调技术、组织和环境三大因素对创新采纳的综合影响。例如，Chen & Hsiao (2014)基于该理论分析和检验了技术基础设施、机构领导支持和法律法规环境对数字政府推进的影响。第三个理论是政策创新扩散理论，该理论关注政策如何在政府间传播。例如，Zhang et al. (2014)基于该理论，提出了一个囊括地理临近性和政治相似性等因素的分析框架来研究电子政务实践的区域扩散。第四个理论是制度理论，该理论探讨正式和非正式制度对组织行为的影响。例如，Luna-Reyes & Gil-Garcia (2011)基于制度理论，分析和验证了法律框架、组织结构和行政文化对数字政府项目的影响。第五个理论是资源依赖理论，该理论强调组织为获取关键资源而采取的战略行为。例如，Cordella & Willcocks (2010)基于该理论分析和验证了资源约束如何促使政府机构通过技术外包和战略联盟以维持数字服务高效运行。通过对数字政府影响因素研究的五大主要理论视角的梳理，我们可以看出这些理论从不同层面共同构建了对数字政府发展机制的系统性理解。这些理论视角启示研究者需要同时关注领导推动、组织能力、外部环境、制度约束和资源获取等多方面因素。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""

    def _build_findings_prompt(self, apa_formatted: List[Dict], subtopic: str) -> str:
        """Build prompt for research findings paragraphs with CoT categorization."""
        combined_content = "\n\n".join(
            [str(item["original_content"]).replace("\n", " ") for item in apa_formatted]
        )

        if self.mode == 1:
            # Long-form mode: allows line breaks and requires 2500+ characters
            return f"""
你是一位文献综述写作专家。请基于<参考资料>，写一段关于"{subtopic}"的既有研究内容的中文段落。
在写作过程中，严格遵循<写作原则>，同时深刻参考<示例结果>中的语言风格，包括句式结构、句与句之间关系、措辞习惯。

<写作原则>
第一步，先思考所有文献按照研究内容（所关注的因素或细分议题）可进一步分为多少类（无需输出文字）；这些类别必须为"{subtopic}"大议题之下，不能跳出这个议题的范围，要在命名上体现这一点。
第二步，段落首句概述现有研究按照研究内容来分可分为多少类。
第三步，随后每类研究都采用两部分来论述：一是介绍该类型研究的特点，二是呈现属于该类型的研究概述（可罗列多个）。

注意：
用[CON_START]和[CON_END]包裹结果。
文献分类通常在4类以下。
请输出尽可能多的文字，不要浪费<参考资料>的文字，文字越多越好，争取在2500个汉字以上。
内容可以按分类结果换行(请勿一篇文献一行,换行用两个换行符)，如[CON_START]第一类……。\n\n第二类……。\n\n第三类……。[CON_END]。
保持严谨的学术风格，用流畅、逻辑清晰的话来阐述。
尽可能使用所有的文献资料，每份文献资料都来之不易。
一定要保留文献材料原文中的细节（如方法、样本、数据等），原文中的所有细节信息对于读者理解文献都至关重要。
参考资料中一行为一篇文献的材料，保持文献在正文中的引用方式为句首的"姓名（年份）"，不要在句子后面重复出现（姓名，年份）。
作者姓名不用翻译。
</写作原则>

<示例结果>
[CON_START]现有治理者公众信任的影响因素研究按照所关注因素的差异可以分为三类。\n\n第一类研究关注的是治理条件对治理者公众信任的影响，前者主要包括治理过程、治理表现、治理情境等因素。例如，在治理过程上，Wang和van Wart（2007）采用路径分析方法分析了2000年美国城市政府层面的调查数据，结果显示政府治理过程的公众参与程度与公众信任水平正相关。在治理表现上，Seyd（2015）采用结构方程模型分析了2008年英国公民对政治人物的信任度数据，结果显示信任度主要由政治人物实际表现决定。在治理情境上，Houston等（2016）采用了多层二元Logit模型分析了2006年国际社会调查项目（ISSP）的21个国家样本数据，结果显示宗教多样性越低，公众对公务员的信任度越高。\n\n第二类研究关注的是沟通特征对治理者公众信任的影响，前者包括沟通程度、沟通策略等因素。例如，在沟通程度上，Park等（2016）采用结构方程模型分析了2012年韩国公民与政府互动的Twitter数据，结果显示政府领导与公民的沟通可以增加他们的政府信任。在沟通策略上，Alon-Barkat（2020）以环境政策为场景，基于以色列公民样本开展了随机调查实验，结果显示，包含真实象征元素（如标志、颜色和名人代言）的沟通可以增加公民对政策的信任。\n\n第三类研究关注的是个体特征对治理者公众信任的影响，前者包括社会身份、认知行为等因素。例如，在社会身份上，LeBas（2020）采用线性回归分析了2010年尼日利亚11个城市的问卷调查数据，研究显示少数族裔对当地官员的信任度显著更低。在认知行为上，Mizrahi 等（2021）采用多层线性分析了2018年以色列公民的代表性抽样调查数据，结果显示公众对公共部门的信任程度与人们对紧急情况的恐惧程度相关。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""
        else:
            # Standard mode: no line breaks, single paragraph
            return f"""
你是一位文献综述写作专家。请基于<参考资料>，写一段关于"{subtopic}"的既有研究内容的段落。
在写作过程中，严格遵循<写作原则>，同时深刻参考<示例结果>中的语言风格，包括句式结构、句与句之间关系、措辞习惯。

<写作原则>
第一步，先思考所有文献按照研究内容（所关注的因素或细分议题）可进一步分为多少类（无需输出文字）；这些类别必须为"{subtopic}"大议题之下，不能跳出这个议题的范围，要在命名上体现这一点。
第二步，段落首句概述现有研究按照研究内容来分可分为多少类。
第三步，随后每类研究都采用两部分来论述：一是介绍该类型研究的特点，二是呈现属于该类型的研究概述（可罗列多个）。

注意：
用[CON_START]和[CON_END]包裹结果。
文献分类通常在4类以下。
保持严谨的学术风格，用一段流畅、逻辑清晰的话来阐述，请勿换行。
尽可能使用所有的文献资料，每份文献资料都来之不易。
尽可能保留文献材料原文中的细节，原文中的所有细节信息对于读者理解文献都至关重要。
参考资料中一行为一篇文献的材料，保持文献在正文中的引用方式为句首的"姓名（年份）"，不要在句子后面重复出现（姓名，年份）。
</写作原则>

<示例结果>
[CON_START]现有治理者公众信任的影响因素研究按照所关注因素的差异可以分为三类。第一类研究关注的是治理条件对治理者公众信任的影响，前者主要包括治理过程、治理表现、治理情境等因素。例如，在治理过程上，Wang和van Wart（2007）采用路径分析方法分析了2000年美国城市政府层面的调查数据，结果显示政府治理过程的公众参与程度与公众信任水平正相关。在治理表现上，Seyd（2015）采用结构方程模型分析了2008年英国公民对政治人物的信任度数据，结果显示信任度主要由政治人物实际表现决定。在治理情境上，Houston等（2016）采用了多层二元Logit模型分析了2006年国际社会调查项目（ISSP）的21个国家样本数据，结果显示宗教多样性越低，公众对公务员的信任度越高。第二类研究关注的是沟通特征对治理者公众信任的影响，前者包括沟通程度、沟通策略等因素。例如，在沟通程度上，Park等（2016）采用结构方程模型分析了2012年韩国公民与政府互动的Twitter数据，结果显示政府领导与公民的沟通可以增加他们的政府信任。在沟通策略上，Alon-Barkat（2020）以环境政策为场景，基于以色列公民样本开展了随机调查实验，结果显示，包含真实象征元素（如标志、颜色和名人代言）的沟通可以增加公民对政策的信任。第三类研究关注的是个体特征对治理者公众信任的影响，前者包括社会身份、认知行为等因素。例如，在社会身份上，LeBas（2020）采用线性回归分析了2010年尼日利亚11个城市的问卷调查数据，研究显示少数族裔对当地官员的信任度显著更低。在认知行为上，Mizrahi 等（2021）采用多层线性分析了2018年以色列公民的代表性抽样调查数据，结果显示公众对公共部门的信任程度与人们对紧急情况的恐惧程度相关。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""

    def _call_llm_for_generation(self, prompt: str) -> str:
        """Call LLM for paragraph generation with provider fallback."""
        # Try Zhipu AI first
        try:
            response = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4.5-air",
                max_tokens=8000
            )
            if "API call failed" not in str(response):
                return response
        except Exception:
            pass

        # Try Doubao
        try:
            response = handler.call_llm(
                provider="ark",
                prompt=prompt,
                model="doubao-1-5-lite-32k-250115",
                max_tokens=8000
            )
            if "API call failed" not in str(response):
                return response
        except Exception:
            pass

        # Try SiliconFlow
        try:
            response = handler.call_llm(
                provider="siliconflow",
                prompt=prompt,
                model="THUDM/GLM-4-32B-0414",
                max_tokens=8000
            )
            if "API call failed" not in str(response):
                return response
        except Exception:
            pass

        # Final fallback to OpenAI
        return handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o",
            max_tokens=8000
        )

    def _extract_content_markers(self, response: str) -> str:
        """Extract content between [CON_START] and [CON_END] markers."""
        pattern = r'\[CON_START\](.*?)\[CON_END\]'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            extracted_content = match.group(1).strip()
            if self.mode == 1:
                extracted_content = extracted_content.replace("\n", "")
        else:
            extracted_content = response.replace("[CON_START]", "").replace("[CON_END]", "").strip()
            if self.mode == 1:
                extracted_content = extracted_content.replace("\n", "")

        return extracted_content

    def _expand_findings_content(
            self,
            current_content: str,
            subtopic: str,
            apa_formatted: List[Dict]
    ) -> str:
        """Expand findings content to meet minimum length requirement (2500 characters)."""
        combined_content = "\n\n".join(
            [str(item["original_content"]).replace("\n", " ") for item in apa_formatted]
        )

        prompt = f"""
你是一位文献综述写作专家。现在你的学生写了一段关于"{subtopic}"的既有研究内容的段落，即<现有段落>。
但是你对它不太满意，因为它的长度小于了2500个汉字。
请你结合<参考资料>扩充它到超过2500个汉字，尽可能多地扩充。
要求严格遵循<扩充原则>。

<现有段落>
{current_content}
</现有段落>

<扩充原则>
第一步，先判断<现有段落>中的内容有哪些仍不够具体（不用输出内容）。
第二步，围绕不够具体的内容，结合<参考资料>的内容，对不具体的内容做进一步的补充。

注意：
用[CON_START]和[CON_END]包裹扩写结果。
请勿重复改变文献数量，仅对文字做补充。
请勿通过重复呈现文字来达到扩充目的。
不要改变原有逻辑框架，尤其是文献的分类。
保持严谨的学术风格，用流畅、逻辑清晰的话来阐述。
扩写后的内容可以按分类结果换行(请勿一篇文献一行,换行用两个换行符)，如[CON_START]第一类……。\n\n第二类……。\n\n第三类……。[CON_END]。
保持文献在正文中的引用方式为句首的"姓名（年份）"，不要在句子后面重复出现（姓名，年份）。
</扩充原则>

<参考资料>
{combined_content}
</参考资料>

扩写结果："""

        # Call LLM for expansion
        try:
            response = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4.5-air",
                max_tokens=8000
            )
            if "API call failed" not in str(response):
                return self._extract_content_markers(response)
        except Exception:
            pass

        try:
            response = handler.call_llm(
                provider="siliconflow",
                prompt=prompt,
                model="THUDM/GLM-4-32B-0414",
                max_tokens=8000
            )
            if "API call failed" not in str(response):
                return self._extract_content_markers(response)
        except Exception:
            pass

        response = handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=8000
        )

        return self._extract_content_markers(response)

    # ==================== Paragraph Refinement ====================

    def refine_paragraph(
            self,
            paragraph_text: str,
            previous_text: str = "",
            following_text: str = "",
            topic: str = ""
    ) -> str:
        """
        Refine a paragraph to improve coherence with surrounding context.

        This enhances the logical flow between paragraphs and ensures smooth
        transitions throughout the review.

        Args:
            paragraph_text: The paragraph to refine
            previous_text: Text of the previous paragraph (empty if first)
            following_text: Text of the following paragraph (empty if last)
            topic: The main topic for context

        Returns:
            Refined paragraph text
        """
        if not previous_text:
            previous_text = "待润色段落为第一段，无上一段"

        if not following_text:
            following_text = "待润色段落为最后一段，无下一段"

        prompt = f'''
我正在撰写一篇关于"{topic}"的文献综述。
现在，你需要帮助润色其中一个段落，以提高整篇综述的连贯性。
以下是<需要润色的段落>，以及<前一个段落>和<后一个段落>的内容：

<需要润色的段落>
{paragraph_text}
</需要润色的段落>

<前一个段落>
{previous_text}
</前一个段落>

<后一个段落>
{following_text}
</后一个段落>

*注意事项*：
请勿换行
对于提出观点和逻辑论证相关的句子，润色后的语言应当满足两个标准：一是简练，也就是句式结构简答，措辞客观易懂；二是流畅，也就是句与句之间、段与段之间有明显的信息流动性，每句话的句首应当为此前出现过的信息。
对于列举证据、陈述事实相关的句子，润色后的语言应当满足详尽的标准，也就是概念应当具体、信息应当充分、细节应当完整。
请记住，润色过程中段落的核心信息和要点不变！
请记住，不要修改句子中的引用标记（如作者和年份）！
请记住，只返回润色后的完整段落内容，不要包含任何其他信息（如"以下是润色后的内容："等说明）！

润色后段落内容：
'''

        try:
            response = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4.5-air",
                max_tokens=8000
            )
            if "API call failed" not in str(response):
                return response.strip()
        except Exception:
            pass

        response = handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=8000
        )
        return response.strip()

    def refine_paragraph_list(
            self,
            paragraphs_list: List[str],
            topic: str = ""
    ) -> List[str]:
        """
        Refine multiple paragraphs in parallel to improve overall coherence.

        Args:
            paragraphs_list: List of paragraph texts to refine
            topic: The main topic for context

        Returns:
            List of refined paragraph texts
        """
        refined_paragraphs = [None] * len(paragraphs_list)

        def process_refine_task(task_data):
            """Task function for refining a single paragraph."""
            index, paragraph = task_data
            prev_paragraph = paragraphs_list[index - 1] if index > 0 else ""
            next_paragraph = paragraphs_list[index + 1] if index < len(paragraphs_list) - 1 else ""

            try:
                result = self.refine_paragraph(paragraph, prev_paragraph, next_paragraph, topic)
                return (index, result)
            except Exception as e:
                print(f"Refinement task failed (paragraph {index + 1}/{len(paragraphs_list)}): {str(e)}")
                return (index, paragraph)  # Return original if failed

        # Prepare refinement tasks
        refine_tasks = [(i, paragraph) for i, paragraph in enumerate(paragraphs_list) if paragraph]

        if not refine_tasks:
            return paragraphs_list

        # Execute all refinement tasks in parallel
        print(f"Launching {len(refine_tasks)} refinement tasks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_refine_task, refine_tasks))

        # Process results
        for index, refined_text in results:
            refined_paragraphs[index] = refined_text

        return refined_paragraphs

    # ==================== Parallelized Generation Orchestration ====================

    def generate_paragraphs_only(self, review_data: Dict) -> Dict:
        """
        Generate all paragraphs in parallel for each sub-topic.

        This implements the parallelized generation strategy to simultaneously
        develop content for each sub-topic, then refine them for coherence.

        Args:
            review_data: Nested dictionary {subtopic: {content_type: [documents]}}

        Returns:
            Nested dictionary {subtopic: {content_type: {paragraph: text}}}
        """
        content_results = {}

        # Initialize result structure
        for topic in review_data.keys():
            content_results[topic] = {}

        def process_paragraph_task(task_data):
            """Task function for generating a single paragraph."""
            topic, content_type, data_list = task_data
            try:
                result = self.generate_paragraph(data_list, content_type, topic)
                return (topic, content_type, result)
            except Exception as e:
                print(f"Paragraph task failed ({topic}, {content_type}): {str(e)}")
                return (topic, content_type, f"Generation failed: {str(e)}")

        # Prepare paragraph generation tasks
        paragraph_tasks = []
        for topic, content_types in review_data.items():
            for content_type, data_list in content_types.items():
                if not data_list:
                    continue

                content_results[topic][content_type] = {}
                paragraph_tasks.append((topic, content_type, data_list))

        # Execute all paragraph tasks in parallel
        print(f"Launching {len(paragraph_tasks)} paragraph generation tasks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            paragraph_results = list(executor.map(process_paragraph_task, paragraph_tasks))

        # Process paragraph results and collect for refinement
        paragraphs_by_topic = {}

        for topic, content_type, result in paragraph_results:
            content_results[topic][content_type]["paragraph"] = result

            # Collect successful paragraphs for refinement
            if not result.startswith("Generation failed:") and not result.startswith("API call failed:"):
                if topic not in paragraphs_by_topic:
                    paragraphs_by_topic[topic] = []
                paragraphs_by_topic[topic].append(result)

        # Refine paragraphs for each topic
        for topic, paragraphs in paragraphs_by_topic.items():
            if len(paragraphs) > 1:
                print(f"Refining {len(paragraphs)} paragraphs for topic '{topic}'...")
                refined_paragraphs = self.refine_paragraph_list(paragraphs, topic)

                # Update results with refined paragraphs
                paragraph_index = 0
                for content_type, results in content_results[topic].items():
                    if "paragraph" in results and \
                            not results["paragraph"].startswith("Generation failed:") and \
                            not results["paragraph"].startswith("API call failed:"):
                        results["paragraph"] = refined_paragraphs[paragraph_index]
                        paragraph_index += 1

        return content_results