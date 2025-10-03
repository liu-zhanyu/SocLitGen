"""
Overview and Commentary Generation Module

This module generates macro-level summaries and critical analyses. Given that the overview
and commentary sections require macro-level summarization and critical analysis of the
overall research, this module is activated after all sub-topic content is completed.

Based on the global understanding capabilities of LLMs, this module:
1. Generates the overview section to systematically elaborate on the academic value of
   research topics and sub-topic classification logic
2. Generates the commentary section to deeply analyze the theoretical contributions of
   existing research and the potential academic value-added of the given research topic

This global overview and evaluation strategy ensures that the review possesses a complete
academic framework and deep critical thinking, providing readers with comprehensive academic
perspectives from whole to part, from current status to prospects.
"""

import concurrent.futures
from typing import Tuple
from components.llm_call import handler


class OverviewCommentaryAgent:
    """
    Overview and Commentary Generation Agent for creating macro-level analysis.

    Provides systematic overviews and critical evaluations after all sub-topic content
    has been generated, ensuring complete academic framework and deep critical thinking.
    """

    def __init__(self):
        self.name = "Overview and Commentary Agent"

    def generate_overview(self, full_text: str, main_topic: str) -> str:
        """
        Generate the overview section with macro-level summary.

        The overview systematically elaborates on:
        1. The academic value and importance of the research topic (1-2 sentences)
        2. The classification logic of sub-topics (1-2 sentences)
        3. Detailed explanation of categories within each sub-topic (1-2 sentences per sub-topic)

        Args:
            full_text: Complete text of all sub-topic reviews
            main_topic: The main research topic

        Returns:
            Overview section text
        """
        prompt = f"""
作为一位专业的文献综述专家，请为以下关于"{main_topic}"的<文献回顾>撰写一个总论部分。请严格按照<写作要求>来完成。

<文献回顾>
{full_text}
</文献回顾>

<写作要求>
总论部分应包含以下内容：
1. 概述"{main_topic}"作为研究选题的重要性(1-2句)。
2. 说明研究选题相关的研究可分为几个子议题及其分类逻辑（1-2句）。
3. 详细阐述每个子议题下包含哪些细分类别（每个子议题1-2句）。

*注意*：
- 总论是一个整体概述，不是引言，用一段流畅、有逻辑的话语来阐述，请勿换行。
- 保持语句逻辑清晰，用第一，第二……等措辞增强结构性。
- 回答中只包含总论内容，不要包含"以下是总论内容"等额外说明。
</写作要求>

总论：
"""

        response = self._call_llm_with_fallback(prompt)
        return response.strip()

    def generate_critique(self, full_text: str, main_topic: str) -> str:
        """
        Generate the commentary section with critical analysis.

        The commentary deeply analyzes:
        1. Main insights from existing research for the given topic in terms of content,
           methods, perspectives, or data (2-3 sentences)
        2. Detailed explanation of how the current research can contribute to existing
           literature for each sub-topic (3-4 sentences per sub-topic)

        Args:
            full_text: Complete text of all sub-topic reviews
            main_topic: The main research topic

        Returns:
            Commentary section text
        """
        prompt = f"""
作为一位专业的文献综述专家，请为以下关于"{main_topic}"的<文献回顾>撰写一个述评部分。请严格按照<写作要求>来完成。

<文献回顾>
{full_text}
</文献回顾>

<写作要求>
述评部分应包含以下内容：
1. 概述现有研究为"{main_topic}"这一选题研究带来的主要启示（内容、方法、视角或数据上）（2-3句）。
2. 详细阐述本研究"{main_topic}"能够为每个子主题的现有文献做出怎样的补充，每个子主题用3-4句分开来阐述。

*注意*：
- 保持语句逻辑清晰，用首先，其次……等措辞增强结构性。
- 用一段流畅、通顺的话来阐述，请勿换行。
- 突出现有研究对"{main_topic}"这一选题的启示价值。
- 突出"{main_topic}"这一选题对现有研究的补充作用。
- 回答中只包含述评内容，不要包含"以下是述评内容"等额外说明。
</写作要求>

述评：
"""

        response = self._call_llm_with_fallback(prompt)
        return response.strip()

    def _call_llm_with_fallback(self, prompt: str) -> str:
        """
        Call LLM with automatic fallback across multiple providers.

        Tries providers in order: Zhipu AI -> Doubao -> SiliconFlow -> OpenAI

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response text
        """
        # Try Zhipu AI first
        try:
            response = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4-air-250414",
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

    def generate_overview_and_critique(
            self,
            full_text: str,
            main_topic: str
    ) -> Tuple[str, str]:
        """
        Generate both overview and critique sections in parallel.

        This is the main entry point that orchestrates parallel generation of both
        macro-level analysis sections for efficiency.

        Args:
            full_text: Complete text of all sub-topic reviews
            main_topic: The main research topic

        Returns:
            Tuple of (overview_text, critique_text)
        """

        def process_overview_task():
            """Task function for generating overview."""
            try:
                result = self.generate_overview(full_text, main_topic)
                return ("overview", result)
            except Exception as e:
                print(f"Overview generation failed: {str(e)}")
                return ("overview", f"Generation failed: {str(e)}")

        def process_critique_task():
            """Task function for generating critique."""
            try:
                result = self.generate_critique(full_text, main_topic)
                return ("critique", result)
            except Exception as e:
                print(f"Critique generation failed: {str(e)}")
                return ("critique", f"Generation failed: {str(e)}")

        # Generate overview and critique in parallel
        print(f"{self.name}: Launching overview and critique generation tasks...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            overview_future = executor.submit(process_overview_task)
            critique_future = executor.submit(process_critique_task)

            overview_result = overview_future.result()
            critique_result = critique_future.result()

        print(f"{self.name}: Completed overview and critique generation")

        return overview_result[1], critique_result[1]