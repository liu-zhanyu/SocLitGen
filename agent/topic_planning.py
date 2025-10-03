"""
Topic Planning Module

This module accepts and interprets user input to output well-structured review sub-topics.
It applies few-shot prompting techniques to guide an LLM in decomposing a user-specified
social science research topic into a set of complementary sub-topics.

Key principles:
1. Conceptual distinctiveness - sub-topics must be non-overlapping in meaning
2. Specificity - each sub-topic must target a concrete research dimension within its focal concept
"""

from typing import List
from components.llm_call import handler


class TopicPlanningAgent:
    """
    Topic Planning Agent for decomposing research topics into structured sub-topics.

    This agent uses LLM-driven planning strategy validated across multiple studies
    for producing well-structured theoretical outlines.
    """

    def __init__(self):
        self.name = "Topic Planning Agent"

    def decompose_topic(self, main_topic: str) -> List[str]:
        """
        Decompose a main research topic into 2-3 complementary sub-topics.

        The research topic typically lies at the intersection of these foundational sub-topics.
        Each sub-topic represents a distinct conceptual dimension with specific research direction.

        Args:
            main_topic: The main research topic to be decomposed

        Returns:
            List of sub-topic strings, each formatted as "A[concept]的B[direction]研究"

        Example:
            Input: "企业数字化转型对技术创新的影响研究"
            Output: ["企业数字化转型的效应研究", "技术创新的影响因素研究"]
        """
        print(f"{self.name}: Decomposing topic '{main_topic}'...")

        prompt = self._build_decomposition_prompt(main_topic)

        # Try multiple LLM providers with fallback mechanism
        response = self._call_llm_with_fallback(prompt)

        # Process response into list of sub-topics
        subtopics = self._parse_subtopics(response)

        print(f"{self.name}: Topic decomposed into {len(subtopics)} sub-topics")
        return subtopics

    def _build_decomposition_prompt(self, main_topic: str) -> str:
        """
        Build the few-shot prompt for topic decomposition.

        The prompt guides the LLM through a structured thinking process:
        1. Identify core concepts in the research topic
        2. Determine specific research directions for each concept
        3. Formulate complete research sub-topics
        """
        prompt = f"""
<任务>
请你分析研究选题，识别构成该选题的2-3个基础研究议题。研究选题通常位于这些基础议题的交叉点。
</任务>

<任务完成过程>
<输入格式>
研究选题：[研究选题文本]
</输入格式>

<思考步骤>
（仅内部思考，不要在最终回答中输出这部分内容）

步骤1：分析选题中包含的核心概念。
我看到的核心概念有：[列出核心概念]

步骤2：确定提供的选题聚焦于每个核心概念的什么研究方向。
在[概念1]的诸多研究方向中，提供的选题聚焦于："[概念1]的[特定方向]"
在[概念2]的诸多研究方向中，提供的选题聚焦于："[概念2]的[特定方向]"
[...]

步骤3：将这些研究方向表述为完整的研究议题。
对于概念1，完整研究议题是：[概念1]的[特定方向]研究
对于概念2，完整研究议题是：[概念2]的[特定方向]研究
[...]
</思考步骤>

<输出格式>
格式要求：输出时不要添加任何序号、标点符号或前缀，每个研究议题单独占一行。不要输出思考过程，只输出最终的研究议题。
第一个研究议题，格式为"A[概念]的B[方向]研究"
第二个研究议题，格式为"C[概念]的D[方向]研究"
第三个研究议题（如有需要），格式为"E[概念]的F[方向]研究"
</输出格式>

<任务完成过程>

<任务示例>
研究选题：企业数字化转型对技术创新的影响研究

步骤1：分析选题中包含的核心概念。
我看到的核心概念有：企业数字化转型、技术创新

步骤2：确定每个核心概念相关的研究方向。
在企业数字化转型的诸多研究中，提供的选题聚焦于："企业数字化转型的效应"
在技术创新的诸多研究中，提供的选题聚焦于："技术创新的影响因素"

步骤3：将这些研究方向表述为完整的研究议题。
对于企业数字化转型，完整研究议题是：企业数字化转型的效应研究
对于技术创新，完整研究议题是：技术创新的影响因素研究

输出：
企业数字化转型的效应研究
技术创新的影响因素研究
</任务示例>

研究选题：{main_topic}

输出：
"""
        return prompt

    def _call_llm_with_fallback(self, prompt: str) -> str:
        """
        Call LLM with automatic fallback across multiple providers.

        Tries providers in order: OpenAI -> Zhipu AI -> Doubao (Ark)
        Each provider is checked for API call failures before moving to next.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response text

        Raises:
            Exception: If all providers fail
        """
        # Try OpenAI first
        try:
            response = handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=1000
            )
            if "API call failed" not in str(response):
                return response
            raise Exception("API call failed detected in response")
        except Exception as e:
            print(f"{self.name}: OpenAI failed - {e}")

        # Fallback to Zhipu AI
        try:
            response = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4-air-250414",
                max_tokens=1000
            )
            if "API call failed" not in str(response):
                return response
            raise Exception("API call failed detected in response")
        except Exception as e:
            print(f"{self.name}: Zhipu AI failed - {e}")

        # Final fallback to Doubao (Ark)
        try:
            response = handler.call_llm(
                provider="ark",
                prompt=prompt,
                model="doubao-1-5-lite-32k-250115",
                max_tokens=1000
            )
            return response
        except Exception as e:
            print(f"{self.name}: All LLM providers failed")
            raise Exception(f"Failed to decompose topic: {e}")

    def _parse_subtopics(self, response: str) -> List[str]:
        """
        Parse the LLM response into a clean list of sub-topics.

        Args:
            response: Raw response text from LLM

        Returns:
            List of cleaned sub-topic strings
        """
        # Split by lines and filter out empty strings
        subtopics = [
            topic.strip()
            for topic in response.strip().split('\n')
            if topic.strip()
        ]

        return subtopics

    def validate_subtopics(self, subtopics: List[str]) -> bool:
        """
        Validate that generated sub-topics meet quality criteria.

        Checks:
        1. Number of sub-topics (should be 2-3)
        2. Each sub-topic follows the expected format pattern
        3. Sub-topics are conceptually distinct (basic length check)

        Args:
            subtopics: List of sub-topic strings to validate

        Returns:
            True if sub-topics pass validation, False otherwise
        """
        # Check number of sub-topics
        if not (2 <= len(subtopics) <= 3):
            print(f"{self.name}: Warning - Expected 2-3 sub-topics, got {len(subtopics)}")
            return False

        # Check minimum length for each sub-topic
        for subtopic in subtopics:
            if len(subtopic) < 5:
                print(f"{self.name}: Warning - Sub-topic too short: {subtopic}")
                return False

        # Check for duplicate sub-topics
        if len(subtopics) != len(set(subtopics)):
            print(f"{self.name}: Warning - Duplicate sub-topics detected")
            return False

        return True