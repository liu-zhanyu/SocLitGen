"""
Table Generation Module

This module generates literature organization tables for each sub-topic. The necessity
stems from traditional conventions in social science academic writing—providing clearer
and more intuitive literature information display through tabular formats.

This tabular presentation method has been widely adopted and encouraged by numerous
authoritative social science journals (e.g., Management Review Quarterly, Journal of
the Academy of Marketing Science).
"""

import concurrent.futures
from typing import Dict
from components.llm_call import handler


class TableGenerationAgent:
    """
    Table Generation Agent for creating structured literature organization tables.

    Generates Markdown tables that provide researchers with effective tools for quickly
    grasping field development trajectories and constructing clear literature navigation maps.
    """

    def __init__(self):
        self.name = "Table Generation Agent"

    def generate_table(
            self,
            paragraph: str,
            content_type: str,
            subtopic: str
    ) -> str:
        """
        Generate a structured table based on the content of a paragraph.

        Tables are designed according to content type:
        - Concept: Shows how different definitions cover core dimensions
        - Theory: Maps theories to their perspectives and research examples
        - Findings: Categorizes research by type, method, and focus

        Args:
            paragraph: The generated paragraph text
            content_type: Type of content (concept/theory/findings)
            subtopic: The research sub-topic

        Returns:
            Markdown-formatted table string
        """
        if content_type == "concept":
            example_table = """
| 文献来源 | 核心要点1 | 核心要点2 | 核心要点…… |
| ------- | ------- | ------- | ------- |
| (作者, 年份) | 定义中该核心要点的表述 | 定义中该核心要点的表述 | 定义中该核心要点的表述 |
| (作者, 年份) | 定义中该核心要点的表述 | 定义中该核心要点的表述 | 定义中该核心要点的表述 |
| (作者, 年份) | 定义中该核心要点的表述 | 定义中该核心要点的表述 | 定义中该核心要点的表述 |
"""
            prompt = f"""
你是一位学术文献分析专家。请基于以下段落，创建一个关于"{subtopic}"核心概念的表格。
表格应包含以下列：文献来源(作者, 年份)、核心要点1、核心要点2等

请从原段落中提取每个引用的文献来源和对应的核心概念要点，保持原文的引用格式。
**非常重要：你的回答必须只包含markdown表格，不要有任何其他文字或解释。表格必须以"|"开头并以"|"结尾。**

参考表格格式如下：
{example_table}

原段落：
{paragraph}
"""
        elif content_type == "theory":
            example_table = """
| 理论名称 | 理论观点 | 文献来源 |
| ------- | ------- | ------- |
| 理论1 | 这是理论1的主要观点 | (作者, 年份) |
| 理论2 | 这是理论2的主要观点 | (作者, 年份) |
| 理论3 | 这是理论3的主要观点 | (作者, 年份) |
"""
            prompt = f"""
你是一位学术文献分析专家。请基于以下段落，创建一个关于"{subtopic}"理论基础的表格。
表格应包含以下列：理论名称、理论观点、文献来源(作者, 年份)

请从原段落中提取每个理论的名称、主要观点和对应的文献来源，保持原文的引用格式。
**非常重要：你的回答必须只包含markdown表格，不要有任何其他文字或解释。表格必须以"|"开头并以"|"结尾。**

参考表格格式如下：
{example_table}

原段落：
{paragraph}
"""
        else:  # findings
            example_table = """
| 研究类型 | 研究方法 | 关注议题 | 文献来源 |
| ------- | ------- | ------- | ------- |
| 类型1的名称 | 问卷调查 | 这是研究1关注的因素 | (作者, 年份) |
| 类型1的名称 | 深度访谈 | 这是研究2关注的因素 | (作者, 年份) |
| 类型2的名称 | 对照实验 | 这是研究3关注的因素 | (作者, 年份) |
"""
            prompt = f"""
你是一位学术文献分析专家。请基于以下段落，创建一个关于"{subtopic}"研究发现的表格。
表格应包含以下列：研究类型、研究方法、关注议题、文献来源(作者, 年份)

注意事项：
请从原段落中提取每项研究的类型、使用的研究方法、关注的议题和对应的文献来源，保持原文的引用格式。
如果原文未明确提及某一列的内容，可以基于上下文合理推断。
每篇文献在表格中只出现一次，请勿重复呈现。
**非常重要：你的回答必须只包含markdown表格，不要有任何其他文字或解释。表格必须以"|"开头并以"|"结尾。**

参考表格格式如下：
{example_table}

原段落：
{paragraph}

你的markdown表格：
"""

        # Call LLM with fallback
        try:
            response = handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=8000
            )
            if "API call failed" not in str(response):
                return self._extract_table_lines(response)
        except Exception:
            pass

        response = handler.call_llm(
            provider="siliconflow",
            prompt=prompt,
            model="THUDM/GLM-4-32B-0414",
            max_tokens=8000
        )

        return self._extract_table_lines(response)

    def _extract_table_lines(self, response: str) -> str:
        """Extract valid table lines from LLM response."""
        lines = response.strip().split('\n')
        table_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                table_lines.append(line)

        if table_lines:
            return '\n'.join(table_lines)
        else:
            print(f"{self.name}: Warning - Could not extract table format. Returning raw response.")
            return response.strip()

    def generate_tables_only(self, paragraph_results: Dict) -> Dict:
        """
        Generate tables for all sub-topics in parallel based on paragraph results.

        Args:
            paragraph_results: Nested dictionary {subtopic: {content_type: {paragraph: text}}}

        Returns:
            Nested dictionary {subtopic: {content_type: {table: text}}}
        """

        def process_table_task(task_data):
            """Task function for generating a single table."""
            topic, content_type, paragraph, subtopic = task_data
            try:
                result = self.generate_table(paragraph, content_type, subtopic)
                return (topic, content_type, result)
            except Exception as e:
                print(f"Table task failed ({topic}, {content_type}): {str(e)}")
                return (topic, content_type, f"Generation failed: {str(e)}")

        # Prepare table generation tasks
        table_tasks = []
        for topic, content_types in paragraph_results.items():
            for content_type, results in content_types.items():
                if "paragraph" in results:
                    paragraph_content = results["paragraph"]
                    if paragraph_content and \
                            not paragraph_content.startswith("Generation failed:") and \
                            not paragraph_content.startswith("API call failed:"):
                        table_tasks.append((topic, content_type, paragraph_content, topic))

        # Execute all table generation tasks in parallel
        print(f"Launching {len(table_tasks)} table generation tasks...")
        table_results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(process_table_task, table_tasks))

        # Process results
        for topic, content_type, result in results:
            if topic not in table_results:
                table_results[topic] = {}
            if content_type not in table_results[topic]:
                table_results[topic][content_type] = {}
            table_results[topic][content_type]["table"] = result

        print(f"Completed table generation for {len(table_results)} topics")
        return table_results