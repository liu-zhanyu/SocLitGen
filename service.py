import logging
from rag_search_service import *
from chat_with_data_service import *
from introduction_agent import *
from hypothesis_agent import *
from literature_review_agent import *
from editor import *
from pay_service import *
from es_search import *
from ai_gen_data import *

import zipfile
import asyncio
import aiohttp
from datetime import datetime
from aigc_reduction import *
from ai_social_scientist import *
from check_state import *
from pre_search_service import extractor,classify_academic_query_type,get_chunk_types,get_optimal_query,get_tanslated_optimal_query,rewrite_and_disambiguate_query
from risk_diagnose_service import Risk_diagnose
from xi_diagnose_service import PDFQuoteMatcher
from full_paper import PaperWritingAgent
from information_extraction import InfoExtra
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('literature_review_service.log')
    ]
)
logger = logging.getLogger("literature_review_service")


def search_documents_simple(query, top_k=30, docnm_kwds=None):
    """
    简单搜索文档函数，kb_id已固定

    Args:
        query: 查询文本
        top_k: 返回结果数量 (默认10)
        docnm_kwd: 限定的文献编号（列表）

    Returns:
        搜索结果列表
    """
    try:
        # 初始化ElasticsearchService
        es_service = ElasticsearchService()

        # 连接到Elasticsearch
        if not es_service.connect():
            print("无法连接到Elasticsearch")
            return []

        # 使用固定的kb_id执行搜索
        # kb_id = "7750e714049611f08aa20242ac120003"
        kb_id="684df8ee1e7511f0a9ff0242ac120006"
        # 执行搜索
        results = es_service.search_documents_with_vector(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            docnm_kwds=docnm_kwds,
        )

        return results

    except Exception as e:
        print(f"搜索出错: {e}")
        return []


def search_with_extracted_params_v3(
        es_service: ElasticsearchService,
        user_query: str,
        top_k: int = 30,
        additional_filters: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    结合参数提取和检索功能，从用户自然语言查询中提取学术参数并执行搜索

    Args:
        es_service: ElasticsearchService实例
        user_query: 用户的自然语言查询
        top_k: 返回结果数量
        default_year_window: 当仅提取到单一年份时的窗口大小(±年数)
        additional_filters: 额外的筛选参数，如kb_id, chunk_type等

    Returns:
        Tuple 包含:
        - 检索结果列表
        - 提取的参数字典
    """
    # 第一步：从用户查询中提取学术参数
    # 使用线程池并行执行参数提取和查询类型分类
    start_time = time.time()

    thread_start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 同时提交两个任务
        extract_params_future = executor.submit(extractor.extract_params, user_query)
        classify_query_future = executor.submit(classify_academic_query_type, user_query)
        optimal_query_future=executor.submit(get_optimal_query, user_query)
        tanslated_optimal_query_future=executor.submit(get_tanslated_optimal_query, user_query)

        # 获取结果
        extracted_params = extract_params_future.result()
        query_type = classify_query_future.result()
        optimal_query=optimal_query_future.result()
        tanslated_optimal_query = tanslated_optimal_query_future.result()
    print(f"[{time.time() - start_time:.4f}s] 完成参数提取和查询分类 (耗时: {time.time() - thread_start_time:.4f}s)")
    print(f"从查询中提取的参数: {extracted_params}")

    # 提取的参数包括year(列表), journal(字符串或列表), authors(列表)
    year_range = extracted_params.get("year", [])
    journals = extracted_params.get("journal", [])
    authors = extracted_params.get("authors", [])
    language=extracted_params.get("language",[])

    chunk_type = get_chunk_types(query_type)
    # 处理年份范围
    if len(year_range) == 1:
        # 如果只有单一年份，创建一个窗口
        single_year = year_range[0]
        year_range = [single_year, single_year]
        # 确保不超过当前年份
        current_year = datetime.now().year
        if year_range[1] > current_year:
            year_range[1] = current_year
        print(f"从单一年份 {single_year} 扩展为年份范围: {year_range}")

    # 准备额外的搜索参数
    search_params = {}
    if additional_filters:
        search_params.update(additional_filters)

    # 如果提取到期刊、作者或年份，从查询文本中移除这些内容以避免重复
    # 这个步骤是可选的，可以根据需要调整
    # 此处简单实现，实际应用可能需要更复杂的文本处理

    # 执行搜索
    print(
        f"执行搜索 - 检索词: '{optimal_query}', 期刊: {journals}, 作者: {authors}, 年份范围: {year_range}, 中层chunk类型：{chunk_type}")
    results = es_service.hybrid_search_v4(
        query_text=optimal_query,
        translated_text=tanslated_optimal_query,
        top_k=top_k,
        journals=journals,
        authors=authors,
        chunk_type=chunk_type,
        year_range=year_range if year_range else None,
        language=language
    )
    # 返回结果和提取的参数
    end_time = time.time()  # ? - 总计时结束
    total_time = end_time - start_time
    print(f"[{end_time - start_time:.4f}s] 函数执行完成，总耗时: {total_time:.4f}s")

    # 返回结果和提取的参数
    return results, {
        "year_range": year_range,
        "journals": journals,
        "authors": authors,
        "language":language,
        "chunk_type": chunk_type,
        "original_query": user_query,
        "search_query": [optimal_query,tanslated_optimal_query]
    }


def generate_followup_questions(dialogue_dict: dict) -> List[str]:
    """
    基于对话字典生成学术性追问（改进版）

    Args:
        dialogue_dict: {
            "user": "合并后的所有用户发言内容",
            "assistant": "合并后的所有助手回复内容"
        }

    Returns:
        包含追问的字符串列表，如:
        ["追问内容1?", "追问内容2?"]

    Raises:
        ValueError: 当输入字典格式不正确时
    """
    # 输入验证
    if not isinstance(dialogue_dict, dict):
        raise ValueError("输入必须是字典类型")
    if "user" not in dialogue_dict or "assistant" not in dialogue_dict:
        raise ValueError("字典必须包含'user'和'assistant'键")
    if not isinstance(dialogue_dict["user"], str) or not isinstance(dialogue_dict["assistant"], str):
        raise ValueError("对话内容必须是字符串类型")

    # 默认追问（用于错误回退）
    DEFAULT_QUESTIONS = [
        "能否进一步阐述核心概念的操作化定义？",
        "该观点的实证支持来自哪些具体研究？"
    ]

    try:
        # 构建对话历史字符串
        history_dialogue = f"用户: {dialogue_dict['user']}\n助手: {dialogue_dict['assistant']}"

        prompt = f"""<TASK>
你是一个学术对话分析引擎，目标是通过精准追问推动讨论明晰化。请基于<HISTORY_DIALOGUE>中的对话内容，生成三个中性追问，严格遵循<QUESTION_GUIDELINES>中的规则。
</TASK>

<HISTORY_DIALOGUE>
{history_dialogue}
</HISTORY_DIALOGUE>

<QUESTION_GUIDELINES>
## 追问准则
1. 模糊点锚定策略：
   - 概念模糊：定位高频未定义术语（如"概念A的定义是什么？"）
   - 逻辑断层：识别未经验证的推论跳跃（如"从A变量影响B变量的内在机制是什么？"）
   - 数据缺口：发现未被量化的核心变量（如"概念A可以通过什么方法和数据来测量？"）

2. 追问生成原则：
   - 采用"概念界定→机制验证→实证路径"递进结构
   - 每个问题必须包含对话中的原文关键词（用单引号标注）
   - 使用客观限定表达式（"在X框架下"/"当控制Y变量时"/"哪些测量维度"）

3. 安全过滤机制：
   - 禁止输出以下类型的问题：
     * 包含政治敏感内容的问题
     * 涉及特定政策评价的问题
     * 针对特定群体的歧视性问题
     * 涉及现行制度批评的问题
     * 引导意识形态站队的问题
     * 涉及宗教信仰判断的问题
     * 暗示地缘政治立场的问题
     * 含有民族或区域偏见的问题
     * 讨论社会争议敏感事件的问题
     * 暗示经济体系优劣的问题
   - 发现敏感内容时，应生成完全中性的学术问题（如"不同约束条件下的表现差异"）

## 逻辑完整性检测
1. 构建论点树状图：
   - 主论点 → 子论点 → 支撑数据
2. 识别三类断裂点：
   - 红区：无数据支撑的终节点
   - 黄区：单一数据支撑多重推论
   - 蓝区：存在矛盾证据的节点
3. 生成对应追问：
   [FQ_START]支撑【子论点X】的数据是否排除竞争性解释？[FQ_END]

## 优质案例
历史对话中出现："领导风格影响员工创新行为"
[FQ_START]'领导风格'包含哪些具体维度？[FQ_END]
[FQ_START]领导风格影响创新行为的内在机理是什么？[FQ_END]
[FQ_START]领导风格可以通过什么方式来测量？[FQ_END]
</QUESTION_GUIDELINES>

<OUTPUT_INSTRUCTION>
1. 问题长度25-35汉字，以问号结尾
2. 每个追问用[FQ_START]和[FQ_END]严格包裹
3. 置于回答末尾，与正文间隔一个空行
4. 仅输出三个追问，不要添加任何解释或其他内容
</OUTPUT_INSTRUCTION>

追问结果："""

        # 模拟API调用（实际使用时替换为真实API调用）
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )

        raw_output = response.choices[0].message.content.strip()
        print(raw_output)

        # 使用正则表达式提取追问内容
        pattern = r"\[FQ_START\](.*?)\[FQ_END\]"
        matches = re.findall(pattern, raw_output, re.DOTALL)

        # 处理提取结果
        if matches:
            # 场景1: 三个问题都是完整的，有[FQ_START]和[FQ_END]包裹的完整句子
            cleaned_questions = []
            for q in matches:
                q = q.strip()
                if not q.endswith("？") and not q.endswith("?"):
                    q += "？"
                cleaned_questions.append(q)
            return cleaned_questions
        else:
            # 场景2: 三个问题都不完整，用问号或换行符分割
            # 先移除[FQ_START]和[FQ_END]标记
            clean_output = re.sub(r"\[FQ_START\]|\[FQ_END\]", "", raw_output)

            # 尝试用问号分割
            question_marks_split = [q.strip() + "？" for q in re.split(r"[?？]", clean_output) if q.strip()]

            if question_marks_split:
                return question_marks_split

            # 如果问号分割失败，尝试用换行符分割
            newline_split = [line.strip() for line in clean_output.split('\n') if line.strip()]
            filtered_questions = []
            for line in newline_split:
                # 如果行不是很短，且看起来像问题（不是指导或解释文本）
                if len(line) > 10 and not line.startswith("注：") and not line.startswith("说明："):
                    if not line.endswith("？") and not line.endswith("?"):
                        line += "？"
                    filtered_questions.append(line)

            if filtered_questions:
                return filtered_questions

            # 场景3: 只有一个问题的情况
            # 如果只剩下一个长字符串，尝试识别其中的问题
            if clean_output:
                # 如果整个输出内容看起来像一个问题，将其作为单个问题返回
                clean_output = clean_output.strip()
                if not clean_output.endswith("？") and not clean_output.endswith("?"):
                    clean_output += "？"
                return [clean_output]

            # 如果所有方法都失败，返回默认问题
            return DEFAULT_QUESTIONS

    except re.error as e:
        print(f"正则表达式处理失败: {e}")
        return DEFAULT_QUESTIONS
    except Exception as e:
        print(f"生成追问时出错: {e}")
        return DEFAULT_QUESTIONS


def generate_dialogue_title(dialogue_dict: dict) -> str:
    """
    基于对话字典生成简洁明了的对话标题

    Args:
        dialogue_dict: {
            "user": "合并后的所有用户发言内容",
            "assistant": "合并后的所有助手回复内容"
        }

    Returns:
        代表对话主题的标题字符串
    """
    # 输入验证
    if not isinstance(dialogue_dict, dict):
        print("错误: 输入必须是字典类型")
        return "未分类对话主题"
    if "user" not in dialogue_dict or "assistant" not in dialogue_dict:
        print("错误: 字典必须包含'user'和'assistant'键")
        return "未分类对话主题"

    combined_text = dialogue_dict["user"] + " " + dialogue_dict["assistant"]

    # 默认标题
    DEFAULT_TITLE = combined_text[:10] + "……"

    try:
        # 构建对话历史字符串
        history_dialogue = f"用户: {dialogue_dict['user']}\n助手: {dialogue_dict['assistant']}"

        prompt = f"""<TASK>
你是一个对话标题生成引擎，目标是为对话生成简洁明了且具有代表性的标题。请基于<HISTORY_DIALOGUE>中的对话内容，生成一个合适的标题，严格遵循<TITLE_GUIDELINES>中的规则。
</TASK>

<HISTORY_DIALOGUE>
{history_dialogue}
</HISTORY_DIALOGUE>

<TITLE_GUIDELINES>
## 标题生成准则
1. 核心主题提取策略：
   - 关键词提取：识别对话中高频出现的专业术语或核心概念
   - 主旨概括：捕捉对话的核心问题或讨论重点
   - 领域归类：确定对话所属的专业或知识领域

2. 标题生成原则：
   - 长度控制在5-40个汉字之间
   - 避免使用过于笼统的词语（如"讨论"、"探讨"等）
   - 使用名词性短语结构，避免使用完整句子
   - 不使用标点符号（包括问号、感叹号等）

3. 安全过滤机制：
   - 禁止生成以下类型的标题：
     * 包含政治敏感内容的标题
     * 涉及特定政策评价的标题
     * 针对特定群体的歧视性标题
     * 涉及制度批评的标题
     * 有明显意识形态倾向的标题
     * 涉及宗教信仰判断的标题
     * 暗示地缘政治立场的标题
     * 含有民族或区域偏见的标题
     * 讨论社会争议敏感事件的标题
   - 发现敏感内容时，应生成中性的学术性标题
</TITLE_GUIDELINES>

<OUTPUT_INSTRUCTION>
1. 标题必须用[TITLE_START]和[TITLE_END]严格包裹
2. 标题应为中文名词性短语，不使用标点符号
3. 标题长度控制在40个汉字以内
4. 不要添加任何解释或其他内容
</OUTPUT_INSTRUCTION>

标题："""

        try:
            # 调用API生成标题
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )

            raw_output = response.choices[0].message.content.strip()
            # print(f"API返回: {raw_output}")

            # 提取标题
            pattern = r"\[TITLE_START\](.*?)\[TITLE_END\]"
            matches = re.findall(pattern, raw_output, re.DOTALL)

            if matches:
                title = matches[0].strip()
                # 移除所有标点符号并截断长度
                title = re.sub(r'[^\w\s]', '', title)
                if len(title) > 40:
                    title = title[:40]
                return title
            else:
                # 使用flashtext提取关键词作为备选方案
                raise Exception("未找到标记包裹的标题")

        except Exception as e:
            print(f"API调用或标题提取出错: {str(e)}")
            print("直接截取文字...")

            return DEFAULT_TITLE
    except Exception as e:
        print(f"生成标题出错: {str(e)}")
        return DEFAULT_TITLE

def summarize_abstracts(abstracts_list: list) -> str:
    """
    总结多篇研究摘要，突出核心观点并比较不同的研究视角。

    Args:
        abstracts_list: 字典列表，每个字典包含'author'、'year'和'abstract'键

    Returns:
        一个总结字符串，呈现核心观点，讨论相似点和差异点，并以(作者，年份)的格式包含引用
    """

    # 检查输入
    if not abstracts_list or not isinstance(abstracts_list, list):
        return "未提供有效摘要进行总结。"

    # 准备摘要文本
    abstracts_text = ""
    for i, item in enumerate(abstracts_list):
        author = item.get("authors", "未知")
        year = item.get("year", "未知")
        abstract = item.get("abstract", "")

        if not abstract:
            continue

        abstracts_text += f"摘要 {i+1}:\n作者: {author}\n年份: {year}\n内容: {abstract}\n\n"

    if not abstracts_text:
        return "未提供有效摘要进行总结。"

    # 构建语言模型提示词
    prompt = f"""
    <任务>
    你是一位研究综述专家。请分析以下几项研究的摘要，创建一个关于以下几项研究的全面的总结，要求：
    1. 识别并呈现所有摘要中的核心观点/发现
    2. 讨论各研究方法或结论的相似点和差异点
    3. 在呈现特定观点时，使用(作者，年份)的格式正确引用作者
    </任务>

    <摘要>
    {abstracts_text}
    </摘要>

    <指令>
    - 提取最重要的研究发现和观点
    - 将不同论文中的相关观点归类在一起
    - 突出研究中的共识和分歧之处
    - 确保每个关键点都正确归属于其来源
    - 以一段流畅的文字呈现所有内容，不要分段，呈现一个融合所有论文的连贯叙述
    - 使用学术、客观的语言风格
    </指令>

    <输出格式>
    用“这几项研究”开头作为主语。
    仅返回总结内容，不要包含任何解释或其他内容。
    使用[SUMMARY_START]和[SUMMARY_END]标记来包裹你的总结。
    总结应是一段流畅的文字，不要分段，要形成一个连贯的整体。
    使用(作者，年份)的格式在具体观点后直接引用。
    </输出格式>

    关于这几项研究的总结：
    """

    # 这里调用你的语言模型API
    # 以下是一个示例实现，你需要替换为实际的API调用
    try:

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # 或其他合适的模型
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        result = response.choices[0].message.content.strip()
        # 使用正则表达式提取总结内容
        match = re.search(r'\[SUMMARY_START\](.*?)\[SUMMARY_END\]', result, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            # 如果未找到标记，返回原始输出
            return result

    except Exception:
        # 如果API调用失败，返回一个示例输出
        return abstracts_text


def summarize_data(metadata_list: list) -> str:
    """
    总结不同测量指标分别代表了什么变量，并分析它们与人工智能研究的相关性。

    Args:
        metadata_json: 包含指标元数据的字典，需要有'meta data'键，其值为指标信息的列表

    Returns:
        一个总结字符串，呈现各指标对应的变量及其在人工智能研究中的意义
    """

    # 准备指标分析文本
    metrics_text = ""

    # 提取每个指标的关键信息
    core_variables = []
    non_core_variables = []
    research_topic=metadata_list[0]["research_topic"]

    for item in metadata_list:
        # 提取字段，处理可能的键名不一致问题
        variable = item.get("variable", "未知变量")
        indicator_name = item.get("indicator_name", "")
        if not indicator_name:
            # 尝试替代键名
            indicator_name = item.get("indicator name", "未知指标")

        is_core = item.get("is_core_variable", item.get("is core variable", False))
        # 获取建议指标列表
        suggested = item.get("suggested_indicators", item.get("suggested indicators", []))
        if isinstance(suggested, str):
            suggested = [s.strip() for s in suggested.replace('"', '').split(",")]

        # 构建指标信息
        indicator_info = {
            "variable": variable,
            "indicator_name": indicator_name,
            "suggested_indicators": suggested
        }

        # 分类为核心变量或非核心变量
        if is_core:
            core_variables.append(indicator_info)
        else:
            non_core_variables.append(indicator_info)

    # 构建语言模型提示词
    prompt = f"""
    <任务>
    你是一位研究指标分析专家。以下是为用户查询《{research_topic}》找到的数据集中的指标信息。请分析数据集指标信息，创建一个全面的总结，要求：
    1. 解释每个指标所代表的变量
    2. 区分核心指标和辅助指标
    </任务>

    <核心指标>
    """

    # 添加核心指标
    for i, item in enumerate(core_variables):
        prompt += f"""
    指标 {i+1}:{item['indicator_name']}
    对应变量: {item['variable']}
    """

    prompt += """
    </核心指标>

    <辅助指标>
    """

    # 添加非核心指标
    for i, item in enumerate(non_core_variables):
        prompt += f"""
    指标 {i+1}:{item['indicator_name']}
    对应变量: {item['variable']}
    """

    prompt += """
    </辅助指标>

    <指令>
    - 解释每个指标实际测量的是什么变量
    - 分析这些指标如何与人工智能研究相关联
    - 解释核心指标与辅助指标的区别和作用
    - 指出指标之间可能存在的相互关系
    - 使用学术、客观的语言风格
    </指令>

    <输出格式>
    以"该数据集"开头作为主语。
    总结分为三部分：
    1. 核心指标分析：分析每个核心指标所代表的变量
    2. 辅助指标分析：分析每个辅助指标所代表的变量

    使用[SUMMARY_START]和[SUMMARY_END]标记来包裹你的总结。请勿换行，用一段通顺的话来阐述。
    </输出格式>

    对数据集的总结：
    """

    # 这里需要调用语言模型API来生成总结
    # 以下是一个示例实现，实际使用时需要替换为真实API调用
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()

        # 使用正则表达式提取总结内容
        import re
        match = re.search(r'\[SUMMARY_START\](.*?)\[SUMMARY_END\]', result, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            # 如果未找到标记，返回原始输出
            return result.replace("[SUMMARY_START]","").replace("[SUMMARY_END]","")

    except Exception as e:
        # 如果API调用失败或处理过程出错，返回错误信息
        return f"生成总结时出现错误: {str(e)}"


def chat_with_data(query, df, metadata):
    analyzer = ChatWithData(df, metadata)
    result = analyzer.run(query)
    return result


#
# def process_text_with_references(text: str, literature_list: List[Dict], citation_checker) -> str:
#     """
#     处理文本，提取有效引用并添加参考文献列表
#
#     Args:
#         text: 需要处理的文本
#         literature_list: 文献列表
#         citation_checker: CitationChecker实例
#
#     Returns:
#         处理后的文本，带参考文献列表
#     """
#     # 提取有效引用的参考文献
#     valid_references = extract_valid_references(text, literature_list, citation_checker)
#
#     # 将参考文献添加到文本末尾
#     return add_references_to_text(text, valid_references)
#


def async_generate_literature_review(query, user_id, task_id, language=None,mode=None,structure=0,chinese_weight=None):
    """
    异步生成文献综述并存储到MongoDB

    Args:
        query: 研究主题
        user_id: 用户ID
        task_id: 任务ID
    """
    try:

        # 初始化文献综述系统
        # kb_id = "3d813266f5ad11ef96ef0242ac120006"
        # kb_id = "7750e714049611f08aa20242ac120003"
        kb_id=KB_ID_SUMMARY
        review_system = LiteratureReviewSystem(user_id, task_id, kb_id, mode)

        # 生成文献综述
        review_system.generate_review(query, language, chinese_weight,structure)

    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["review"]

        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {"state": 0, "error": str(e), "update_time": datetime.now()}}
        )
        print(f"生成文献综述时出错: {e}")


def async_generate_introduction(research_topic,user_id,task_id):

    # 初始化引言生成系统
    try:
        intro_agent = IntroductionAgent(kb_id=KB_ID_SUMMARY)
        introduction=intro_agent.generate_full_introduction(research_topic, user_id, task_id)
    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["introduction"]

        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {"state": 0, "error": str(e), "update_time": datetime.now()}}
        )
        print(f"生成引言时出错: {e}")


def async_generate_research_data(query, user_id, task_id):
    """
    异步生成研究数据并存储到MongoDB

    Args:
        query: 研究主题
        user_id: 用户ID
        task_id: 任务ID
    """
    excel_filename = f"research_data_{task_id}.xlsx"

    try:
        # 直接创建MongoDB连接
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["research_data"]

        # 更新任务状态为进行中(2)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {"state": 2, "update_time": datetime.now(), "query": query}},
            upsert=True
        )

        # 初始化数据检索器
        retriever = ResearchDataRetriever()

        # 处理研究主题
        result_df, meta_data = retriever.process_research_topic(query)

        # 将DataFrame转换为可存储的格式
        if not result_df.empty:
            # 保存DataFrame为Excel文件
            result_df.to_excel(excel_filename, index=False)

            # 上传文件到OSS
            oss_dir_path = 'ai_generate_data'
            oss_object_key = f"{oss_dir_path}/{excel_filename}"

            # 上传文件
            result = oss_bucket.put_object_from_file(oss_object_key, excel_filename)

            # 检查上传状态
            if result.status == 200:
                # 生成下载链接
                download_url = f"http://{BUCKET_NAME}.oss-cn-qingdao.aliyuncs.com/{oss_object_key}"
                logger.info(f"File {excel_filename} uploaded successfully to OSS.")

                # 删除本地文件
                if os.path.exists(excel_filename):
                    os.remove(excel_filename)
                    logger.info(f"Local file {excel_filename} deleted.")

                # 保存研究数据和元数据到MongoDB并更新任务状态为成功(1)
                collection.update_one(
                    {"user_id": user_id, "task_id": task_id},
                    {"$set": {
                        "research_data": download_url,  # 直接使用OSS下载链接作为research_data值
                        "meta_data": meta_data,
                        "state": 1,
                        "update_time": datetime.now()
                    }}
                )
                print(f"研究数据生成成功，已保存到MongoDB和OSS，数据规模: {result_df.shape}")
            else:
                # 上传失败
                if os.path.exists(excel_filename):
                    os.remove(excel_filename)  # 删除本地临时文件
                logger.error(f"Failed to upload file {excel_filename} to OSS. Status code: {result.status}")

                # 上传失败，记录错误信息
                collection.update_one(
                    {"user_id": user_id, "task_id": task_id},
                    {"$set": {
                        "research_data": "",
                        "meta_data": meta_data,
                        "state": 1,
                        "update_time": datetime.now(),
                        "message": "文件上传OSS失败"
                    }}
                )
        else:
            # 数据为空也算成功，但记录为空数据
            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {
                    "research_data": "{}",
                    "meta_data": meta_data,
                    "state": 1,
                    "update_time": datetime.now(),
                    "message": "未找到相关研究数据"
                }}
            )
            print("研究数据为空，已记录元数据")

    except Exception as e:
        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        try:
            # 直接创建新的MongoDB连接
            mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
            db = mongo_client["Newbit"]
            collection = db["research_data"]

            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {"state": 0, "error": str(e), "update_time": datetime.now()}}
            )
        except Exception as mongo_error:
            print(f"更新错误状态失败: {mongo_error}")

        print(f"生成研究数据时出错: {e}")
        logger.error(f"生成研究数据时出错: 用户 {user_id}, 任务 {task_id}, 错误: {str(e)}")
    finally:
        # 确保临时文件被删除
        if os.path.exists(excel_filename):
            try:
                os.remove(excel_filename)
                logger.info(f"Cleaned up temporary file {excel_filename}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file: {str(e)}")

def async_generate_hypothesis(hypothesis, user_id, task_id):
    """
    异步生成研究假设并存储到MongoDB

    Args:
        hypothesis: 研究假设
        user_id: 用户ID
        task_id: 任务ID
    """
    # 建立MongoDB连接
    try:
        hypothesis_agent = HypothesisWritingAgent(user_id=user_id,task_id=task_id)
        text_with_references, valid_references,pdf_urls=hypothesis_agent.generate_full_hypothesis(hypothesis)
    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["hypothesis"]

        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {"state": 0, "error": str(e), "update_time": datetime.now()}}
        )
        print(f"生成研究假设时出错: {e}")

def async_generate_full_paper(topic, user_id, task_id,mode):
    """
    异步生成研究假设并存储到MongoDB

    Args:
        论文主题: topic
        user_id: 用户ID
        task_id: 任务ID
    """
    # 建立MongoDB连接
    try:
        ai_scientist = AISocialScientist(user_id=user_id,task_id=task_id,mode=mode)
        full_paper=ai_scientist.generate_complete_research_plan(topic)
    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["full_paper"]

        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                "state": 0,  # 失败状态
                "error": str(e),
                "update_time": datetime.now()
            }}
        )
        print(f"生成全文时出错: {e}")

def async_generate_full_paper_v2(topic, user_id, task_id):
    """
    异步生成研究假设并存储到MongoDB

    Args:
        论文主题: topic
        user_id: 用户ID
        task_id: 任务ID
    """
    # 建立MongoDB连接
    try:
        paperwriter = PaperWritingAgent(KB_ID_SUMMARY, user_id, task_id)
        full_paper_text, sorted_references, sorted_pdf_urls = paperwriter.generate_full_paper(
            topic)
    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["papers"]

        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                "state": 0,  # 失败状态
                "error": str(e),
                "update_time": datetime.now()
            }}
        )
        print(f"生成全文时出错: {e}")

def async_risk_diagnose_service(file_path, user_id, task_id):
    """
    异步生成研究假设并存储到MongoDB

    Args:
        论文主题: topic
        user_id: 用户ID
        task_id: 任务ID
    """
    # 建立MongoDB连接
    try:
        risk_diagnoser = Risk_diagnose(user_id=user_id,task_id=task_id)
        result=risk_diagnoser.run_diagnose_and_filter(file_path)
    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["risk_diagnose"]

        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                "state": 0,  # 失败状态
                "error": str(e),
                "update_time": datetime.now()
            }}
        )
        print(f"学术风险诊断时出错: {e}")

def async_info_extraction_service(entity_name,field_descriptions,sources,user_id,task_id):
    """
    异步生成研究假设并存储到MongoDB

    Args:
        论文主题: topic
        user_id: 用户ID
        task_id: 任务ID
    """
    # 建立MongoDB连接
    try:
        info_extractor = InfoExtra(user_id=user_id,task_id=task_id)
        result=info_extractor.extract_and_save_entity_info(entity_name,field_descriptions,sources)
    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["info_extraction"]

        # 如果在生成过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                "state": 0,  # 失败状态
                "error": str(e),
                "update_time": datetime.now()
            }}
        )
        print(f"信息提取时出错: {e}")

def async_quote_match_service(file_path, user_id, task_id):
    """
    异步语录匹配服务并存储到MongoDB

    Args:
        file_path: 文件路径
        user_id: 用户ID
        task_id: 任务ID
    """
    # 建立MongoDB连接
    try:
        quote_matcher = PDFQuoteMatcher(user_id=user_id, task_id=task_id)
        result = quote_matcher.process_file_quotes(file_path)
        return result
    except Exception as e:
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["quote_match"]

        # 如果在处理过程中出现错误，更新任务状态为失败(0)
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                "state": 0,  # 失败状态
                "error": str(e),
                "update_time": datetime.now()
            }},
            upsert=True
        )
        mongo_client.close()
        print(f"语录匹配处理时出错: {e}")
        raise e

def process_questionnaire_data(demo_list, questionnaire, sample_size, user_id, task_id,country_of_origin=None, timestamp=None, subgroup=None, experimental_group=None):
    print("参数：", {"demo_list": demo_list, "questionnaire": questionnaire, "sample_size": sample_size, "user_id": user_id, "task_id": task_id, "country_of_origin": country_of_origin, "timestamp": timestamp, "subgroup": subgroup, "experimental_group": experimental_group})

    try:
        # 连接MongoDB
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["ai_data"]

        def read_keys(file_path):
            """Read API keys from file"""
            try:
                with open(file_path, 'r') as file:
                    # 去除每行末尾的空白字符(包括换行符)
                    return [line.strip() for line in file.readlines()]
            except FileNotFoundError:
                # 如果在默认路径找不到文件，尝试在data目录下查找
                data_path = os.path.join('data', '.env.keys')
                with open(data_path, 'r') as file:
                    return [line.strip() for line in file.readlines()]

        # Load Zhipu API keys - 使用相对路径
        keys = read_keys(".env.keys")
        api_key=random.choices(keys)[0]
        print("API-KEY:",api_key)
        zhipu_client = ZhipuAI(api_key=api_key)  # 请填写您自己的APIKey

        # 创建任务ID
        # task_id = f"questionnaire_data_{str(uuid.uuid4())}"
        print("创建任务：",task_id)
        # 遍历问卷中的每个问题，添加问题ID
        for i, question_dict in enumerate(questionnaire):
            # 添加问题ID (Q1, Q2, ..., QN)
            question_dict["q_id"] = f"Q{i+1}"

        # 在开始处理前先创建记录
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                "state": 2,
                "update_time": datetime.now(),
                "questionnaire": questionnaire,
                # "results": {},
                "api_key":api_key
            }},
            upsert=True
        )

        print(f"开始为 {sample_size} 个参与者处理问卷...")

        # 定义生成问卷回答的工作函数
        def generate_worker(participant_id):
            try:
                # 生成问卷回答并提交到API
                result = generate_questionnaire_responses(
                    participant_id=participant_id,
                    demo_list=demo_list,
                    questionnaire=questionnaire,
                    country_of_origin=country_of_origin,
                    timestamp=timestamp,
                    subgroup=subgroup,
                    experimental_group=experimental_group,
                    max_workers=10,
                    zhipu_client=zhipu_client
                )
                print(f"参与者 {participant_id} 的任务已创建")
                return {"success": True, "result": result}
            except Exception as e:
                print(f"参与者 {participant_id} 任务创建时出错: {str(e)}")
                return {"success": False, "error": str(e)}

        # 定义查询回答结果的工作函数
        def retrieve_worker(result):
            try:
                # 查询回答结果
                task_responses = retrieve_questionnaire_responses(result=result, timeout=300, max_workers=10,zhipu_client=zhipu_client)

                # 将查询结果添加到原始结果中
                if task_responses is not None:
                    result["task_responses"] = task_responses
                    # 这一步是为了复原选项数值
                    for key in result["task_responses"]:
                        result["task_responses"][key] -= 10
                    return {"success": True, "result": result}
                else:
                    print(f"参与者 {result['participant_id']} 的回答查询失败")
                    return {"success": False, "result": result, "error": "回答查询失败，可能超时或API错误"}
            except Exception as e:
                print(f"处理参与者 {result['participant_id']} 的回答时出错: {str(e)}")
                return {"success": False, "result": result, "error": str(e)}

        # 第一阶段：使用多线程为每个参与者生成问卷并提交到API
        results = []
        generation_successful = 0
        generation_failed = 0

        # 确定线程数量，可以根据实际情况调整
        max_workers = min(20, sample_size)  # 最多20个线程，避免创建过多线程

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有生成任务
            future_to_participant = {
                executor.submit(generate_worker, participant_id): participant_id
                for participant_id in range(1, sample_size + 1)
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_participant):
                participant_id = future_to_participant[future]
                try:
                    data = future.result()
                    if data["success"]:
                        results.append(data["result"])
                        generation_successful += 1
                    else:
                        generation_failed += 1
                        # 记录失败的具体错误
                        error_info = data.get("error", "未知错误")
                        results.append({"participant_id": participant_id, "error": error_info, "success": False})
                except Exception as exc:
                    error_msg = f"参与者 {participant_id} 生成时发生异常: {exc}"
                    print(error_msg)
                    generation_failed += 1
                    results.append({"participant_id": participant_id, "error": str(exc), "success": False})

        # 将收集到的结果更新到MongoDB
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                # "results": results,
                "creation_successful": generation_successful,
                "creation_failed": generation_failed,
                "state": 3,
                "update_time": datetime.now(),
            }}
        )

        print(f"任务创建完成: 成功 {generation_successful} 个, 失败 {generation_failed} 个")
        print("所有任务创建完成并存入数据库，等待API处理...")

        # 第二阶段：使用多线程查询回答结果
        print("开始查询回答结果...")

        successful_count = 0
        failed_count = 0
        updated_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 只为成功创建的结果提交查询任务
            future_to_result = {}
            for i, result in enumerate(results):
                if result.get("success", True):  # 默认为True是为了兼容原有结构
                    future_to_result[executor.submit(retrieve_worker, result)] = i
                else:
                    # 已失败的任务直接添加到更新结果中
                    updated_results.append(result)
                    failed_count += 1

            # 收集查询结果
            for future in concurrent.futures.as_completed(future_to_result):
                result_index = future_to_result[future]
                try:
                    data = future.result()
                    if data["success"]:
                        # 添加成功的结果
                        updated_results.append(data["result"])
                        successful_count += 1
                    else:
                        # 添加失败的结果，保留错误信息
                        error_info = data.get("error", "未知错误")
                        result = data["result"]
                        result["error"] = error_info
                        result["success"] = False
                        updated_results.append(result)
                        failed_count += 1
                except Exception as exc:
                    error_msg = f"查询结果 {result_index} 时发生异常: {exc}"
                    print(error_msg)
                    # 添加异常信息
                    results[result_index]["error"] = str(exc)
                    results[result_index]["success"] = False
                    updated_results.append(results[result_index])
                    failed_count += 1

        # 确定最终状态
        final_state = 1  # 默认成功完成
        error_message = None
        if failed_count > 0:
            # 部分成功部分失败
            if successful_count > 0:
                final_state = 1
                error_message = f"部分成功，{failed_count}个任务失败"
            else:
                # 全部失败
                final_state = 0
                error_message = "所有任务均失败"

        # 上传oss和生成markdown
        custom_document = {
            "user_id": user_id,
            "task_id": task_id,
            "questionnaire": questionnaire,
            "results": updated_results  # 使用上面处理好的结果
        }
        file_name = f"questionnaire_data_{task_id}.xlsx"
        local_file_path = f"{file_name}"

        markdown_table, excel_url = export_to_excel(custom_document, local_file_path)
        print(f"Excel文件已上传到OSS，URL: {excel_url}")
        print("统计表生成完成")
        # 检查文件是否存在，然后删除临时文件
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
            print(f"临时文件 {local_file_path} 已删除")
        else:
            print(f"警告: 临时文件 {local_file_path} 不存在，无法删除")

        # 将更新后的结果写回MongoDB
        update_data = {
            # "results": updated_results,
            "state": final_state,
            "update_time": datetime.now(),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "excel_url":excel_url,
            "markdown_table":markdown_table
        }

        if error_message:
            update_data["error"] = error_message

        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": update_data}
        )

        # 关闭MongoDB连接
        mongo_client.close()
        # return task_id

    except Exception as e:
        # 捕获整个处理过程中的任何异常
        error_details = traceback.format_exc()
        print(f"处理问卷数据时发生严重错误: {str(e)}\n{error_details}")

        try:
            # 尝试更新数据库，记录错误
            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {
                    "state": 0,
                    "error": str(e),
                    "error_details": error_details,
                    "update_time": datetime.now()
                }}
            )
            mongo_client.close()
        except Exception as db_error:
            # 无法更新数据库的情况
            print(f"无法更新错误状态到数据库: {str(db_error)}")

        # 返回任务ID，即使发生错误也可以通过它查询
        # return task_id


# 下载目录配置（临时存储）
DOWNLOAD_DIR = "temp_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# PDF下载函数
async def fetch_pdf(url, session):
    """下载单个PDF文件"""
    try:
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                filename = url.split('/')[-1] if '/' in url else f"document_{hash(url)}.pdf"
                content = await response.read()
                return url, filename, content, True
            return url, None, None, False
    except Exception as e:
        logger.error(f"下载文件失败: {url}, 错误: {str(e)}")
        return url, None, None, False


# 异步处理PDF批量下载函数
async def async_process_download(pdf_urls, user_id, task_id):
    """
    异步处理PDF批量下载任务，上传到OSS并存储到MongoDB

    Args:
        pdf_urls: PDF链接列表
        user_id: 用户ID
        task_id: 任务ID
    """
    # 先定义file_path变量，避免在异常处理中访问未定义的变量
    file_path = os.path.join(DOWNLOAD_DIR, f"{task_id}.zip")

    try:
        # 创建MongoDB连接
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["download_tasks"]

        # 更新任务状态为进行中
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {
                "state": 2,  # 进行中
                "message": "正在处理下载请求",
                "update_time": datetime.now().isoformat(),  # 使用isoformat()转换为字符串
                "query": f"批量下载 {len(pdf_urls)} 个PDF文件"  # 确保有query字段
            }},
            upsert=True
        )

        logger.info(f"任务 {task_id} 开始处理，用户: {user_id}")

        # 创建zip文件名
        zip_filename = f"{task_id}.zip"
        file_path = os.path.join(DOWNLOAD_DIR, zip_filename)

        # 确保临时目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 下载并打包文件
        successful_files = []
        failed_urls = []

        async with aiohttp.ClientSession() as session:
            # 并行下载所有PDF
            download_tasks = []
            for url in pdf_urls:
                task = asyncio.create_task(fetch_pdf(url, session))
                download_tasks.append(task)

            results = await asyncio.gather(*download_tasks)

            # 创建zip文件
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for url, pdf_filename, content, success in results:
                    if success and content:
                        zip_file.writestr(pdf_filename, content)
                        successful_files.append({
                            "url": url,
                            "filename": pdf_filename
                        })
                    else:
                        failed_urls.append(url)

        # 计算统计数据
        total_files = len(pdf_urls)
        success_count = len(successful_files)
        failed_count = len(failed_urls)

        # 上传到OSS并获取下载URL
        logger.info(f"正在将ZIP文件 {zip_filename} 上传到OSS")
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise ValueError(f"ZIP文件不存在: {file_path}")

            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"ZIP文件大小为0: {file_path}")

            # 上传到OSS
            oss_dir_path = 'pdf_downloads'
            download_url = upload_file_to_oss(file_path, BUCKET_NAME, zip_filename)
            logger.info(f"ZIP文件已上传到OSS，下载URL: {download_url}")

            # 删除本地临时文件
            os.remove(file_path)
            logger.info(f"本地临时文件 {file_path} 已删除")

            # 更新任务状态为已完成，保存关键字段
            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {
                    "state": 1,  # 成功
                    "message": f"下载完成，成功: {success_count}/{total_files} 文件",
                    "update_time": datetime.now().isoformat(),  # 使用isoformat()转换为字符串
                    "download_url": download_url,  # 直接存储这两个关键字段
                    "total_files": total_files,
                    # 以下字段可以保留，为了详细记录
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "successful_files": successful_files,
                    "failed_urls": failed_urls
                }}
            )

            logger.info(f"任务 {task_id} 处理完成，用户: {user_id}，成功数: {success_count}/{total_files}")

        except Exception as e:
            error_msg = f"上传到OSS失败: {str(e)}"
            logger.error(error_msg)

            # 如果上传失败，更新任务状态
            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {
                    "state": 0,  # 失败
                    "message": error_msg,
                    "error": error_msg,
                    "update_time": datetime.now().isoformat()  # 使用isoformat()转换为字符串
                }}
            )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"任务 {task_id} 处理失败，用户: {user_id}，错误: {error_msg}")

        # 更新任务状态为失败
        try:
            # 直接创建新的MongoDB连接
            mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
            db = mongo_client["Newbit"]
            collection = db["download_tasks"]

            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {
                    "state": 0,  # 失败
                    "message": f"处理失败: {error_msg}",
                    "error": error_msg,
                    "update_time": datetime.now().isoformat()  # 使用isoformat()转换为字符串
                }}
            )
        except Exception as mongo_error:
            logger.error(f"更新错误状态失败: {mongo_error}")

        # 清理可能存在的临时文件
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"已清理临时文件: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"清理临时文件失败: {cleanup_error}")

def process_aigc(user_id, task_id, task_type="review"):
    """
    基于用户ID和任务ID从MongoDB获取文档内容，并对main_text进行AIGC率降低处理

    Args:
        user_id: 用户ID
        task_id: 任务ID
        task_type: 任务类型，默认为"review"，目前仅支持"review"类型

    Returns:
        dict: 包含处理结果的字典
    """
    print(f"获取并处理文档内容: 用户 {user_id}, 任务 {task_id}, 类型 {task_type}")

    try:
        # 创建MongoDB连接
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]

        # 获取文档
        collection = db["review"]
        document = collection.find_one({"user_id": user_id, "task_id": task_id})

        if not document:
            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {"aigc_erro": "任务不存在"}}
            )
            print("任务不存在")

        # 获取main_text字段
        main_text = document.get("review_text", "")["main_text"]
        if not main_text:
            collection.update_one(
                {"user_id": user_id, "task_id": task_id},
                {"$set": {"aigc_erro": "main_text不存在"}}
            )
            print("main_text不存在")

        # 处理main_text，降低AIGC率
        processed_text = reduce_aigc_rate(main_text)+"\n\n"+document.get("review_text", "")["reference_text"]
        # 更新文档中的main_text字段
        collection.update_one(
            {"user_id": user_id, "task_id": task_id},
            {"$set": {"processed_review_text": processed_text}}
        )
        print("AIGC处理完成")


    except Exception as e:
        print(f"AIGC处理出错:{e}")