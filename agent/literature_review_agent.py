import concurrent
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import jieba

from es_search import *
from client import *
from editor import *
from llm_call import *
from datetime import datetime

class TopicUnderstandingAgent:
    """
    选题理解和议题分解Agent
    """

    def __init__(self):
        self.name = "议题分解Agent"

    def decompose_topic(self, main_topic: str) -> List[str]:
        """
        将主题分解为多个子议题
        Args:
            main_topic: 主要研究主题
        Returns:
            子议题列表
        """
        print(f"{self.name}: 正在分解主题 '{main_topic}'...")

        # 使用Claude API来分解主题
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

        try:
            response = handler.call_llm(provider="openai",
                                        prompt=prompt,
                                        model="gpt-4o-mini",
                                        max_tokens=1000)

            # 检查响应是否包含API调用失败信息
            if "API call failed" in str(response):
                raise Exception("API call failed detected in response")

        except:
            try:
                response = handler.call_llm(provider="zhipuai",
                                            prompt=prompt,
                                            model="glm-4.5-air",
                                            max_tokens=1000)
                # 检查响应是否包含API调用失败信息
                if "API call failed" in str(response):
                    raise Exception("API call failed detected in response")

            except:
                response = handler.call_llm(provider="ark",
                                          prompt=prompt,
                                          model="doubao-1-5-lite-32k-250115",
                                          max_tokens=1000)
        # 处理返回的文本，分割成列表
        subtopics = [topic.strip() for topic in response.strip().split('\n') if topic.strip()]

        print(f"{self.name}: 主题已分解为 {len(subtopics)} 个子议题")
        return subtopics


class LiteratureCollectionAgent:
    """
    文献收集Agent
    """

    def __init__(self, kb_id: str, mode=None):
        self.name = "文献收集Agent"
        self.kb_id = kb_id
        self.mode = mode

    def generate_optimal_query(self, subtopic: str, content_type: str) -> str:
        """
        为特定议题和内容类型生成最优查询
        Args:
            subtopic: 子议题
            content_type: 内容类型 (concept, theory, findings)
        Returns:
            优化后的查询字符串
        """
        try:
            if content_type == "concept":
                prompt = f"""
我需要为文献检索生成最佳的查询词。我的目标是查找有关"{subtopic}"的核心概念和定义。
请生成一个简短、精确的查询字符串，包含5-7个关键词，用空格分隔。
只返回查询字符串，不要有其他任何解释或额外文字。
                """
            elif content_type == "theory":
                prompt = f"""
我需要为文献检索生成最佳的查询词。我的目标是查找有关"{subtopic}"的理论基础和模型。
请生成一个简短、精确的查询字符串，包含5-7个关键词，用空格分隔。
只返回查询字符串，不要有其他任何解释或额外文字。
                """
            else:  # findings
                prompt = f"""
我需要为文献检索生成最佳的查询词。我的目标是查找有关"{subtopic}"的研究发现和实证结果。
请生成一个简短、精确的查询字符串，包含5-7个关键词，用空格分隔。
只返回查询字符串，不要有其他任何解释或额外文字。
                """
            try:
                response = handler.call_llm(provider="openai",
                                    prompt=prompt,
                                    model="gpt-4o-mini",
                                    max_tokens=20)
                # 检查响应是否包含API调用失败信息
                if "API call failed" in str(response):
                    raise Exception("API call failed detected in response")
                    seg_list = jieba.cut(subtopic, cut_all=False)
                    segmented_query = " ".join(seg_list)
                    return segmented_query
                else:
                    return response
            except:
                seg_list = jieba.cut(subtopic, cut_all=False)
                segmented_query = " ".join(seg_list)
                return segmented_query
        except Exception as e:
            print(f"使用查询生成 API生成查询时出错: {e}")
            seg_list = jieba.cut(subtopic, cut_all=False)
            segmented_query = " ".join(seg_list)
            return segmented_query

        # 备选方案，直接组合关键词
        if content_type == "concept":
            return f"{subtopic} 定义 概念 特征"
        elif content_type == "theory":
            return f"{subtopic} 理论 模型 框架"
        else:  # findings
            return f"{subtopic} 研究发现 实证研究 结论"

    def is_relevant_to_subtopic(self, title: str, abstract: str, subtopic: str) -> int:
        """
        使用OpenAI GPT-4o-mini判断文献是否真正属于某子议题
        Args:
            title: 文献标题
            abstract: 文献摘要
            subtopic: 子议题
        Returns:
            1表示属于该议题，0表示不属于
        """
        return 1
#         try:
#             prompt = f"""
# 以下文献是检索系统为"{subtopic}"这个研究议题检索到的一篇文献的<文献信息>。
# 请根据其中的文献标题和文献摘要判断该文献是否真正属于"{subtopic}"这个研究议题。
#
# <文献信息>
# 文献标题：{title}
# 文献摘要：{abstract}
# </文献信息>
#
# 你需要严格遵循我提供的<思维步骤>来做判断。
#
# <思维步骤>
# 第一步，先做概念判断：判断提供的文献是否围绕给定研究议题中的*核心概念*展开讨论。例如，对于"企业数字化转型的效应研究"这一议题来说，这一步首先需要判断文献内容是否围绕"企业数字化转型"这个关键概念展开讨论。若是，则进入下一步判断；若不是，则直接判定为"不属于"；
# 第二步，再做方向判断：判断提供的文献是否在*方向*上与给定研究议题一致。例如，对于"企业数字化转型的效应研究"这一议题来说，第一步确定了一篇文献是围绕"企业数字化转型"这个关键概念展开讨论，那么这一步则需要进一步确定这篇文献是不是属于"效应研究"的范畴。若是，则代表提供的文献属于给定的研究议题，若不是，则判定为"不属于"。
# </思维步骤>
#
# 判断结果只返回0或1，1表示属于，0表示不属于。不要返回任何其他内容，也无需解释理由。
#
# 判断结果："""
#
#
#             try:
#                 result = handler.call_llm(provider="zhipuai",
#                                           prompt=prompt,
#                                           model="glm-4.5-air",
#                                           max_tokens=100,
#                                           temperature=0.1)
#                 # 检查响应是否包含API调用失败信息
#                 if "API call failed" in str(result):
#                     raise Exception("API call failed detected in response")
#
#             except:
#                 try:
#                     result = handler.call_llm(provider="ark",
#                                                 prompt=prompt,
#                                                 model="doubao-1-5-lite-32k-250115",
#                                                 max_tokens=100,
#                                                 temperature=0.1)
#                     # 检查响应是否包含API调用失败信息
#                     if "API call failed" in str(result):
#                         raise Exception("API call failed detected in response")
#
#                 except:
#                     result = handler.call_llm(provider="openai",
#                                               prompt=prompt,
#                                               model="gpt-4o-mini",
#                                               max_tokens=100,
#                                               temperature=0.1)
#
#
#             # 确保结果是0或1
#             if "1" not in result:
#                 return 0
#             else:
#                 return 1
#
#         except Exception as e:
#             print(f"调用大模型判断文献相关性失败: {e}")
#             # 如果API调用失败，默认保留该文献
#             return 1

    def collect_single_subtopic_literature(self, subtopic: str, min_docs_per_category: int = 15, language=None) -> Dict[
        str, List[Dict]]:
        print(f"{self.name}: 开始收集子议题 '{subtopic}' 的文献资料...")

        content_type_list = ["findings"]
        subtopic_results = {content_type: [] for content_type in content_type_list}

        # 用于全局去重的集合，跟踪所有已收集的文献（基于标题和作者）
        all_collected_docs_keys = set()

        for content_type in content_type_list:
            print(f"{self.name}: 开始收集 '{subtopic}' 的{content_type}类文献...")
            chunk_type = "summary" if content_type == "findings" else f"{content_type}_answer"

            query = self.generate_optimal_query(subtopic, content_type)
            all_relevant_docs = []
            used_queries = set([query])
            first_batch_results = []  # 新增：保存第一批结果

            max_iterations = 3
            for iteration in range(max_iterations):
                search_results = es_service.search_documents(
                    query=query,
                    kb_id=self.kb_id,
                    chunk_type=[chunk_type],
                    top_k=30,
                    language=language
                )

                # 如果是第一轮，保存全部结果
                if iteration == 0:
                    first_batch_results = search_results.copy()  # 保存第一批结果

                relevant_docs_lock = threading.Lock()

                def process_document(doc):
                    title = doc.get("title", "")
                    authors = doc.get("authors", "")
                    abstract = doc.get("abstract", "")

                    # 创建文献唯一标识（基于标题和作者）
                    doc_key = (title, authors)

                    with relevant_docs_lock:
                        # 检查是否已经在当前内容类型的结果中
                        doc_already_included = any(
                            existing_doc.get("title") == title
                            for existing_doc in all_relevant_docs
                        )

                        # 检查是否已在全局收集的文献中（跨内容类型去重）
                        doc_already_collected = doc_key in all_collected_docs_keys

                    # 只有当文献既不在当前结果中，也不在全局收集中时，才考虑添加
                    if not doc_already_included and not doc_already_collected:
                        relevance_score = self.is_relevant_to_subtopic(title, abstract, subtopic)
                        if relevance_score == 1:
                            with relevant_docs_lock:
                                all_relevant_docs.append(doc)
                                # 添加到全局去重集合
                                all_collected_docs_keys.add(doc_key)
                                return True
                    return False

                with ThreadPoolExecutor(max_workers=30) as executor:
                    future_to_doc = {executor.submit(process_document, doc): doc for doc in search_results}
                    for future in as_completed(future_to_doc):
                        try:
                            future.result()
                        except Exception as exc:
                            print(f"{self.name}: 处理文档时发生错误: {exc}")

                # 检查是否已收集到足够文献
                print(min_docs_per_category)
                if len(all_relevant_docs) >= min_docs_per_category:
                    print(f"{self.name}: 已收集到足够数量的{content_type}类文献 ({len(all_relevant_docs)}篇)")
                    break

                # 准备下一轮查询
                if iteration < max_iterations - 1:
                    new_query_title = ""

                    # 修改点1：如果当前轮次没有通过核查的文献，使用这一轮的第一篇文献标题
                    if len(all_relevant_docs) == 0:
                        if search_results:  # 确保有搜索结果
                            new_query_title = search_results[0].get("title", "")
                            print(
                                f"{self.name}: 当前轮次无通过核查文献，使用第一篇文献标题作为新查询词: {new_query_title}")
                    else:
                        new_query_title = all_relevant_docs[-1].get("title", "")

                    if new_query_title:
                        new_query = f"{subtopic} {new_query_title}"
                        if new_query not in used_queries:
                            query = new_query
                            used_queries.add(query)
                            print(
                                f"{self.name}: 已收集{len(all_relevant_docs)}篇相关{content_type}类文献，不足{min_docs_per_category}篇，使用新查询继续搜索: {query}")
                        else:
                            print(
                                f"{self.name}: 无法生成新的查询词，已收集{len(all_relevant_docs)}篇相关{content_type}类文献")
                            break
                    else:
                        print(f"{self.name}: 无法生成有效的查询词，终止搜索")
                        break

            # 修改点2：如果5轮后仍为0篇，使用第一批结果的前15篇
            # 但需要进行去重检查
            if len(all_relevant_docs) == 0 and first_batch_results:
                print(
                    f"{self.name}: 5轮搜索后仅收集到{len(all_relevant_docs)}篇文献，使用第一批结果的前{min_docs_per_category}篇")

                # 从第一批结果中筛选未收集过的文献
                filtered_first_batch = []
                for doc in first_batch_results:
                    doc_key = (doc.get("title", ""), doc.get("authors", ""))
                    if doc_key not in all_collected_docs_keys:
                        filtered_first_batch.append(doc)
                        all_collected_docs_keys.add(doc_key)

                        # 如果已经达到所需数量，则停止筛选
                        if len(filtered_first_batch) >= min_docs_per_category:
                            break

                all_relevant_docs = filtered_first_batch[:min_docs_per_category]

            subtopic_results[content_type] = all_relevant_docs

        return subtopic_results

    def collect_single_subtopic_literature_by_weight(self, subtopic: str, min_docs_per_category: int = 15,
                                           chinese_weight: float = 0.5) -> Dict[str, List[Dict]]:
        print(f"{self.name}: 开始收集子议题 '{subtopic}' 的文献资料...")
        print(f"文献比例模式：中文文献比例为{chinese_weight}")
        content_type_list = ["findings"]
        subtopic_results = {content_type: [] for content_type in content_type_list}

        for content_type in content_type_list:
            print(f"{self.name}: 开始收集 '{subtopic}' 的{content_type}类文献...")
            chunk_type = "summary" if content_type == "findings" else f"{content_type}_answer"

            # 计算中文文献的最小数量
            min_chinese_docs = int(min_docs_per_category * chinese_weight)
            print(
                f"{self.name}: 计划收集中文文献 {min_chinese_docs} 篇，英文文献 {min_docs_per_category - min_chinese_docs} 篇")

            # 第一轮：收集中文文献
            chinese_docs = self._collect_docs_by_language(subtopic, content_type, chunk_type, min_chinese_docs, "中文",
                                                          max_iterations=5)

            # 第二轮：收集英文文献
            remaining_docs_needed = min_docs_per_category - len(chinese_docs)
            english_docs = []
            if remaining_docs_needed > 0:
                english_docs = self._collect_docs_by_language(subtopic, content_type, chunk_type, remaining_docs_needed,
                                                              "英文", max_iterations=5)

            # 合并文献结果
            all_relevant_docs = chinese_docs + english_docs
            subtopic_results[content_type] = all_relevant_docs

            print(
                f"{self.name}: 已收集 {len(chinese_docs)} 篇中文文献和 {len(english_docs)} 篇英文文献，总计 {len(all_relevant_docs)} 篇")

        return subtopic_results

    def _collect_docs_by_language(self, subtopic: str, content_type: str, chunk_type: str, min_docs: int, language: str,
                                  max_iterations: int = 5) -> List[Dict]:
        print(f"{self.name}: 开始收集 '{subtopic}' 的{language}{content_type}类文献...")

        query = self.generate_optimal_query(subtopic, content_type)
        all_relevant_docs = []
        used_queries = set([query])
        first_batch_results = []  # 保存第一批结果

        for iteration in range(max_iterations):
            search_results = es_service.search_documents(
                query=query,
                kb_id=self.kb_id,
                chunk_type=[chunk_type],
                top_k=30,
                language=language  # 传入语言参数
            )

            # 如果是第一轮，保存全部结果
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
                future_to_doc = {executor.submit(process_document, doc): doc for doc in search_results}
                for future in as_completed(future_to_doc):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"{self.name}: 处理文档时发生错误: {exc}")

            # 检查是否已收集到足够文献
            if len(all_relevant_docs) >= min_docs:
                print(f"{self.name}: 已收集到足够数量的{language}{content_type}类文献 ({len(all_relevant_docs)}篇)")
                break

            # 准备下一轮查询
            if iteration < max_iterations - 1:
                new_query_title = ""

                # 如果当前轮次没有通过核查的文献，使用这一轮的第一篇文献标题
                if len(all_relevant_docs) == 0:
                    if search_results:  # 确保有搜索结果
                        new_query_title = search_results[0].get("title", "")
                        print(f"{self.name}: 当前轮次无通过核查文献，使用第一篇文献标题作为新查询词: {new_query_title}")
                else:
                    new_query_title = all_relevant_docs[-1].get("title", "")

                if new_query_title:
                    new_query = f"{subtopic} {new_query_title}"
                    if new_query not in used_queries:
                        query = new_query
                        used_queries.add(query)
                        print(
                            f"{self.name}: 已收集{len(all_relevant_docs)}篇相关{language}{content_type}类文献，不足{min_docs}篇，使用新查询继续搜索: {query}")
                    else:
                        print(
                            f"{self.name}: 无法生成新的查询词，已收集{len(all_relevant_docs)}篇相关{language}{content_type}类文献")
                        break
                else:
                    print(f"{self.name}: 无法生成有效的查询词，终止搜索")
                    break

        # 如果多轮后仍未收集到足够文献，使用第一批结果
        if len(all_relevant_docs) == 0 and first_batch_results:
            print(
                f"{self.name}: {max_iterations}轮搜索后仅收集到{len(all_relevant_docs)}篇{language}文献，使用第一批结果的前{min_docs}篇")
            all_relevant_docs = first_batch_results[:min_docs]

        return all_relevant_docs

    def collect_literature(self, subtopics: List[str], language=None, chinese_weight=None) -> Dict[
        str, Dict[str, List[Dict]]]:
        """
        收集与子议题相关的文献 - 多线程版本
        Args:
            subtopics: 子议题列表
            language: 语言参数
            chinese_weight: 中文权重参数
        Returns:
            按议题和内容类型组织的文献资料
        """
        print(f"{self.name}: 开始在知识库：{self.kb_id}收集文献资料...")

        def process_subtopic_task(subtopic):
            """处理单个子议题的任务函数"""
            try:
                if chinese_weight is None:
                    # 使用单议题收集函数
                    if self.mode == 1:
                        print(f"长文模式 - 处理子议题: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature(
                            subtopic=subtopic,
                            min_docs_per_category=30,
                            language=language
                        )
                    else:
                        print(f"普通模式 - 处理子议题: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature(
                            subtopic=subtopic,
                            language=language
                        )
                else:
                    # 使用带权重的单议题收集函数
                    if self.mode == 1:
                        print(f"长文模式 - 处理子议题: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature_by_weight(
                            subtopic=subtopic,
                            min_docs_per_category=30,
                            chinese_weight=chinese_weight
                        )
                    else:
                        print(f"普通模式 - 处理子议题: {subtopic}")
                        subtopic_result = self.collect_single_subtopic_literature_by_weight(
                            subtopic=subtopic,
                            chinese_weight=chinese_weight
                        )

                print(f"完成子议题 '{subtopic}' 的文献收集")
                return (subtopic, subtopic_result)

            except Exception as e:
                print(f"处理子议题 '{subtopic}' 时发生错误: {str(e)}")
                return (subtopic, {})

        # 使用线程池并行处理所有子议题
        results = {}
        max_workers = min(len(subtopics), 10)  # 限制最大线程数为10，避免过多并发

        print(f"启动 {len(subtopics)} 个子议题的文献收集任务，使用 {max_workers} 个线程...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_subtopic = {
                executor.submit(process_subtopic_task, subtopic): subtopic
                for subtopic in subtopics
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_subtopic):
                subtopic = future_to_subtopic[future]
                try:
                    subtopic_name, subtopic_result = future.result()
                    results[subtopic_name] = subtopic_result
                except Exception as e:
                    print(f"子议题 '{subtopic}' 的任务执行失败: {str(e)}")
                    results[subtopic] = {}

        print(f"所有子议题的文献收集完成，共处理 {len(results)} 个子议题")
        return results

class WritingAgent:
    """
    写作Agent - 使用AsyncAnthropic的异步版本
    """

    def __init__(self,mode=None):
        self.name = "写作Agent"
        self.mode = mode
    def transform_list_to_apa_format(self, data_list):
        """
        将文献数据列表转换为APA格式的引用和参考文献
        Args:
            data_list (list): 包含文献信息的字典列表
        Returns:
            list: 包含APA格式引用和参考文献的字典列表
        """
        result_list = []

        for item in data_list:
            # Create a new dictionary for this item
            result_item = {}

            # Extract necessary information
            authors = item.get('authors', '')
            year = item.get('year', '')
            title = item.get('title', '')
            journal = item.get('journal', '')
            content = item.get('content_with_weight', '')  # 使用'content_with_weight'

            # Extract additional fields for reference using the correct key names
            volume = item.get('vO', '')  # 卷号
            issue = item.get('issue', '')  # 期号
            pages = item.get('page_range', '')  # 页码范围
            pdf_url=item.get('pdf_url','')

            # Process authors according to the rules
            author_list = authors.split(";")

            # For English authors, keep only the last name
            # For Chinese authors, keep the full name
            processed_authors = []
            for author in author_list:
                author = author.strip()
                if author:
                    # Check if author name contains a comma (likely English format: Last, First)
                    if "," in author:
                        processed_authors.append(author.split(",")[0].strip())
                    else:
                        processed_authors.append(author)

            # Format author citation based on number of authors
            if len(processed_authors) == 0:
                formatted_author = ""
            elif len(processed_authors) == 1:
                formatted_author = processed_authors[0]
            elif len(processed_authors) == 2:
                formatted_author = f"{processed_authors[0]} & {processed_authors[1]}"
            else:  # 3 or more authors
                formatted_author = f"{processed_authors[0]} et al."

            # Create APA citation (Author, Year)
            apa_citation = f"({formatted_author}, {year})"

            # Create content with citation
            content_with_citation = f"{content} {apa_citation}"

            # Create full APA reference
            # For the reference, we'll use all authors
            if len(processed_authors) == 0:
                authors_for_reference = ""
            elif len(processed_authors) == 1:
                authors_for_reference = processed_authors[0]
            elif len(processed_authors) == 2:
                authors_for_reference = f"{processed_authors[0]} & {processed_authors[1]}"
            else:
                # In full references, APA uses up to 20 authors before using et al.
                # For simplicity, we'll include all authors
                authors_for_reference = ", ".join(processed_authors[:-1]) + ", & " + processed_authors[-1]

            # Build the reference with volume, issue, and pages if available
            apa_reference = f"{authors_for_reference}. ({year}). {title}. {journal}"

            # Add volume if available
            if volume:
                apa_reference += f", {volume}"

                # Add issue if available (only if volume is also available)
                if issue:
                    apa_reference += f"({issue})"

            # Add pages if available
            if pages:
                apa_reference += f", {pages}"

            # Add final period
            apa_reference += "."
            apa_reference+= f"[PDF_URL_START]{pdf_url}[PDF_URL_END]"

            # Add to result dictionary
            result_item['apa_citation'] = apa_citation
            result_item['content_with_citation'] = content_with_citation
            result_item['original_content'] = content
            result_item['apa_reference'] = apa_reference

            # Add this dictionary to the result list
            result_list.append(result_item)

        return result_list

    def generate_paragraph(self, content_list: list, content_type: str, subtopic: str) -> str:
        """
        基于文献内容生成段落 (异步版本)
        """
        # 先将文献转换为APA格式
        apa_formatted = self.transform_list_to_apa_format(content_list)


        if content_type == "concept":
            # 获取所有带引用的内容
            combined_content = "\n\n".join(
                [str(item["content_with_citation"]).replace("\n", " ") for item in apa_formatted])

            prompt = f"""
你是一位文献综述写作专家。请基于<参考资料>，写一段关于"{subtopic}"的既有概念定义的段落。
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
</写作原则>

<示例结果>
[CON_START]目前，已有诸多文献对气候变化叙事这一概念做出定义。例如，Paschen & Ison (2014)提出，气候变化叙事是指以故事化方式讲述气候变化信息和政策，目的是获取各利益相关者信任并指导特定类型行动。Curran (2012)提出，气候变化叙事是指传播气候变化信息，以增进公众对气候变化政策与知识的理解。Bushell et al. (2017)提出，气候变化叙事是指以叙事传播方式讲述气候变化的未来结果，目的是促进各相关方进行战略规划以应对未来风险。Holden et al. (2021)提出，气候变化叙事是指以故事性和修辞性方式，使用场景、寓意、情节、角色等元素介绍气候变化缓解路径，以促使公众改变能源行为。Jepson (2019)提出，气候变化叙事是指讲述"环境现状、这种现状如何影响人类及人类需要做什么"的故事，从而以有意义和有说服力的方式建构环境议题。总体来看，气候变化叙事的定义总共涉及三大核心要点：一是利用叙事元素，二是围绕特定议题，三是达到特定目的。不同定义的差异主要体现在这三大核心要点上。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""
        elif content_type == "theory":
            combined_content = "\n\n".join(
                [str(item["content_with_citation"]).replace("\n", " ") for item in apa_formatted])

            prompt = f"""
你是一位文献综述写作专家。请基于<参考资料>，写一段关于"{subtopic}"的既有理论视角的段落。
在写作过程中，严格遵循<写作原则>和<写作要求>，同时模仿<示例结果>中的语言风格，包括句式结构、句与句之间关系、措辞习惯。

<写作原则>
第一步，段落首句概述现有研究所用理论视角的数量。
第二步，每个理论视角都采用两部分来论述：一是理论的内涵及观点，二是理论相关研究示例（若有多个研究用同一理论则提供多个示例）的概述。
第三步，段落最后一句或两句总结现有理论视角对于研究者的启示。
</写作原则>

<写作要求>
1.用[CON_START]和[CON_END]包裹结果。
2.保持严谨的学术风格，用一段流畅、逻辑清晰的话来阐述，请勿换行。
3.尽可能使用所有的文献资料，每份文献资料都来之不易。
4.尽可能保留文献材料原文中的细节，原文中的所有细节信息对于读者理解文献都至关重要。
5.参考资料中一行为一篇文献的材料，后面的（姓名，年份）为Citation，需在生成的正文中以合适的形式保留（见示例结果）。
</写作要求>

<示例结果>
[CON_START]既有数字政府的影响因素研究主要立足于五个理论视角。第一个理论是政策企业家理论，该理论强调特定个体在政策过程中的关键推动作用。例如，Mergel (2019)基于该理论分析和检验了机构领导人在推动政府数字化转型中的作用。第二个理论是TOE框架，该框架强调技术、组织和环境三大因素对创新采纳的综合影响。例如，Chen & Hsiao (2014)基于该理论分析和检验了技术基础设施、机构领导支持和法律法规环境对数字政府推进的影响。第三个理论是政策创新扩散理论，该理论关注政策如何在政府间传播。例如，Zhang et al. (2014)基于该理论，提出了一个囊括地理临近性和政治相似性等因素的分析框架来研究电子政务实践的区域扩散。第四个理论是制度理论，该理论探讨正式和非正式制度对组织行为的影响。例如，Luna-Reyes & Gil-Garcia (2011)基于制度理论，分析和验证了法律框架、组织结构和行政文化对数字政府项目的影响。第五个理论是资源依赖理论，该理论强调组织为获取关键资源而采取的战略行为。例如，Cordella & Willcocks (2010)基于该理论分析和验证了资源约束如何促使政府机构通过技术外包和战略联盟以维持数字服务高效运行。通过对数字政府影响因素研究的五大主要理论视角的梳理，我们可以看出这些理论从不同层面共同构建了对数字政府发展机制的系统性理解。这些理论视角启示研究者需要同时关注领导推动、组织能力、外部环境、制度约束和资源获取等多方面因素。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""
        else:  # findings
            combined_content = "\n\n".join(
                [str(item["original_content"]).replace("\n", " ") for item in apa_formatted])

            if self.mode == 1:
                prompt = f"""
你是一位文献综述写作专家。请基于<参考资料>，写一段关于"{subtopic}"的既有研究内容的段落。
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
尽可能保留文献材料原文中的细节，原文中的所有细节信息对于读者理解文献都至关重要。
参考资料中一行为一篇文献的材料，保持文献在正文中的引用方式为句首的“姓名（年份）”，不要在句子后面重复出现（姓名，年份）。
</写作原则>

<示例结果>
[CON_START]现有治理者公众信任的影响因素研究按照所关注因素的差异可以分为三类。\n\n第一类研究关注的是治理条件对治理者公众信任的影响，前者主要包括治理过程、治理表现、治理情境等因素。例如，在治理过程上，Wang和van Wart（2007）采用路径分析方法分析了2000年美国城市政府层面的调查数据，结果显示政府治理过程的公众参与程度与公众信任水平正相关。在治理表现上，Seyd（2015）采用结构方程模型分析了2008年英国公民对政治人物的信任度数据，结果显示信任度主要由政治人物实际表现决定。在治理情境上，Houston等（2016）采用了多层二元Logit模型分析了2006年国际社会调查项目（ISSP）的21个国家样本数据，结果显示宗教多样性越低，公众对公务员的信任度越高。\n\n第二类研究关注的是沟通特征对治理者公众信任的影响，前者包括沟通程度、沟通策略等因素。例如，在沟通程度上，Park等（2016）采用结构方程模型分析了2012年韩国公民与政府互动的Twitter数据，结果显示政府领导与公民的沟通可以增加他们的政府信任。在沟通策略上，Alon-Barkat（2020）以环境政策为场景，基于以色列公民样本开展了随机调查实验，结果显示，包含真实象征元素（如标志、颜色和名人代言）的沟通可以增加公民对政策的信任。\n\n第三类研究关注的是个体特征对治理者公众信任的影响，前者包括社会身份、认知行为等因素。例如，在社会身份上，LeBas（2020）采用线性回归分析了2010年尼日利亚11个城市的问卷调查数据，研究显示少数族裔对当地官员的信任度显著更低。在认知行为上，Mizrahi 等（2021）采用多层线性分析了2018年以色列公民的代表性抽样调查数据，结果显示公众对公共部门的信任程度与人们对紧急情况的恐惧程度相关。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""

            else:
                prompt = f"""
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
参考资料中一行为一篇文献的材料，保持文献在正文中的引用方式为句首的“姓名（年份）”，不要在句子后面重复出现（姓名，年份）。
</写作原则>

<示例结果>
[CON_START]现有治理者公众信任的影响因素研究按照所关注因素的差异可以分为三类。第一类研究关注的是治理条件对治理者公众信任的影响，前者主要包括治理过程、治理表现、治理情境等因素。例如，在治理过程上，Wang和van Wart（2007）采用路径分析方法分析了2000年美国城市政府层面的调查数据，结果显示政府治理过程的公众参与程度与公众信任水平正相关。在治理表现上，Seyd（2015）采用结构方程模型分析了2008年英国公民对政治人物的信任度数据，结果显示信任度主要由政治人物实际表现决定。在治理情境上，Houston等（2016）采用了多层二元Logit模型分析了2006年国际社会调查项目（ISSP）的21个国家样本数据，结果显示宗教多样性越低，公众对公务员的信任度越高。第二类研究关注的是沟通特征对治理者公众信任的影响，前者包括沟通程度、沟通策略等因素。例如，在沟通程度上，Park等（2016）采用结构方程模型分析了2012年韩国公民与政府互动的Twitter数据，结果显示政府领导与公民的沟通可以增加他们的政府信任。在沟通策略上，Alon-Barkat（2020）以环境政策为场景，基于以色列公民样本开展了随机调查实验，结果显示，包含真实象征元素（如标志、颜色和名人代言）的沟通可以增加公民对政策的信任。第三类研究关注的是个体特征对治理者公众信任的影响，前者包括社会身份、认知行为等因素。例如，在社会身份上，LeBas（2020）采用线性回归分析了2010年尼日利亚11个城市的问卷调查数据，研究显示少数族裔对当地官员的信任度显著更低。在认知行为上，Mizrahi 等（2021）采用多层线性分析了2018年以色列公民的代表性抽样调查数据，结果显示公众对公共部门的信任程度与人们对紧急情况的恐惧程度相关。[CON_END]
</示例结果>

<参考资料>
{combined_content}
</参考资料>

结果："""
        try:
            response = handler.call_llm(provider="zhipuai",
                                      prompt=prompt,
                                      model="glm-4.5-air",
                                      max_tokens=8000)
            # 检查响应是否包含API调用失败信息
            if "API call failed" in str(response):
                raise Exception("API call failed detected in response")
        except:
            try:
                response = handler.call_llm(provider="ark",
                                            prompt=prompt,
                                            model="doubao-1-5-lite-32k-250115",
                                            max_tokens=8000)
                # 检查响应是否包含API调用失败信息
                if "API call failed" in str(response):
                    raise Exception("API call failed detected in response")
            except:
                try:
                    response = handler.call_llm(provider="siliconflow",
                                              prompt=prompt,
                                              model="THUDM/GLM-4-32B-0414",
                                              max_tokens=8000)
                    # 检查响应是否包含API调用失败信息
                    if "API call failed" in str(response):
                        raise Exception("API call failed detected in response")
                except:
                    response = handler.call_llm(provider="openai",
                                                prompt=prompt,
                                                model="gpt-4o",
                                                max_tokens=8000)

        pattern = r'\[CON_START\](.*?)\[CON_END\]'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            # 如果找到匹配的内容，返回匹配部分
            extracted_content = match.group(1).strip()
            if self.mode == 1:
                extracted_content=extracted_content.replace("\n", "")
        else:
            # 如果没有找到匹配的内容，则移除标签后返回
            extracted_content = response.replace("[CON_START]", "").replace("[CON_END]", "").strip()
            if self.mode == 1:
                extracted_content=extracted_content.replace("\n", "")

        if self.mode == 1:
            if len(response.strip())<2500:
                prompt = f"""
你是一位文献综述写作专家。现在你的学生写了一段关于"{subtopic}"的既有研究内容的段落，即<现有段落>。
但是你对它不太满意，因为它的长度小于了2500个汉字。
请你结合<参考资料>扩充它到超过2500个汉字，尽可能多地扩充。
要求严格遵循<扩充原则>。

<现有段落>
{response.strip()}
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
保持文献在正文中的引用方式为句首的“姓名（年份）”，不要在句子后面重复出现（姓名，年份）。
</扩充原则>

<参考资料>
{combined_content}
</参考资料>

扩写结果："""
                try:
                    response = handler.call_llm(provider="zhipuai",
                                                prompt=prompt,
                                                model="glm-4.5-air",
                                                max_tokens=8000)
                    # 检查响应是否包含API调用失败信息
                    if "API call failed" in str(response):
                        raise Exception("API call failed detected in response")

                except:
                    try:
                        response = handler.call_llm(provider="siliconflow",
                                                    prompt=prompt,
                                                    model="THUDM/GLM-4-32B-0414",
                                                    max_tokens=8000)
                        # 检查响应是否包含API调用失败信息
                        if "API call failed" in str(response):
                            raise Exception("API call failed detected in response")
                    except:
                        response = handler.call_llm(provider="openai",
                                                    prompt=prompt,
                                                    model="gpt-4o-mini",
                                                    max_tokens=8000)

                pattern = r'\[CON_START\](.*?)\[CON_END\]'
                match = re.search(pattern, response, re.DOTALL)

                if match:
                    # 如果找到匹配的内容，返回匹配部分
                    extracted_content = match.group(1).strip()
                else:
                    # 如果没有找到匹配的内容，则移除标签后返回
                    extracted_content = response.replace("[CON_START]", "").replace("[CON_END]", "").strip()

                return extracted_content

        return extracted_content

    def generate_table(self, paragraph: str, content_type: str, subtopic: str) -> str:
        """
        基于生成的段落为特定类型的内容生成表格
        """
        # 修改后的表格生成函数，使用已生成的段落作为输入

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
        try:
            response = handler.call_llm(provider="openai",
                                        prompt=prompt,
                                        model="gpt-4o-mini",
                                        max_tokens=8000)
            # 检查响应是否包含API调用失败信息
            if "API call failed" in str(response):
                raise Exception("API call failed detected in response")
        except:
            response = handler.call_llm(provider="siliconflow",
                                        prompt=prompt,
                                        model="THUDM/GLM-4-32B-0414",
                                        max_tokens=8000)
        # response = self.ask_claude(prompt)
        # response = handler.call_llm(provider="claude",
        #                             prompt=prompt,
        #                             model="claude-3-5-sonnet-latest",
        #                             max_tokens=2048)
        # 使用正则表达式提取表格内容

        # 直接提取所有以|开头和结尾的行
        lines = response.strip().split('\n')
        table_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                table_lines.append(line)

        if table_lines:
            return '\n'.join(table_lines)
        else:
            print("警告: 无法从大模型的响应中提取表格格式。返回原始响应。")
            return response.strip()

    def refine_paragraph(self, paragraph_text: str, previous_text: str = "", following_text: str = "", topic: str = ""):
        """
        润色段落文本，提升与前后段落的连贯性

        Args:
            paragraph_text (str): 待润色的段落文本
            previous_text (str): 前一段落文本，若为第一段则为空
            following_text (str): 后一段落文本，若为最后一段则为空
            topic (str): 主题，用于提示词中

        Returns:
            str: 润色后的段落文本
        """
        # 准备前一段落文本
        if not previous_text:
            previous_text = "待润色段落为第一段，无上一段"

        # 准备后一段落文本
        if not following_text:
            following_text = "待润色段落为最后一段，无下一段"

        # 使用模板构建提示词
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

        # 调用大模型 API
        try:

            response = handler.call_llm(provider="siliconflow",
                                        prompt=prompt,
                                        model="THUDM/GLM-4-32B-0414",
                                        max_tokens=8000)
            # 检查响应是否包含API调用失败信息
            if "API call failed" in str(response):
                raise Exception("API call failed detected in response")
        except:
            response = handler.call_llm(provider="openai",
                                        prompt=prompt,
                                        model="gpt-4o-mini",
                                        max_tokens=8000)
        # response = self.ask_claude(prompt)
        # response = handler.call_llm(provider="claude",
        #                             prompt=prompt,
        #                             model="claude-3-5-sonnet-latest",
        #                             max_tokens=2048)
        return response.strip()

    def refine_paragraphs(self, paragraphs_list: list, topic: str = "") -> list:
        """
        使用多线程并行润色段落列表

        Args:
            paragraphs_list (list): 待润色的段落文本列表
            topic (str): 主题

        Returns:
            list: 润色后的段落文本列表
        """
        refined_paragraphs = [None] * len(paragraphs_list)

        # 创建任务处理函数
        def process_refine_task(task_data):
            index, paragraph = task_data
            prev_paragraph = paragraphs_list[index - 1] if index > 0 else ""
            next_paragraph = paragraphs_list[index + 1] if index < len(paragraphs_list) - 1 else ""

            try:
                result = self.refine_paragraph(paragraph, prev_paragraph, next_paragraph, topic)
                return (index, result)
            except Exception as e:
                print(f"润色任务 (段落 {index + 1}/{len(paragraphs_list)}) 失败: {str(e)}")
                return (index, paragraph)  # 如果失败，返回原段落

        # 准备所有润色任务
        refine_tasks = [(i, paragraph) for i, paragraph in enumerate(paragraphs_list) if paragraph]

        # 如果没有任务，直接返回原列表
        if not refine_tasks:
            return paragraphs_list

        # 使用线程池并行执行所有润色任务
        print(f"启动 {len(refine_tasks)} 个润色任务...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_refine_task, refine_tasks))

        # 处理结果
        for index, refined_text in results:
            refined_paragraphs[index] = refined_text

        return refined_paragraphs

    def _is_valid_citation(self, citation, content):
        """代理到CitationChecker类的方法"""
        return citation_checker.is_citation_in_content(citation, content)

    def _generate_specific_section(self, full_text: str, main_topic: str, section_type: str) -> str:
        """
        生成特定类型的章节（总论或述评）

        Args:
            full_text (str): 完整的文献综述文本
            main_topic (str): 整体研究选题
            section_type (str): 章节类型，'overview'表示总论，'critique'表示述评

        Returns:
            str: 生成的章节内容
        """
        if section_type == "overview":
            title = "总论"
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
        else:  # section_type == "critique"
            title = "述评"
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

        # 调用大模型 API获取响应
        try:
            response = handler.call_llm(provider="zhipuai",
                                      prompt=prompt,
                                      model="glm-4.5-air",
                                      max_tokens=8000)
            # 检查响应是否包含API调用失败信息
            if "API call failed" in str(response):
                raise Exception("API call failed detected in response")
        except:
            try:
                response = handler.call_llm(provider="ark",
                                            prompt=prompt,
                                            model="doubao-1-5-lite-32k-250115",
                                            max_tokens=8000)
                # 检查响应是否包含API调用失败信息
                if "API call failed" in str(response):
                    raise Exception("API call failed detected in response")
            except:
                try:
                    response = handler.call_llm(provider="siliconflow",
                                              prompt=prompt,
                                              model="THUDM/GLM-4-32B-0414",
                                              max_tokens=8000)
                    # 检查响应是否包含API调用失败信息
                    if "API call failed" in str(response):
                        raise Exception("API call failed detected in response")
                except:
                    response = handler.call_llm(provider="openai",
                                                prompt=prompt,
                                                model="gpt-4o",
                                                max_tokens=8000)


        # response = handler.call_llm(provider="claude",
        #                             prompt=prompt,
        #                             model="claude-3-5-sonnet-latest",
        #                             max_tokens=2048)
        # response = self.ask_claude(prompt)

        # 处理响应，添加标题
        content = response.strip()

        # # 如果内容没有以标题开头，添加标题
        # if not content.startswith("# "):
        #     content = f"# {title}\n\n{content}"

        return content

    def split_literature_by_language(self,literature_data):
        """
        将文献数据按语言分成中文和英文两部分

        Args:
            literature_data (dict): 原始文献数据

        Returns:
            tuple: (中文文献数据, 英文文献数据)，保持原有数据结构
        """
        # 初始化中文和英文文献数据结构
        zh_literature = {}
        en_literature = {}

        # 遍历每个主题
        for topic, content_types in literature_data.items():
            # 为每个主题在中英文数据中创建对应结构
            zh_literature[topic] = {}
            en_literature[topic] = {}

            # 遍历每种内容类型
            for content_type, items in content_types.items():
                # 分别存储中文和英文文献
                zh_items = [item for item in items if item["language"] == "中文"]
                en_items = [item for item in items if item["language"] == "英文"]

                # 添加到相应的数据结构中
                if zh_items:
                    zh_literature[topic][content_type] = zh_items
                if en_items:
                    en_literature[topic][content_type] = en_items

        return zh_literature, en_literature

    def generate_paragraphs_only(self, review_data: dict) -> dict:
        """
        只生成段落内容
        """
        content_results = {}
        # 为每个主题初始化结果字典
        for topic in review_data.keys():
            content_results[topic] = {}

        def process_paragraph_task(task_data):
            topic, content_type, data_list = task_data
            try:
                result = self.generate_paragraph(data_list, content_type, topic)
                return (topic, content_type, result)
            except Exception as e:
                print(f"段落任务 ({topic}, {content_type}) 失败: {str(e)}")
                return (topic, content_type, f"生成失败: {str(e)}")

        # 准备段落任务
        paragraph_tasks = []
        for topic, content_types in review_data.items():
            for content_type, data_list in content_types.items():
                if not data_list:  # 如果列表为空，跳过
                    continue
                # 初始化该主题的该内容类型的结果容器
                content_results[topic][content_type] = {}
                # 添加段落任务
                paragraph_tasks.append((topic, content_type, data_list))

        # 执行所有段落任务
        print(f"启动 {len(paragraph_tasks)} 个段落生成任务...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            paragraph_results = list(executor.map(process_paragraph_task, paragraph_tasks))

        # 处理段落结果
        paragraphs_by_topic = {}  # 用于收集每个主题的段落，供后续润色使用
        for topic, content_type, result in paragraph_results:
            content_results[topic][content_type]["paragraph"] = result

            # 收集成功生成的段落，按主题分组
            if not result.startswith("生成失败:") and not result.startswith("API调用失败:"):
                if topic not in paragraphs_by_topic:
                    paragraphs_by_topic[topic] = []
                paragraphs_by_topic[topic].append(result)

        # 润色各主题的段落
        for topic, paragraphs in paragraphs_by_topic.items():
            if len(paragraphs) > 1:  # 只有当有多个段落时才进行润色
                print(f"开始润色主题 '{topic}' 的 {len(paragraphs)} 个段落...")
                refined_paragraphs = self.refine_paragraphs(paragraphs, topic)

                # 更新段落结果
                paragraph_index = 0
                for content_type, results in content_results[topic].items():
                    if "paragraph" in results and not results["paragraph"].startswith("生成失败:") and not results[
                        "paragraph"].startswith("API调用失败:"):
                        results["paragraph"] = refined_paragraphs[paragraph_index]
                        paragraph_index += 1

        return content_results

    def generate_tables_only(self, paragraph_results: dict) -> dict:
        """
        只生成表格内容，基于已有的段落结果
        """

        def process_table_task(task_data):
            topic, content_type, paragraph, subtopic = task_data
            try:
                result = self.generate_table(paragraph, content_type, subtopic)
                return (topic, content_type, result)
            except Exception as e:
                print(f"表格任务 ({topic}, {content_type}) 失败: {str(e)}")
                return (topic, content_type, f"生成失败: {str(e)}")

        # 准备表格任务
        table_tasks = []
        for topic, content_types in paragraph_results.items():
            for content_type, results in content_types.items():
                if "paragraph" in results:
                    paragraph_content = results["paragraph"]
                    if paragraph_content and not paragraph_content.startswith(
                            "生成失败:") and not paragraph_content.startswith("API调用失败:"):
                        table_tasks.append((topic, content_type, paragraph_content, topic))

        # 执行所有表格任务
        print(f"启动 {len(table_tasks)} 个表格生成任务...")
        table_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(process_table_task, table_tasks))

        # 处理表格结果
        for topic, content_type, result in results:
            if topic not in table_results:
                table_results[topic] = {}
            if content_type not in table_results[topic]:
                table_results[topic][content_type] = {}
            table_results[topic][content_type]["table"] = result

        return table_results

    def generate_overview_and_critique(self, full_text: str, main_topic: str) -> tuple:
        """
        生成总论和述评
        """

        def process_overview_task():
            try:
                result = self._generate_specific_section(full_text, main_topic, "overview")
                return ("overview", result)
            except Exception as e:
                print(f"总论生成任务失败: {str(e)}")
                return ("overview", f"生成失败: {str(e)}")

        def process_critique_task():
            try:
                result = self._generate_specific_section(full_text, main_topic, "critique")
                return ("critique", result)
            except Exception as e:
                print(f"述评生成任务失败: {str(e)}")
                return ("critique", f"生成失败: {str(e)}")

        # 并行生成总论和述评
        print("启动总论和述评生成任务...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            overview_future = executor.submit(process_overview_task)
            critique_future = executor.submit(process_critique_task)

            overview_result = overview_future.result()
            critique_result = critique_future.result()

        return overview_result[1], critique_result[1]

    def build_full_text_from_results(self, review_data: dict, paragraph_results: dict, table_results: dict) -> tuple:
        """
        基于段落和表格结果构建完整文本和参考文献
        """
        all_references = []  # 收集所有主题的参考文献
        chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

        # 处理每个主题的内容
        topics = {}
        topic_contents = []

        for i, (topic, content_types) in enumerate(review_data.items()):
            # 添加序号到标题
            topic_number = chinese_numbers[i] if i < len(chinese_numbers) else str(i + 1)
            topic_title = f"## （{topic_number}）{topic}"
            topic_sections = [topic_title]

            # 用于记录已被引用的文献
            cited_references = set()

            # 处理每种内容类型
            for content_type, data_list in content_types.items():
                if not data_list:
                    continue

                # 添加段落
                if (topic in paragraph_results and
                        content_type in paragraph_results[topic] and
                        "paragraph" in paragraph_results[topic][content_type]):

                    paragraph_content = paragraph_results[topic][content_type]["paragraph"]
                    if not paragraph_content.startswith("生成失败:") and not paragraph_content.startswith(
                            "API调用失败:"):
                        topic_sections.append(paragraph_content)

                        # 收集段落中使用的引用
                        if data_list:
                            apa_formatted = self.transform_list_to_apa_format(data_list)
                            for item in apa_formatted:
                                citation = item.get("apa_citation", "")
                                content = paragraph_content.lower()
                                if citation and self._is_valid_citation(citation, content):
                                    cited_references.add(item["apa_reference"])

                # 添加表格
                if (topic in table_results and
                        content_type in table_results[topic] and
                        "table" in table_results[topic][content_type]):

                    table_content = table_results[topic][content_type]["table"]
                    if not table_content.startswith("生成失败:") and not table_content.startswith("API调用失败:"):
                        topic_sections.append(table_content)

            # 收集引用的参考文献
            all_references.extend(cited_references)

            # 组合该主题的所有内容
            topic_text = "\n\n".join(topic_sections)
            topics[topic] = topic_text
            topic_contents.append(topic_text)

        # 处理参考文献
        unique_references = list(set(all_references))
        unique_references.sort()

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

        full_text = "\n\n".join(topic_contents)

        return topics, full_text, clean_references, pdf_urls

    def run_review(self, review_data: dict, main_topic: str) -> dict:
        """
        运行文献综述生成的入口点 - 单语言版本
        """
        # 步骤1: 生成段落
        paragraph_results = self.generate_paragraphs_only(review_data)

        # 步骤2: 构建用于生成overview和critique的文本（仅使用段落内容）
        topics_temp, full_text_for_overview, _, _ = self.build_full_text_from_results(
            review_data, paragraph_results, {}  # 传入空的table_results
        )

        # 步骤3: 并行生成表格和总论述评
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # 提交表格生成任务
            table_future = executor.submit(self.generate_tables_only, paragraph_results)
            # 提交总论述评生成任务
            overview_critique_future = executor.submit(self.generate_overview_and_critique, full_text_for_overview,
                                                       main_topic)

            # 获取结果
            table_results = table_future.result()
            overview, critique = overview_critique_future.result()

        # 步骤4: 构建完整文本（包含表格）
        topics, full_text, unique_references, pdf_urls = self.build_full_text_from_results(
            review_data, paragraph_results, table_results
        )

        # 步骤5: 组装最终结果
        main_text = "\n\n".join([
            overview,
            full_text,
            "## 研究评述" + "\n\n" + critique
        ])

        complete_text = "\n\n".join([
            overview,
            full_text,
            "## 研究评述\n\n" + critique,
            "## 参考文献\n\n" + "\n\n".join(unique_references)
        ])
        complete_text = complete_text.strip()

        return topics, overview, main_text, critique, unique_references, complete_text, pdf_urls

    def run_review_v2(self, review_data: dict, main_topic: str) -> dict:
        """
        运行文献综述生成的入口点 - 中英文分离版本
        """
        # 步骤1: 按语言分离文献
        zh_literature, en_literature = self.split_literature_by_language(review_data)

        # 步骤2: 并行生成中英文段落
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            zh_paragraph_future = executor.submit(self.generate_paragraphs_only, zh_literature)
            en_paragraph_future = executor.submit(self.generate_paragraphs_only, en_literature)

            paragraph_results_zn = zh_paragraph_future.result()
            paragraph_results_en = en_paragraph_future.result()

        # 步骤3: 构建用于生成overview和critique的中英文文本（仅使用段落内容）
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            zh_build_temp_future = executor.submit(
                self.build_full_text_from_results, zh_literature, paragraph_results_zn, {}
            )
            en_build_temp_future = executor.submit(
                self.build_full_text_from_results, en_literature, paragraph_results_en, {}
            )

            topics_zn_temp, full_text_zn_temp, _, _ = zh_build_temp_future.result()
            topics_en_temp, full_text_en_temp, _, _ = en_build_temp_future.result()

        # 构建合并的文本用于生成overview和critique
        combined_full_text_temp = "# 一、国内相关研究\n\n" + full_text_zn_temp + "\n\n" + "# 二、国外相关研究\n\n" + full_text_en_temp

        # 步骤4: 并行生成中英文表格和总论述评
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            zh_table_future = executor.submit(self.generate_tables_only, paragraph_results_zn)
            en_table_future = executor.submit(self.generate_tables_only, paragraph_results_en)
            overview_critique_future = executor.submit(self.generate_overview_and_critique, combined_full_text_temp,
                                                       main_topic)

            table_results_zn = zh_table_future.result()
            table_results_en = en_table_future.result()
            overview, critique = overview_critique_future.result()

        # 步骤5: 构建最终的中英文完整文本（包含表格）
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            zh_build_future = executor.submit(
                self.build_full_text_from_results, zh_literature, paragraph_results_zn, table_results_zn
            )
            en_build_future = executor.submit(
                self.build_full_text_from_results, en_literature, paragraph_results_en, table_results_en
            )

            topics_zn, full_text_zn, unique_references_zn, pdf_urls_zn = zh_build_future.result()
            topics_en, full_text_en, unique_references_en, pdf_urls_en = en_build_future.result()

        # 步骤6: 合并最终结果
        topics = {
            "topics_zn": topics_zn,
            "topics_en": topics_en
        }

        combined_full_text = "# 一、国内相关研究\n\n" + full_text_zn + "\n\n" + "# 二、国外相关研究\n\n" + full_text_en
        unique_references = unique_references_zn + unique_references_en
        pdf_urls = pdf_urls_zn + pdf_urls_en

        # 步骤7: 组装最终结果
        main_text = "\n\n".join([
            overview,
            combined_full_text,
            "# 研究评述" + "\n\n" + critique
        ])

        complete_text = "\n\n".join([
            overview,
            combined_full_text,
            "# 研究评述\n\n" + critique,
            "# 参考文献\n\n" + "\n\n".join(unique_references)
        ])
        complete_text = complete_text.strip()

        return topics, overview, main_text, critique, unique_references, complete_text, pdf_urls

class LiteratureReviewSystem:
    def __init__(self, user_id, task_id, kb_id: str, mode: int = None, structure: int=0):
        self.user_id=user_id
        self.task_id=task_id
        self.topic_agent = TopicUnderstandingAgent()
        self.literature_agent = LiteratureCollectionAgent(kb_id, mode=mode)
        self.writing_agent = WritingAgent(mode=mode)
        self.mode=mode

    def calculate_reference_language_ratio(self,references):
        """
        计算参考文献列表的中英文比例
        判断规则：包含汉字的视为中文文献，不包含汉字的视为英文文献
        返回: { "zh_ratio": 中文比例, "en_ratio": 英文比例 }
        """
        zh_count = 0
        en_count = 0

        # 汉字Unicode范围：\u4e00-\u9fa5
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')

        for ref in references:
            if chinese_pattern.search(ref):
                zh_count += 1
            else:
                en_count += 1

        total = len(references)
        return round(zh_count / total, 4)

    def translate_markdown(self,markdown_text: str, target_language: str = "English") -> str:
        """
        Translates markdown content using a language model while preserving all original formatting.

        Args:
            markdown_text: Original markdown text content
            target_language: Target language code, defaults to English ("en")

        Returns:
            Translated markdown text with preserved formatting
        """
        try:

            # Construct translation prompt in idiomatic English
            prompt = f"""
    <TASK>
    You are a precision markdown translation specialist. Please translate the following markdown content into {target_language}, while meticulously preserving all original formatting and placeholders.
    </TASK>

    <MARKDOWN_TO_TRANSLATE>
    {markdown_text}
    </MARKDOWN_TO_TRANSLATE>

    <TRANSLATION_REQUIREMENTS>
    1. Translate only the plain text content
    2. Preserve all formatting markers (such as #, *, -, >) without modification or translation
    3. Maintain all placeholders (such as CODE_BLOCK_0, INLINE_CODE_0, LINK_0, IMAGE_0) exactly as they appear
    4. Retain all line breaks, indentation, and whitespace in their original form
    5. Ensure headers, lists, tables, and other structural elements remain identical to the source
    6. Please ensure the translation is idiomatic, fluid, professionally precise, and elegantly crafted, with special attention to preserving all table formatting symbols including pipes (|), hyphens (-), and colons (:) in their exact original arrangement
    </TRANSLATION_REQUIREMENTS>

    <OUTPUT_INSTRUCTION>
    Wrap your translation with [TRANS_START] and [TRANS_END] tags. Return only the translated markdown content without any explanations or comments. Ensure the output is ready to be used directly as a valid markdown file.
    </OUTPUT_INSTRUCTION>

    Your translated markdown (in {target_language}):
    """

            # 调用API进行翻译
            try:
                response_content = handler.call_llm(provider="openai",
                                                    prompt=prompt,
                                                    model="gpt-4o-mini",
                                                    max_tokens=8000,
                                                    temperature=0.7)
                # 检查响应是否包含API调用失败信息
                if "API call failed" in str(response_content):
                    raise Exception("API call failed detected in response")
            except:
                response_content = handler.call_llm(provider="ark",
                                                    prompt=prompt,
                                                    model="doubao-1-5-lite-32k-250115",
                                                    max_tokens=8000,
                                                    temperature=0.7)

            # Get translation result

            print(response_content)
            # Extract content between tags using regex
            translation_pattern = r'\[TRANS_START\]([\s\S]*)\[TRANS_END\]'
            match = re.search(translation_pattern, response_content)

            if match:
                translated_markdown = match.group(1)
            else:
                # Fallback if tags aren't found
                translated_markdown = response_content.replace('[TRANS_START]', '').replace('[TRANS_END]', '')

            return translated_markdown

        except Exception as e:
            print(f"Failed to translate markdown: {e},try it without protect")
            # Return the original text if API call fails
            # Construct translation prompt in idiomatic English
            prompt = f"""
            <TASK>
            You are a precision markdown translation specialist. Please translate the following markdown content into {target_language}, while meticulously preserving all original formatting and placeholders.
            </TASK>

            <MARKDOWN_TO_TRANSLATE>
            {markdown_text}
            </MARKDOWN_TO_TRANSLATE>

            <TRANSLATION_REQUIREMENTS>
            1. Translate only the plain text content
            2. Preserve all formatting markers (such as #, *, -, >) without modification or translation
            3. Maintain all placeholders (such as CODE_BLOCK_0, INLINE_CODE_0, LINK_0, IMAGE_0) exactly as they appear
            4. Retain all line breaks, indentation, and whitespace in their original form
            5. Ensure headers, lists, tables, and other structural elements remain identical to the source
            6. Please ensure the translation is idiomatic, fluid, professionally precise, and elegantly crafted, with special attention to preserving all table formatting symbols including pipes (|), hyphens (-), and colons (:) in their exact original arrangement
            </TRANSLATION_REQUIREMENTS>

            <OUTPUT_INSTRUCTION>
            Wrap your translation with [TRANS_START] and [TRANS_END] tags. Return only the translated markdown content without any explanations or comments. Ensure the output is ready to be used directly as a valid markdown file.
            </OUTPUT_INSTRUCTION>

            Your translated markdown:
            """

            # 调用API进行翻译
            try:
                response_content = handler.call_llm(provider="openai",
                                                    prompt=prompt,
                                                    model="gpt-4o-mini",
                                                    max_tokens=8000,
                                                    temperature=0.7)
                # 检查响应是否包含API调用失败信息
                if "API call failed" in str(response_content):
                    raise Exception("API call failed detected in response")
            except:
                response_content = handler.call_llm(provider="ark",
                                                    prompt=prompt,
                                                    model="doubao-1-5-lite-32k-250115",
                                                    max_tokens=8000,
                                                    temperature=0.7)
            # Extract content between tags using regex
            translation_pattern = r'\[TRANS_START\]([\s\S]*)\[TRANS_END\]'
            match = re.search(translation_pattern, response_content)

            if match:
                translated_markdown = match.group(1)
            else:
                # Fallback if tags aren't found
                translated_markdown = response_content.replace('[TRANS_START]', '').replace('[TRANS_END]', '')

            return translated_markdown

    def parallel_translate_fields(
            self,review_text: Dict[str, Any],
            translate_function: Callable[[str, str], str],
            target_language: str = "English"
    ) -> str:
        """
        Parallel processes and translates overview, critique, and all fields in topics
        from the review_text data, then organizes results as a single text maintaining the original
        topics order: overview, topics fields (in original order), critique.

        Args:
            review_text: Dictionary containing overview, critique and topics fields
            translate_function: Function that accepts (text, target_language) and returns translated text
            target_language: Target language for translation, defaults to English

        Returns:
            Single text with translated fields organized in the order: overview, topics fields (in original order), critique
        """
        # Check for required fields
        if not all(field in review_text for field in ["overview", "critique", "topics"]):
            raise ValueError("review_text must contain 'overview', 'critique', and 'topics' fields")

        # Store the original topic keys order
        original_topic_keys = list(review_text["topics"].keys())

        # Prepare fields for translation - handle each topic as a separate field
        fields_to_translate = {
            "overview": review_text["overview"]
        }

        # Add all topic fields individually
        for topic_key, topic_value in review_text["topics"].items():
            fields_to_translate[f"topics.{topic_key}"] = topic_value

        # Add critique (will be placed at the end)
        fields_to_translate["critique"] = review_text["critique"]

        # Translate fields in parallel
        translated_fields = {}

        def translate_field(field_name, field_content):
            try:
                translated_content = translate_function(field_content, target_language)
                return field_name, translated_content
            except Exception as e:
                print(f"Error translating field '{field_name}': {e}")
                return field_name, field_content  # Return original if translation fails

        # Execute translations in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_field = {
                executor.submit(translate_field, field_name, content): field_name
                for field_name, content in fields_to_translate.items()
            }

            for future in concurrent.futures.as_completed(future_to_field):
                field_name, translated_content = future.result()
                translated_fields[field_name] = translated_content

        # Organize results as a single text in the specified order (overview, topics in original order, then critique)
        result_parts = []

        # Add overview
        result_parts.append(translated_fields["overview"])

        # Add all translated topic fields in original order
        for topic_key in original_topic_keys:
            field_name = f"topics.{topic_key}"
            if field_name in translated_fields:
                result_parts.append(translated_fields[field_name])

        # Add critique
        result_parts.append("# Summary\n\n" + translated_fields["critique"])

        # Add reference text if it exists
        result_parts.append("# References\n\n"+"\n\n".join(review_text["references"]) )

        # Join all parts with double newlines to separate sections
        return "\n\n".join(result_parts)

    def check_empty_literature(self, literature_data, review_text, collection):
        """
        检查文献数据是否为空，如果为空则更新数据库并返回True

        Args:
            literature_data (dict): 文献数据结构
            review_text (dict): 文献综述文本
            collection: 数据库集合对象

        Returns:
            bool: 如果文献为空返回True，否则返回False
        """
        # 检查是否所有主题和内容类型都没有文献
        all_empty = True
        for subtopic, data in literature_data.items():
            for content_type, docs in data.items():
                if len(docs) > 0:
                    all_empty = False
                    break
            if not all_empty:
                break

        # 如果没有找到任何文献，更新数据库并返回True
        if all_empty:
            review_text["references"] = []
            review_text["complete_text"] = "在社会科学文献库中未检索到相关文献，生成失败。"
            collection.update_one(
                {"user_id": self.user_id, "task_id": self.task_id},
                {"$set": {"review_text": review_text, "state": 1, "update_time": datetime.now()}}
            )
            print("未找到相关文献，生成终止。")
            return True

        return False

    def generate_review(self, main_topic: str, language=None,chinese_weight=None,structure=0) -> str:
        """
        生成文献综述的主函数
        Args:
            main_topic: 主要研究主题
            language:全文语言
            chinese_weight:中文文献比例
            structure:综述结构
        Returns:
            文献综述文本
        """
        # 直接创建MongoDB连接
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]
        collection = db["review"]

        # 更新任务状态为进行中(2)
        collection.update_one(
            {"user_id": self.user_id, "task_id": self.task_id},
            {"$set": {"state": 2, "update_time": datetime.now(), "query": main_topic,"structure":structure,"mode":self.mode}},
            upsert=True
        )

        print(f"开始为主题 '{main_topic}' 生成文献综述...")
        try:
            review_text={}
            review_text["main_topic"]=main_topic

            # 步骤1：分解主题
            subtopics = self.topic_agent.decompose_topic(main_topic)
            review_text["subtopics"] = subtopics

            # 更新任务状态为进行中(2)
            collection.update_one(
                {"user_id": self.user_id, "task_id": self.task_id},
                {"$set": {"review_text": review_text, "state": 2, "update_time": datetime.now()}}
            )
            print(f"子议题生成成功，已保存到MongoDB")

            if structure == 1:
                # 步骤2：收集文献
                literature_data = self.literature_agent.collect_literature(subtopics=subtopics, language=language,
                                                                           chinese_weight=0.5)
                if self.check_empty_literature(literature_data, review_text, collection):
                    return  # 如果文献为空，提前退出函数
                # 步骤3：生成综述
                topics,overview,main_text,critique,unique_references,complete_text,pdf_urls = self.writing_agent.run_review_v2(
                    literature_data,
                    main_topic)
            else:
                # 步骤2：收集文献
                literature_data = self.literature_agent.collect_literature(subtopics=subtopics, language=language,chinese_weight=chinese_weight)
                # print(literature_data)
                if self.check_empty_literature(literature_data, review_text, collection):
                    return  # 如果文献为空，提前退出函数

                # 步骤3：生成综述
                topics,overview,main_text,critique,unique_references,complete_text,pdf_urls = self.writing_agent.run_review(
                    literature_data,
                main_topic)
            review_text["topics"]=topics
            review_text["overview"] = overview
            review_text["main_text"] = main_text
            review_text["critique"] = critique
            review_text["complete_text"]=complete_text
            review_text["references"]=unique_references
            review_text["pdf_urls"] = pdf_urls

            if language == "英文":
                # 步骤4：若是选择英文模式，则翻译
                print("开始翻译……")
                translated_review_text = self.parallel_translate_fields(review_text, self.translate_markdown)
                review_text["complete_text"] = translated_review_text
                # 保存文献综述到MongoDB并更新任务状态为成功(1)
                collection.update_one(
                    {"user_id": self.user_id, "task_id": self.task_id},
                    {"$set": {"review_text": review_text, "state": 1, "update_time": datetime.now()}}
                )
                print(f"英文文献综述生成成功，已保存到MongoDB")
            else:
                if chinese_weight is None:
                    # 步骤4：若是没有选择英文模式，且未设置中文文献比例，则无需计算最终中文比例

                    # 保存文献综述到MongoDB并更新任务状态为成功(1)
                    collection.update_one(
                        {"user_id": self.user_id, "task_id": self.task_id},
                        {"$set": {"review_text": review_text, "state": 1, "update_time": datetime.now()}}
                    )
                    print(f"文献综述生成成功，已保存到MongoDB")
                else:
                    # 步骤4：若是没有选择英文模式，且设置中文文献比例，则计算最终中文比例
                    # 计算中文比例
                    ratio = self.calculate_reference_language_ratio(review_text["references"])
                    print(f"中文文献比例: {ratio:.2%}")  # 输出示例: 中文文献比例: 60.00%        # 检查是否所有主题的文献长度都为0

                    if ratio<chinese_weight:
                        review_text["chinese_paper_check"]=f"中文文献不足{chinese_weight:.2%}，已自动补充英文文献完成综述。当前中文文献比例为 {ratio:.2%}。"

                    # 保存文献综述到MongoDB并更新任务状态为成功(1)
                    collection.update_one(
                        {"user_id": self.user_id, "task_id": self.task_id},
                        {"$set": {"review_text": review_text, "state": 1, "update_time": datetime.now()}}
                    )
                    print(f"文献综述生成成功，已保存到MongoDB")
                    return review_text
        except Exception as e:
            # 如果在生成过程中出现错误，更新任务状态为失败(0)
            try:
                # 确保有一个有效的MongoDB连接
                mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
                db = mongo_client["Newbit"]
                collection = db["review"]

                collection.update_one(
                    {"user_id": self.user_id, "task_id": self.task_id},
                    {"$set": {"state": 0, "error": str(e), "update_time": datetime.now()}}
                )
            except Exception as mongo_error:
                print(f"更新错误状态失败: {mongo_error}")

            print(f"生成文献综述时出错: {e}")


def translate_to_chinese(text: str) -> str:
    """
    使用大模型将任何文本翻译为中文

    Args:
        text: 需要翻译的文本

    Returns:
        翻译后的中文文本
    """
    try:
        # 初始化OpenAI客户端

        # 构建提示词
        prompt = f"""
请将以下文本翻译成中文，保持专业、流畅、自然的表达，若本来就是中文表达则输出原文：

[TEXT_TO_TRANSLATE]
{text}
[/TEXT_TO_TRANSLATE]

将翻译结果用[TRANS_START]和[TRANS_END]标签包裹起来，不要添加任何解释或注释。

中文文本：
"""

        # 调用API
        try:
            response_text = handler.call_llm(provider="openai",
                                        prompt=prompt,
                                        model="gpt-4o-mini",
                                        max_tokens=8000,
                                        temperature=0.3)
            # 检查响应是否包含API调用失败信息
            if "API call failed" in str(response_text):
                raise Exception("API call failed detected in response")
        except:
            response_text = handler.call_llm(provider="ark",
                                                prompt=prompt,
                                                model="doubao-1-5-lite-32k-250115",
                                                max_tokens=8000,
                                                temperature=0.3)

        # 使用正则表达式提取标签之间的内容
        pattern = r'\[TRANS_START\]([\s\S]*)\[TRANS_END\]'
        match = re.search(pattern, response_text)

        if match:
            translated_text = match.group(1).strip()
        else:
            # 如果正则提取失败，返回完整响应但剔除可能存在的标记
            translated_text = response_text.replace('[TRANS_START]', '').replace('[TRANS_END]', '')

        return translated_text

    except Exception as e:
        print(f"翻译失败: {e}")
        return text  # 如果翻译失败，返回原文


