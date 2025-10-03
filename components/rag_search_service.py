import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from es_search import *
from chat_with_data_service import *

class MongoDBConnector:
    """处理与MongoDB的连接和查询"""

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self) -> bool:
        """建立与MongoDB的连接"""
        try:
            self.db = mongo_client[MONGO_DB]
            self.collection = self.db[MONGO_COLLECTION]

            # 测试连接
            self.client.server_info()
            print(f"成功连接到MongoDB: {MONGO_DB}.{MONGO_COLLECTION}")
            return True
        except Exception as e:
            print(f"连接MongoDB时出错: {e}")
            return False

    def get_data_for_indicator(self, indicator_name: str) -> pd.DataFrame:
        """
        从MongoDB检索给定指标的数据（单个指标版本）

        参数:
            indicator_name: 要检索的指标名称

        返回:
            包含指标数据的DataFrame
        """
        try:
            # 查询MongoDB中的指标
            query = {indicator_name: {"$exists": True}}
            projection = {"_id": 0, indicator_name: 1, "id": 1}

            cursor = self.collection.find(query, projection)
            data = list(cursor)

            if not data:
                print(f"未找到指标的数据: {indicator_name}")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            print(f"检索到指标 {indicator_name} 的 {len(df)} 条记录")
            return df

        except Exception as e:
            print(f"检索指标 {indicator_name} 的数据时出错: {e}")
            return pd.DataFrame()

    def get_data_for_all_indicators(self, indicator_names: list) -> pd.DataFrame:
        """
        从MongoDB一次性检索多个指标的数据，仅获取year字段大于等于2010的文档

        参数:
            indicator_names: 要检索的指标名称列表

        返回:
            包含所有指标数据的合并DataFrame，id、city和year字段排在最前面
        """
        print(f"正在从MongoDB一次性检索 {len(indicator_names)} 个指标的2010年及之后的数据...")

        if not indicator_names:
            print("没有提供指标名称")
            return pd.DataFrame()

        try:
            # 构建查询条件 - 查找至少包含一个指标的文档，且year字段大于等于2010
            indicators_conditions = [
                {name: {"$exists": True}} for name in indicator_names
            ]

            # 组合指标条件和年份条件
            query = {
                "$and": [
                    {"$or": indicators_conditions},
                    {"year": {"$gte": 2010}}  # year字段大于等于2010
                ]
            }

            # 构建投影，只选择需要的字段
            projection = {"_id": 0, "id": 1, "year": 1, "city": 1}  # 添加year和city字段
            for name in indicator_names:
                projection[name] = 1

            # 执行查询
            cursor = self.collection.find(query, projection)
            data = list(cursor)

            if not data:
                print("未找到2010年及之后的任何指标数据")
                return pd.DataFrame()

            # 将结果转换为DataFrame
            df = pd.DataFrame(data)

            # 确保id、city和year排在最前面
            # 先获取所有列名
            all_columns = list(df.columns)
            # 移除id、city和year（如果存在）
            priority_columns = ['id', 'city', 'year']
            for col in priority_columns:
                if col in all_columns:
                    all_columns.remove(col)

            # 重新组织列顺序：先是priority_columns中存在的列，然后是其余的列
            new_column_order = [col for col in priority_columns if col in df.columns] + all_columns
            df = df[new_column_order]

            print(f"一次性检索到 {len(df)} 条2010年及之后的记录，包含多个指标")

            # 输出各年份的记录数
            if 'year' in df.columns:
                year_counts = df['year'].value_counts().sort_index()
                print("\n各年份记录数:")
                for year, count in year_counts.items():
                    print(f"  - {year}: {count} 条记录")

            # 输出有多少条记录包含每个指标
            print("\n各指标的记录数:")
            for name in indicator_names:
                if name in df.columns:
                    count = df[name].notna().sum()
                    print(f"  - {name}: {count} 条记录")
                else:
                    print(f"  - {name}: 0 条记录")

            return df

        except Exception as e:
            print(f"一次性检索多个指标数据时出错: {e}")
            return pd.DataFrame()


class VariableExtractionAgent:
    """
    主题理解和变量提取代理
    """

    def __init__(self):
        self.name = "主题分析代理"

    def ask_gpt(self, prompt: str) -> str:
        """
        调用GPT-4o-mini API获取回答
        参数:
            prompt: 问题内容
        返回:
            GPT的回答
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"GPT API调用失败: {e}")
            return f"API调用失败: {str(e)}"

    def extract_core_variables(self, main_topic: str) -> List[str]:
        """
        从研究主题中提取核心变量
        参数:
            main_topic: 主要研究主题
        返回:
            核心变量列表
        """
        print(f"{self.name}: 正在从主题'{main_topic}'中提取核心变量...")

        # 使用GPT API提取核心变量
        prompt = f"""
<task>
请分析研究主题并提取2-3个构成该主题的核心变量。研究主题通常由这些关键变量之间的关系组成。
</task>

<process>
<input format>
研究主题: [研究主题文本]
</input format>

<thinking steps>
(仅内部思考，不要在最终答案中输出此部分)

步骤1: 识别主题中包含的核心变量。
我识别到的核心变量是: [列出核心变量]

步骤2: 确定这些变量在研究中的角色。
[变量1]在此研究中可能作为[自变量/因变量/中介变量/调节变量]
[变量2]在此研究中可能作为[自变量/因变量/中介变量/调节变量]
[...]

步骤3: 将这些变量表达为简洁的术语。
[变量1]可以表达为: [术语1]
[变量2]可以表达为: [术语2]
[...]
</thinking steps>

<output format>
格式要求: 输出时不要添加任何数字、标点或前缀。每个核心变量应占一行。不要输出思考过程，只输出最终的核心变量。输出必须与输入研究主题使用相同的语言。
[核心变量1]
[核心变量2]
[核心变量3(如有必要)]
</output format>
</process>

<example>
研究主题: 企业数字化转型对技术创新的影响研究

步骤1: 识别主题中包含的核心变量。
我识别到的核心变量是: 企业数字化转型、技术创新

步骤2: 确定这些变量在研究中的角色。
企业数字化转型在此研究中可能作为自变量
技术创新在此研究中可能作为因变量

步骤3: 将这些变量表达为简洁的术语。
企业数字化转型可以表达为: 企业数字化转型
技术创新可以表达为: 技术创新

输出:
企业数字化转型
技术创新
</example>

研究主题: {main_topic}

输出:
"""

        response = self.ask_gpt(prompt)
        # 处理返回的文本，分割成列表
        core_variables = [variable.strip() for variable in response.strip().split('\n') if variable.strip()]

        print(f"{self.name}: 已从主题中提取 {len(core_variables)} 个核心变量")
        return core_variables

    def identify_control_variables(self, core_variables: List[str]) -> List[str]:
        """
        确定适合影响所有核心变量的控制变量
        参数:
            core_variables: 核心变量列表
        返回:
            控制变量列表
        """
        print(f"{self.name}: 正在为 {len(core_variables)} 个核心变量确定控制变量...")

        # 将核心变量连接起来用于提示
        variables_text = "\n".join(core_variables)

        prompt = f"""
<task>
根据提供的研究核心变量，请确定5-8个可能同时影响这些变量并应在研究中控制的变量。
</task>

<process>
<input>
研究核心变量:
{variables_text}
</input>

<thinking>
(仅内部思考，不要在最终答案中输出此部分)

步骤1: 考虑哪些因素可能影响所有这些核心变量。
可能的影响因素包括:
- [因素1]: 这影响所有核心变量，因为[原因]
- [因素2]: 这影响所有核心变量，因为[原因]
[...]

步骤2: 从这些因素中选择5-8个最重要的控制变量。
最重要的控制变量是:
1. [控制变量1]
2. [控制变量2]
[...]
</thinking>

<output format>
格式要求: 仅输出控制变量，每行一个，没有数字、前缀或额外解释。每个控制变量应是特定的测量或指标。输出必须与输入研究核心变量使用相同的语言。
[控制变量1]
[控制变量2]
[控制变量3]
[控制变量4]
[控制变量5]
[如需要，附加变量]
</output format>
</process>

<example>
研究核心变量:
企业数字化转型
技术创新

思考:
步骤1: 考虑哪些因素可能影响这两个变量。
可能的影响因素包括:
- 企业规模: 大型企业可能拥有更多资源进行数字化转型和创新
- 行业类型: 某些行业更有利于数字化转型和创新
- 研发投入: 直接影响创新并促进数字化转型
- 市场竞争: 推动数字化采用和创新
- 人力资本: 知识型员工对这两个过程都至关重要
- 政府政策: 法规和激励措施影响这两个领域
- 组织结构: 灵活的结构可能有助于这两个过程
- 技术基础设施: 基础技术使两个过程成为可能

步骤2: 选择最重要的控制变量。
最重要的控制变量是:
1. 企业规模
2. 研发投入强度
3. 行业分类
4. 市场竞争强度
5. 人力资本质量
6. 政府政策支持

输出:
企业规模
研发投入强度
行业分类
市场竞争强度
人力资本质量
政府政策支持
</example>

研究核心变量:
{variables_text}

输出:
"""

        response = self.ask_gpt(prompt)
        control_variables = [var.strip() for var in response.strip().split('\n') if var.strip()]

        print(f"{self.name}: 已确定 {len(control_variables)} 个控制变量")
        return control_variables

    def suggest_measurement_indicators(self, variable: str) -> List[str]:
        """
        为每个变量推荐3个测量指标

        参数:
            variable: 需要测量的变量

        返回:
            推荐的测量指标列表
        """
        print(f"{self.name}: 正在为变量 '{variable}' 推荐测量指标...")

        prompt = f"""
<task>
请为给定的研究变量推荐3个具体的测量指标，这些指标应在实际数据中可测量且与变量直接相关。
</task>

<process>
<input>
研究变量: {variable}
</input>

<thinking>
(仅内部思考，不要在最终答案中输出此部分)

步骤1: 考虑如何在实际研究中测量这个变量。
该变量可以通过以下几种方式测量:
- [测量方式1]
- [测量方式2]
- [测量方式3]
[...]

步骤2: 将这些测量方式转化为具体的数据指标。
具体指标可以是:
1. [指标1]: 这能够衡量[变量的哪个方面]
2. [指标2]: 这能够衡量[变量的哪个方面]
3. [指标3]: 这能够衡量[变量的哪个方面]
[...]

步骤3: 确保这些指标在数据库中很可能存在。
[指标1]在统计数据中常见度: [高/中/低]
[指标2]在统计数据中常见度: [高/中/低]
[指标3]在统计数据中常见度: [高/中/低]
[...]

步骤4: 选择最合适的3个指标。
最佳指标是:
1. [最终指标1]
2. [最终指标2]
3. [最终指标3]
</thinking>

<output format>
格式要求: 只输出3个测量指标，每行一个，不要有任何数字编号、解释或前缀。每个指标应是简洁明确的名称，通常用于数据库或统计数据中。输出必须与输入变量使用相同的语言。
[测量指标1]
[测量指标2]
[测量指标3]
</output format>
</process>

<example>
研究变量: 企业数字化转型

思考:
步骤1: 考虑如何在实际研究中测量企业数字化转型。
企业数字化转型可以通过以下几种方式测量:
- 数字技术投入
- 业务流程数字化程度
- 数字化人才比例
- 线上业务占比
- 数字化管理系统应用
- 大数据应用水平

步骤2: 将这些测量方式转化为具体的数据指标。
具体指标可以是:
1. 信息化投入占总资产比例: 衡量数字技术投入强度
2. 数字化业务收入占比: 衡量业务数字化转型成效
3. IT人员比例: 衡量数字化人才水平
4. 核心业务流程数字化率: 衡量流程数字化程度
5. 数据分析应用水平: 衡量大数据应用情况
6. 云服务采用率: 衡量企业云技术应用情况

步骤3: 确保这些指标在数据库中很可能存在。
信息化投入占总资产比例在统计数据中常见度: 高
数字化业务收入占比在统计数据中常见度: 高
IT人员比例在统计数据中常见度: 中
核心业务流程数字化率在统计数据中常见度: 中
数据分析应用水平在统计数据中常见度: 低
云服务采用率在统计数据中常见度: 中

步骤4: 选择最合适的3个指标。
最佳指标是:
1. 信息化投入占比
2. 数字化业务收入占比
3. IT人员比例

输出:
信息化投入占比
数字化业务收入占比
IT人员比例
</example>

研究变量: {variable}

输出:
"""

        response = self.ask_gpt(prompt)
        indicators = [ind.strip() for ind in response.strip().split('\n') if ind.strip()]

        # 确保只返回3个指标
        indicators = indicators[:3]

        print(f"{self.name}: 已为变量 '{variable}' 推荐 {len(indicators)} 个测量指标: {indicators}")
        return indicators


class ResearchDataRetriever:
    """协调研究数据检索过程的主类"""

    def __init__(self):
        self.mongo_connector = MongoDBConnector()
        self.topic_agent = VariableExtractionAgent()

    def search_indicators(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索与查询相关的指标

        参数:
            query: 搜索查询
            top_k: 返回的结果数量

        返回:
            匹配的指标列表
        """
        # 为数据指标知识库定义筛选条件
        filters = {
            "kb_id": DATA_INDICATORS_KB_ID
        }

        # 执行使用jieba分词的混合搜索
        results = es_service.hybrid_search_indicator(
            query_text=query,
            top_k=top_k,
            vector_weight=DEFAULT_VECTOR_WEIGHT,
            text_weight=DEFAULT_TEXT_WEIGHT,
            filters=filters
        )

        return results

    def find_best_indicator_match(self, variable: str, measurement_indicators: List[str]) -> Dict[str, Any]:
        """
        查找最匹配的指标

        参数:
            variable: 变量名称
            measurement_indicators: 建议的测量指标列表

        返回:
            最匹配的指标信息
        """
        print(f"为变量 '{variable}' 查找最匹配的指标...")

        all_indicators = []
        top_similarities = {}

        # 直接搜索每个建议的测量指标
        for indicator in measurement_indicators:
            results = self.search_indicators(indicator, top_k=1)
            if results:
                result = results[0]
                all_indicators.append(result)

                # 计算相似度，可以基于score或自定义规则
                # 这里简单地使用搜索返回的分数归一化处理
                similarity = result["score"]
                top_similarities[result["id"]] = similarity

        # 如果没有找到任何指标
        if not all_indicators:
            print(f"未找到变量 '{variable}' 的任何匹配指标")
            return {}

        # 找出相似度最高的指标
        best_id = max(top_similarities.items(), key=lambda x: x[1])[0]
        best_indicator = next((ind for ind in all_indicators if ind["id"] == best_id), all_indicators[0])

        print(f"找到最匹配的指标: {best_indicator.get('docnm_kwd', '')}")
        return best_indicator

    def process_research_topic(self, main_topic: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        通过提取核心变量、确定控制变量、查找相关指标并从MongoDB检索数据来处理研究主题

        参数:
            main_topic: 主要研究主题

        返回:
            Tuple[pd.DataFrame, List[Dict[str, Any]]]:
                - 包含研究所有相关数据的DataFrame
                - 所有指标的元数据信息列表
        """
        print(f"开始分析研究主题: '{main_topic}'")

        # 步骤1: 从主题中提取核心变量
        core_variables = self.topic_agent.extract_core_variables(main_topic)
        print(f"已提取核心变量: {core_variables}")

        # 步骤2: 确定控制变量
        control_variables = self.topic_agent.identify_control_variables(core_variables)
        print(f"已确定控制变量: {control_variables}")

        # 步骤3: 合并核心变量和控制变量
        all_variables = core_variables + control_variables

        # 步骤4: 为每个变量生成测量指标并查找最匹配的指标
        def process_variable(variable):
            print(f"\n正在处理变量 '{variable}'...")

            # 为变量生成建议的测量指标
            measurement_indicators = self.topic_agent.suggest_measurement_indicators(variable)

            # 查找最匹配的指标
            best_indicator = self.find_best_indicator_match(variable, measurement_indicators)

            if best_indicator:
                # 从搜索结果中直接获取指标信息
                indicator_name = best_indicator.get("docnm_kwd", "")
                statistical_unit = best_indicator.get("Statistical unit", "")
                source = best_indicator.get("Source", "")
                content = best_indicator.get("content_with_weight", "")

                # 返回提取的数据
                return {
                    "variable": variable,
                    "indicator_name": indicator_name,
                    "description": content,
                    "statistical_unit": statistical_unit if statistical_unit else "未指定",
                    "source": source if source else "未指定",
                    "is_core_variable": variable in core_variables,
                    "suggested_indicators": measurement_indicators
                }
            else:
                print(f"未找到变量 '{variable}' 的对应指标")
                return None

        # 使用线程池并行处理变量
        print(f"开始并行处理 {len(all_variables)} 个变量...")
        all_indicators = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_variable = {executor.submit(process_variable, variable): variable
                                  for variable in all_variables}

            # 收集结果
            for future in future_to_variable:
                result = future.result()
                if result:
                    all_indicators.append(result)

        print(f"变量指标匹配完成，共找到 {len(all_indicators)} 个有效指标")

        # 步骤5: 从MongoDB一次性批量检索所有指标数据
        print("\n开始从MongoDB批量检索所有指标数据...")
        indicator_names = [info["indicator_name"] for info in all_indicators if info["indicator_name"]]

        all_data = pd.DataFrame()
        if indicator_names:
            # 调用批量检索方法一次性获取所有数据
            all_data = self.mongo_connector.get_data_for_all_indicators(indicator_names)

            if all_data.empty:
                print("数据库中未找到任何指标数据")
            else:
                print(f"\n数据检索成功，最终数据集规模: {all_data.shape[0]}行 x {all_data.shape[1]}列")

                # 添加额外的研究信息到指标元数据
                for indicator in all_indicators:
                    indicator["research_topic"] = main_topic

                # 打印数据集概况
                print("\n研究数据概况:")
                print(f"- 研究主题: {main_topic}")
                print(f"- 核心变量: {', '.join(core_variables)}")
                print(f"- 控制变量: {', '.join(control_variables)}")
                print(f"- 数据规模: {all_data.shape[0]}条记录")
                print(f"- 指标数量: {len(indicator_names)}个")

                # 打印每个变量的推荐测量指标和最终选择的指标
                print("\n变量与测量指标对应关系:")
                for info in all_indicators:
                    var_type = "核心变量" if info["is_core_variable"] else "控制变量"
                    print(f"- {info['variable']} ({var_type}):")
                    print(f"  推荐测量指标: {', '.join(info.get('suggested_indicators', []))}")
                    print(f"  选择的指标: {info['indicator_name']}")
        else:
            print("未找到任何有效指标，无法检索数据")

        return all_data, all_indicators
