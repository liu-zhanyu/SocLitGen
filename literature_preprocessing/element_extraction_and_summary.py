from zhipuai import ZhipuAI
from pymongo import MongoClient
import random
import time
from datetime import datetime
import argparse
import pandas as pd
from components.config import *

# ================== 新增配置部分 ==================


def load_classification_field_index(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    classification_field_index = {}
    for index, row in df.iterrows():
        classification = row['classification']
        fields = row['field'].split(' ') if row['field'] else []
        classification_field_index[classification] = fields
    return classification_field_index


# 从 Excel 文件中加载 field_definition_index
def load_field_definition_index(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet2')
    field_definition_index = {}
    for index, row in df.iterrows():
        field = row['field']
        field_definition = {
            "question": row['question'],
            "logic": row['logic'],
            "note": row['note'],
            "standard": row['standard']
        }
        field_definition_index[field] = field_definition
    return field_definition_index


classification_field_index = load_classification_field_index("/root/research_config.xlsx")
field_definition_index = load_field_definition_index("/root/research_config.xlsx")
# ================================================


# 初次连接 MongoDB 服务器
def connect_to_mongo():
    client = MongoClient(
        f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"
    )
    client.admin.command('ping')
    return client


# 检查 MongoDB 连接状态
def is_connected(client):
    try:
        client.admin.command('ping')
        return True
    except:
        return False


# 读取 API keys
def read_keys(filename):
    keys = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            keys.append(line.strip())
    return keys


# 更新后的模板
template = """
根据以下要求处理文本内容：
{text}

需要解决的问题：{question}
问题分析逻辑：{logic}
重要提示说明：{note}
答案生成标准：{standard}

请按照上述要求生成中文答案：
"""

# 文献类型模板映射
literature_templates = {
    '案例类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<理论依据>{theory_answer}</理论依据>
<研究方法>{method_answer}</研究方法>
<数据来源>{data_answer}</数据来源>
<样本描述>{sample_answer}</样本描述>
<研究结论>{conclusion_answer}</研究结论>
<实践启示>{implication_answer}</实践启示>

<语句结构>
{author}（{year}）基于某理论依据，采用某研究方法对某样本的数据进行分析，得出某研究结论，并提出某研究贡献和实践启示。
</语句结构>
""",

    '二手数据分析类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<理论依据>{theory_answer}</理论依据>
<研究方法>{method_answer}</研究方法>
<样本描述>{sample_answer}</样本描述>
<数据来源>{data_answer}</数据来源>
<分析方法>{analysis_answer}</分析方法>
<研究结论>{conclusion_answer}</研究结论>
<实践启示>{implication_answer}</实践启示>


<语句结构>
{author}（{year}）基于某理论依据，遵循某研究方法的逻辑，采用某分析方法分析了某样本的数据，得出某研究结论，并提出某研究贡献和实践启示。
</语句结构>
""",

    '决策模拟类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究问题>{question_answer}</研究问题>
<理论依据>{theory_answer}</理论依据>
<研究方法>{method_answer}</研究方法>
<样本描述>{sample_answer}</样本描述>
<数据来源>{data_answer}</数据来源
<测量方法>{measurement_answer}</测量方法>
<研究结论>{conclusion_answer}</研究结论>
<实践启示>{implication_answer}</实践启示>

<语句结构>
{author}（{year}）针对某研究问题进行决策模拟，运用某理论依据构建模拟模型，通过某样本数据验证模型，得出某研究结论并提出某研究贡献和实践启示。
</语句结构>
""",

    '内容分析类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<理论依据>{theory_answer}</理论依据>
<研究方法>{method_answer}</研究方法>
<数据来源>{data_answer}</数据来源>
<分析方法>{analysis_answer}</分析方法>
<研究结论>{conclusion_answer}</研究结论>
<实践启示>{implication_answer}</实践启示>

<语句结构>
{author}（{year}）基于某理论依据，遵循某研究方法的逻辑，采用某分析方法分析了某样本的数据，得出某研究结论，并提出某研究贡献和实践启示。
</语句结构>
""",

    '实验类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究问题>{question_answer}</研究问题>
<理论依据>{theory_answer}</理论依据>
<研究方法>{method_answer}</研究方法>
<样本描述>{sample_answer}</样本描述>
<数据来源>{data_answer}</数据来源>
<研究结论>{conclusion_answer}</研究结论>
<实践启示>{implication_answer}</实践启示>

<语句结构>
{author}（{year}）基于某理论依据，采用某研究方法研究了某研究问题，在此过程中其招募了某样本采集数据来开展，结果发现了某研究结论，并提出某研究贡献和实践启示。
</语句结构>
""",

    '问卷调查类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究问题>{question_answer}</研究问题>
<理论依据>{theory_answer}</理论依据>
<研究方法>{method_answer}</研究方法>
<样本描述>{sample_answer}</样本描述>
<数据来源>{data_answer}</数据来源>
<研究结论>{conclusion_answer}</研究结论>
<实践启示>{implication_answer}</实践启示>

<语句结构>
{author}（{year}）基于某理论依据，采用某研究方法研究了某研究问题，在此过程中其招募了某样本采集数据来开展，结果发现了某研究结论，并提出某研究贡献和实践启示。
</语句结构>
""",

    '预测建模类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究问题>{question_answer}</研究问题>
<理论依据>{theory_answer}</理论依据>
<样本描述>{sample_answer}</样本描述>
<数据来源>{data_answer}</数据来源>
<分析方法>{analysis_answer}</分析方法>
<研究结论>{conclusion_answer}</研究结论>
<实践启示>{implication_answer}</实践启示>

<语句结构>
{author}（{year}）针对某研究问题构建预测模型，运用某理论依据和研究方法，通过某样本数据训练模型，采用某分析方法验证模型效果，得出某研究结论并提出某研究贡献。
</语句结构>
""",

    '综述类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究问题>{question_answer}</研究问题>
<核心概念>{concept_answer}</核心概念>
<研究方法>{method_answer}</研究方法>
<研究结论>{conclusion_answer}</研究结论>
<研究局限>{limitation_answer}</研究局限>

<语句结构>
{author}（{year}）围绕某研究问题或某核心概念进行系统综述，得出某研究结论并提出某研究贡献。
</语句结构>
""",

    '方法工具类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究问题>{question_answer}</研究问题>
<研究方法>{method_answer}</研究方法>
<研究结论>{conclusion_answer}</研究结论>

<语句结构>
{author}（{year}）针对某研究问题，提出了某研究方法，通过验证得出某研究结论。
</语句结构>
""",

    '理论观点类文献': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究背景>{background_answer}</研究背景>
<研究问题>{question_answer}</研究问题>
<核心概念>{concept_answer}</核心概念>
<研究结论>{conclusion_answer}</研究结论>

<语句结构>
{author}（{year}）基于某研究背景，针对某研究问题，围绕某核心概念提出理论观点，得出某研究结论并提出某研究贡献。
</语句结构>
""",

    '书评': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究背景>{background_answer}</研究背景>
<研究结论>{conclusion_answer}</研究结论>

<语句结构>
{author}（{year}）基于某研究背景，对相关著作进行评述，得出某研究结论。
</语句结构>
""",

    '新闻观点与会议资讯': """
请你将基于以下信息写一个文献概要，严格遵循<语句结构>的逻辑。

<作者>{author}</作者>
<年份>{year}</年份>
<研究背景>{background_answer}</研究背景>
<研究结论>{conclusion_answer}</研究结论>

<语句结构>
{author}（{year}）基于某研究背景，提出某观点和结论。
</语句结构>
"""
}


def generate_paper_summary(result, client):
    # 定义新的字段列表
    field_names = ['background', 'question', 'concept', 'theory', 'hypothesis',
                   'method', 'data', 'sample', 'measurement', 'analysis',
                   'conclusion', 'implication', 'limitation']

    # 获取各字段的答案
    answers_dict = {}
    for field in field_names:
        answer_field = f"{field}_answer"
        answers_dict[answer_field] = result.get(answer_field, "")

    author = result["authors"]
    year = result["year"]
    classification = result["classification"]

    # 根据文献分类选择对应的模板
    if classification in literature_templates:
        template = literature_templates[classification]
    else:
        # 默认使用理论观点类模板
        template = literature_templates['理论观点类文献']

    author_list = author.split(";")
    author_list = [author.strip().split(", ")[0].strip() for author in author_list if author.strip()]
    if len(author_list) >= 3:
        author = author_list[0] + "等人"
    else:
        author = "和".join(author_list)

    # 调用大语言模型生成总结，如果结果中出现"某"则重试，最多重试3次
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="GLM-4-flash",  # 填写需要调用的模型名称
                messages=[{
                    "role": "user",
                    "content": template.format(**answers_dict, year=year, author=author)
                }],
                temperature=1
            )

            result_content = response.choices[0].message.content
            # print(result_content)
            # 检查结果中是否包含"某"
            if "某" not in result_content:
                return result_content

            retry_count += 1
        except Exception as e:
            print(f"{datetime.now()}: 生成summary时出错: {str(e)}")
            if "contentFilter" in str(e):
                return 0
            retry_count += 1
            response = client.chat.completions.create(
                model="GLM-4-flash",  # 填写需要调用的模型名称
                messages=[{
                    "role": "user",
                    "content": template.format(**answers_dict, year=year, author=author)
                }],
                temperature=1
            )

            result_content = response.choices[0].message.content
            # 检查结果中是否包含"某"
            if "某" not in result_content:
                return result_content
    # 如果重试3次后仍然包含"某"，则返回空字符串
    return 0


def process_documents(start_index, batch_size):
    try:
        mongo_client = connect_to_mongo()
        db = mongo_client.get_database("data_analysis_v2")
        mongo_collection = db['LLLL_backup']

        keys = read_keys("/root/keys.txt")
        key = random.choice(keys)
        print(key)

        # 修改查询条件，检查新的字段
        field_names = ['background', 'question', 'concept', 'theory', 'hypothesis',
                       'method', 'data', 'sample', 'measurement', 'analysis',
                       'conlection', 'implication', 'limitation']

        # 构建字段存在的查询条件
        field_exists_conditions = []
        for field in field_names:
            answer_field = f"{field}_answer"
            field_exists_conditions.append({answer_field: {"$exists": True}})

        query = {
            "$and": [
                {"text": {"$ne": ""}},
                {"subject":{"$in": ["心理学", "教育学"]}},
                {"$or": [{"process": 2}, {"process": {"$exists": False}}]},
            ]
        }

        results = list(mongo_collection.find(query).skip(start_index).limit(batch_size))

        if not results:
            print(f"{datetime.now()}: 没有找到需要处理的文档")
            return False

        print(f"{datetime.now()}: 找到 {len(results)} 个文档需要处理")

        def process_field(field, result, field_config, keys):
            """处理单个字段的函数"""
            zhipu_client = ZhipuAI(api_key=key)

            # 构造答案字段名
            answer_field = f"{field}_answer"
            current_answer = result.get(answer_field, "")

            # 跳过已处理的字段
            if current_answer and len(current_answer) >= 50:
                return answer_field, current_answer, False, "success"

            # 生成动态prompt
            prompt = template.format(
                text=result["text"],
                question=field_config["question"],
                logic=field_config["logic"],
                note=field_config["note"],
                standard=field_config["standard"]
            )

            try:
                response = zhipu_client.chat.completions.create(
                    model="GLM-4-flash",
                    messages=[
                        {"role": "system",
                         "content": "你是一名社会科学领域的博士研究生，你在理解学术文献内容及寻找关键信息方面有很多的训练。请完成用户提出的任务。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1
                )

                new_answer = response.choices[0].message.content
                print(f"{datetime.now()}: 文档ID {result['id']}, 字段 {field} 已更新")
                return answer_field, new_answer, False, "success"

            except Exception as e:
                error_message = str(e)
                print(f"{datetime.now()}: 处理文档 {result['id']} 字段 {field} 时出错: {error_message}")
                with open('/root/results.txt', "a") as file:
                    file.write(f"{datetime.now()}, {result['id']}, {error_message}\n")

                if "敏感" in error_message or "1301" in error_message:
                    return answer_field, None, True, "sensitive"  # 敏感内容错误
                else:
                    return answer_field, None, True, "other_error"  # 其他错误

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        for index, result in enumerate(results):
            print(f"{datetime.now()}: 文档ID {result['id']}")
            try:
                update_fields = {}
                sensitive_occurred = False

                # 获取文献分类
                doc_classification = result.get('classification', 'default')
                field_list = classification_field_index.get(doc_classification, [])

                # 跳过没有字段定义的文档
                if not field_list:
                    print(f"{datetime.now()}: 文档ID {result['id']} 无有效分类字段")
                    field_list = classification_field_index.get("理论观点类文献", [])
                    
                if not is_connected(mongo_client):
                    mongo_client = connect_to_mongo()
                    db = mongo_client.get_database("data_analysis_v2")
                    mongo_collection = db['LLLL_backup']

                # 使用线程池处理字段
                with ThreadPoolExecutor(max_workers=16) as executor:
                    # 提交所有字段处理任务
                    future_to_field = {}
                    for field in field_list:
                        field_config = field_definition_index.get(field)
                        if field_config:
                            future = executor.submit(process_field, field, result, field_config, keys)
                            future_to_field[future] = field

                    # 收集结果 - 等待所有任务完成
                    for future in as_completed(future_to_field):
                        field = future_to_field[future]
                        try:
                            answer_field, new_answer, has_error, error_type = future.result()

                            if has_error:
                                if error_type == "sensitive":
                                    sensitive_occurred = True
                            elif new_answer:  # 成功生成新答案
                                update_fields[answer_field] = new_answer

                        except Exception as e:
                            error_message = str(e)
                            print(f"{datetime.now()}: 处理字段 {field} 时发生意外错误: {error_message}")
                            if "contentFilter" in error_message:
                                sensitive_occurred = True

                # 处理完所有字段后，判断是否有敏感内容
                if sensitive_occurred:
                    update_fields["process"] = 0
                    print(f"{datetime.now()}: 文档ID {result['id']} 包含敏感内容，标记为process=0")
                else:
                    # 没有敏感内容，正常处理summary生成逻辑
                    # 合并原始结果和更新字段来检查summary生成条件
                    merged_result = {**result, **update_fields}
                    has_required_fields = all(merged_result.get(f"{field}_answer", "") for field in field_list)
                    print(has_required_fields)

                    if has_required_fields:
                        zhipu_client = ZhipuAI(api_key=key)
                        summary = generate_paper_summary(merged_result, zhipu_client)
                        print("总结", summary)
                        if summary==0:
                            update_fields["process"] = 0
                            print(f"{datetime.now()}: 文档ID {result['id']} 包含敏感内容，标记为process=0")
                        elif summary is not None and summary != "":
                            update_fields["summary"] = summary
                            print(f"{datetime.now()}: 文档ID {result['id']}, 字段 summary 已更新")
                            update_fields["process"] = 1
                            
                        else:
                            update_fields["process"] = 2
                    else:
                        update_fields["process"] = 2

                mongo_collection.update_one(
                    {"id": result["id"]},
                    {"$set": update_fields}
                )

            except Exception as e:
                print(f"{datetime.now()}: 处理文档 {result['id']} 时出错: {str(e)}")
                with open('/root/results.txt', "a") as file:
                    file.write(f"{datetime.now()}, {result['id']}, {str(e)}\n")
                mongo_collection.update_one(
                    {"id": result['id']},
                    {"$set": {"process": 2}}
                )
                continue


        mongo_client.close()
        return True

    except Exception as e:
        print(f"{datetime.now()}: 主程序出错: {str(e)}")
        return False


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='文档处理程序')
    parser.add_argument('--start', type=int, required=True, help='起始序号')
    parser.add_argument('--batch', type=int, required=True, help='处理数量')
    args = parser.parse_args()

    print(f"{datetime.now()}: 程序启动 | 起始序号: {args.start} | 处理数量: {args.batch}")

    current_start = args.start  # 维护当前起始位置

    while True:
        try:
            process_result = process_documents(current_start, args.batch)
            print(f"{datetime.now()}: 等待20秒后继续...")
            time.sleep(0)
        except KeyboardInterrupt:
            print(f"{datetime.now()}: 程序被用户中断")
            break
        except Exception as e:
            print(f"{datetime.now()}: 发生错误: {str(e)}")
            time.sleep(0)


if __name__ == "__main__":
    main()