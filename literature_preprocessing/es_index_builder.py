import time
import uuid
import jieba
from datetime import datetime
from pymongo import MongoClient
from elasticsearch import Elasticsearch
import argparse
import random
import requests
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
from components.config import *

mongo_client = MongoClient(
        f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"
    )

# 使用新的数据库和集合名称
db = mongo_client['data_analysis_v2']
collection = db['article_info_v2']

# Elasticsearch连接参数
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False
)

# BGE API配置
BGE_API_URL = "https://api.siliconflow.cn/v1/embeddings"
BGE_API_TOKEN = random.choice(BGE_API_KEY)


def convert_to_dict(kb_id, mongo_doc, field_name, field_content):
    """
    Convert field content to a dictionary with specified format.

    Args:
        kb_id (str): The kb_id to be included in the dictionary
        mongo_doc (dict): MongoDB document containing necessary fields
        field_name (str): Name of the field being processed (summary/concept_answer)
        field_content (str): Content of the field

    Returns:
        dict: A dictionary with the specified format
    """
    # 生成唯一的doc_id，格式为：原文档id_字段名
    doc_id = mongo_doc.get('id', '')
    unique_id = f"{doc_id}_{field_name}"

    # Create the docnm_kwd from mongo_doc's id + ".txt"
    docnm_kwd = f"{doc_id}.txt"

    # Create title_tks and title_sm_tks from lowercase id
    title_tks = str(doc_id).lower()
    title_sm_tks = title_tks

    # Get the current timestamp
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_float = str(time.time())

    # 执行jieba分词
    content_tokens = ' '.join(jieba.cut(field_content))

    # Create the dictionary with the required format
    return {
        "doc_id": unique_id,  # 使用新的唯一ID格式
        "kb_id": kb_id,
        "docnm_kwd": docnm_kwd,
        "title_tks": title_tks,
        "title_sm_tks": title_sm_tks,
        "content_with_weight": field_content,
        "content_ltks": content_tokens,  # 使用jieba分词结果
        "content_sm_ltks": content_tokens,
        "create_time": timestamp_str,
        "create_timestamp_flt": timestamp_float,
        "img_id": "",
        "q_1024_vec": [],
        "page_range": mongo_doc.get("page_range", ""),
        "keywords": mongo_doc.get("keywords", ""),
        "year": mongo_doc.get("year", ""),
        "subject": mongo_doc.get("subject", ""),
        "link": mongo_doc.get("link", ""),
        "affiliations": mongo_doc.get("affiliations", ""),
        "pdfpage_count": mongo_doc.get("pdfpage_count", ""),
        "translated_keywords": mongo_doc.get("translated_keywords", ""),
        "pdf_url": mongo_doc.get("pdf_url", ""),
        "language": mongo_doc.get("language", ""),
        "title": mongo_doc.get("title", ""),
        "paper_number": mongo_doc.get("paper_number", ""),
        "reference": mongo_doc.get("reference", ""),
        "download": mongo_doc.get("download", 0),
        "database": mongo_doc.get("database", ""),
        "journal": mongo_doc.get("journal", ""),
        "translated_title": mongo_doc.get("translated_title", ""),
        "meta_data": 1,
        "vO": mongo_doc.get("vO", ""),
        "sn": mongo_doc.get("sn", ""),
        "issue": mongo_doc.get("issue", ""),
        "level": mongo_doc.get("level", ""),
        "abstract": mongo_doc.get("abstract", ""),
        "classification": mongo_doc.get("classification", ""),
        "impact_factor": mongo_doc.get("impact_factor", ""),
        "translated_abstract": mongo_doc.get("translated_abstract", ""),
        "authors": mongo_doc.get("authors", ""),
        "processed": 1,
        "chunk_type": field_name,  # 使用字段名作为chunk_type
        "unique": 1
    }


def get_embeddings(texts, max_retries=3):
    """获取文本的嵌入向量"""
    if not texts:
        return []

    payload = {
        "model": "BAAI/bge-m3",
        "input": texts,
        "encoding_format": "float"
    }

    headers = {
        "Authorization": f"Bearer {BGE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.post(BGE_API_URL, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]
                return np.array(embeddings)
            else:
                print(f"API错误: {response.status_code}")
                retry_count += 1
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
        except Exception as e:
            retry_count += 1
            wait_time = 2 ** retry_count
            time.sleep(wait_time)

    print("无法获取嵌入向量")
    return np.array([])


def process_embeddings(docs, batch_size=64, max_retries=10):
    """
    Process embeddings for documents in batches.

    Args:
        docs (list): List of document dictionaries
        batch_size (int): Number of documents to process in each batch
        max_retries (int): Maximum number of retries for embedding API calls

    Returns:
        list: List of documents with embeddings added
    """
    processed_docs = docs.copy()
    total_docs = len(processed_docs)

    # Process in batches
    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        batch_docs = processed_docs[start_idx:end_idx]

        # Extract titles and contents for the current batch
        titles = [doc["title"] for doc in batch_docs]
        contents = [doc["content_with_weight"][:7500] for doc in batch_docs]

        # Process embeddings for this batch
        print(f"获取向量中... 处理 {start_idx + 1} 到 {end_idx} (共 {total_docs})")
        title_vectors = get_embeddings(titles, max_retries)
        content_vectors = get_embeddings(contents, max_retries)

        # Calculate weighted vectors for each document in the batch
        for i in range(len(batch_docs)):
            if i < len(title_vectors) and i < len(content_vectors):
                # Weighted calculation
                weighted_vec = 0.1 * title_vectors[i] + 0.9 * content_vectors[i]

                # Normalize
                norm = np.linalg.norm(weighted_vec)
                if norm > 0:
                    weighted_vec = weighted_vec / norm

                # Save to document
                processed_docs[start_idx + i]["q_1024_vec"] = weighted_vec.tolist()

        print(f"已完成批次 {start_idx // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}")

    return processed_docs


def process_field_documents(kb_id, skip=0):
    """
    处理包含summary和concept_answer字段的文档

    Args:
        kb_id (str): 知识库ID
        skip (int): 跳过前几条文档，默认为0

    Returns:
        bool: 处理是否成功
    """
    try:
        # 查询包含summary字段且相应字段未处理的文档
        query = {
            "$or": [
                {
                    "$and": [
                        {"summary": {"$ne": "", "$exists": True}},
                        {"summary_upload": {"$ne": 1}}
                    ]
                }
            ]
        }

        cursor = collection.find(query, projection={"text": 0}).skip(skip).limit(1)
        documents = list(cursor)

        if not documents:
            print("没有更多包含目标字段的文档需要处理")
            return True

        document_processed = False
        for doc in documents:
            document_processed = True
            doc_id = doc.get("id")
            print(f"处理文档：{doc_id}")

            if not doc_id:
                print(f"警告: 文档没有id字段")
                return False

            docs_to_upload = []
            fields_to_process = []

            # 检查summary字段
            if doc.get("summary") and doc.get("summary_upload") != 1:
                summary_content = doc["summary"]
                summary_dict = convert_to_dict(kb_id, doc, "summary", summary_content)
                docs_to_upload.append(summary_dict)
                fields_to_process.append("summary")
                print(f"添加summary字段到处理队列")


            if not docs_to_upload:
                print("没有需要处理的字段")
                return True

            # 处理嵌入向量
            final_docs = process_embeddings(docs_to_upload)

            # 上传到Elasticsearch
            print(f"准备上传 {len(final_docs)} 个文档...")
            start_time = time.time()

            def upload_doc(doc_item, index):
                try:
                    # 使用unique_id作为ES文档的唯一ID
                    es.index(index=DEFAULT_INDEX, id=doc_item["doc_id"], document=doc_item)
                    if (index + 1) % 10 == 0 or index == len(final_docs) - 1:
                        print(f"已上传 {index + 1}/{len(final_docs)} 个文档")
                    return True
                except Exception as e:
                    print(f"上传第 {index + 1} 个文档时发生错误: {str(e)}")
                    return False

            # 使用多线程并行上传
            upload_success = True
            with ThreadPoolExecutor(max_workers=30) as executor:
                future_to_index = {executor.submit(upload_doc, doc_item, i): i
                                   for i, doc_item in enumerate(final_docs)}

                results = []
                for future in future_to_index:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"执行线程时出错: {str(e)}")
                        results.append(False)

                upload_success = all(results)

            end_time = time.time()
            print(f"上传完成，耗时: {end_time - start_time:.2f} 秒")

            # 如果上传成功，更新MongoDB中的处理标记
            if upload_success:
                update_fields = {}
                for field in fields_to_process:
                    if field == "summary":
                        update_fields["summary_upload"] = 1

                collection.update_one(
                    {"id": doc_id},
                    {"$set": update_fields}
                )
                print(f"文档 {doc_id} 的字段 {fields_to_process} 处理完成并已标记")
                return True
            else:
                print(f"文档 {doc_id} 上传不完整，未标记为已处理")
                # 清理已上传的文档
                for doc_item in final_docs:
                    try:
                        delete_query = {
                            "query": {
                                "term": {"doc_id": doc_item["doc_id"]}
                            }
                        }
                        es.delete_by_query(index=DEFAULT_INDEX, body=delete_query, refresh=True)
                    except Exception as e:
                        print(f"清理文档时出错: {str(e)}")
                return False

        if not document_processed:
            print("没有找到需要处理的文档")
            return True

        print(f"完成处理知识库 {kb_id} 的字段文档")
        return True

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False


def main():
    """
    主函数，处理命令行参数并执行文档处理，持续循环处理文档
    """
    parser = argparse.ArgumentParser(description='处理知识库中包含summary字段的文档')

    parser.add_argument('--kb_id', type=str, default="3dcd9e360c6811f081000242ac120004", help='知识库ID')
    parser.add_argument('--skip', type=int, default=0, help='跳过前几条文档，默认为0')

    args = parser.parse_args()

    kb_id = args.kb_id
    current_skip = args.skip

    print(f"开始处理知识库 {kb_id} 的文档，初始跳过前 {current_skip} 条")
    total_processed = 0
    total_failed = 0
    start_time = time.time()

    try:
        while True:
            print(f"当前跳过设置: {current_skip}")

            try:
                success = process_field_documents(kb_id, args.skip)

                if success:
                    total_processed += 1
                    print(f"成功处理文档，总成功数: {total_processed}")
                else:
                    current_skip += 1
                    total_failed += 1
                    print(f"处理失败，跳过值增加为 {current_skip}，总失败数: {total_failed}")

            except Exception as e:
                print(f"处理过程中发生错误: {str(e)}")
                current_skip += 1
                total_failed += 1
                print(f"错误后跳过值增加为 {current_skip}，总失败数: {total_failed}")

            current_time = time.time()
            elapsed_time = current_time - start_time
            print(f"已运行: {elapsed_time:.2f} 秒，成功: {total_processed}，失败: {total_failed}")

            time.sleep(0)

    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        print(f"最终结果 - 跳过值: {current_skip}, 成功: {total_processed}, 失败: {total_failed}")

    except Exception as e:
        print(f"程序异常退出: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()