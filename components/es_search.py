import time
from elasticsearch import Elasticsearch

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import random
import jieba
import requests
import numpy as np
import math

import threading
import traceback
from collections import defaultdict

jieba.initialize()

KB_ID_PAPER = "302b50b61e7911f0822c0242ac120006"
KB_ID_CHUNK = "684df8ee1e7511f0a9ff0242ac120006"
KB_ID_SUMMARY="3dcd9e360c6811f081000242ac120004"

# 连接参数
ES_HOST = "http://43.134.113.96:1200"
# ES_HOST = "http://es01:9200"
ES_USER = "elastic"
ES_PASSWORD = "infini_rag_flow"

BGE_API_URL = "https://api.siliconflow.cn/v1/embeddings"
BGE_API_TOKEN = "sk-nmkifxhohubaoezcbfafmzojukdokvyvcekystkcolzxcyrc"

# 默认搜索参数
DEFAULT_INDEX = "ragflow_b7a5a24ce9ba11efa0ca0242ac120006"  # 替换为您的索引名称
# DEFAULT_INDEX = "ragflow_optimized_vectors"  # 替换为您的索引名称
DEFAULT_TOP_K = 30
DEFAULT_VECTOR_WEIGHT = 0.7  # 向量搜索的权重
DEFAULT_TEXT_WEIGHT = 0.3  # 文本搜索的权重
DEFAULT_CHUNK_TYPE = ["raw"]  # 默认的chunk_type值

DATA_INDICATORS_KB_ID = "ab8652a4f9aa11ef9d410242ac120006"  # 数据指标的知识库ID

class ElasticsearchService:
    """Elasticsearch服务类，提供向量搜索和混合搜索功能"""

    def __init__(self):
        self.es = None

    def connect(self) -> bool:
        """建立与Elasticsearch的连接"""
        connect_start = time.time()
        try:
            self.es = Elasticsearch(
                ES_HOST,
                basic_auth=(ES_USER, ES_PASSWORD),
                verify_certs=False,
                timeout=30
            )

            ping_start = time.time()
            if not self.es.ping():
                print("连接Elasticsearch失败")
                return False
            ping_end = time.time()
            print(f"[时间戳] ES ping 耗时: {(ping_end - ping_start):.3f}秒")

            # 获取Elasticsearch版本信息
            info = self.es.info()
            version = info.get('version', {}).get('number', 'unknown')
            print(f"成功连接到Elasticsearch (版本: {version})")

            connect_end = time.time()
            print(f"[时间戳] ES 连接总耗时: {(connect_end - connect_start):.3f}秒")
            return True
        except Exception as e:
            connect_end = time.time()
            print(f"[时间戳] ES 连接失败总耗时: {(connect_end - connect_start):.3f}秒")
            print(f"连接Elasticsearch时出错: {e}")
            self.es = None
            return False

    def get_vector_embedding(self, query_input: Union[str, List[str]]) -> Union[List[float], List[List[float]], None]:
        """
        从BGE API获取查询文本的向量嵌入，支持单个字符串或字符串列表输入

        Args:
            query_input: 查询文本，可以是单个字符串或字符串列表

        Returns:
            单个查询返回单个向量，多个查询返回向量列表，失败返回None
        """
        embedding_start = time.time()

        try:
            # 将输入标准化为列表格式
            if isinstance(query_input, str):
                is_single_input = True
                query_texts = [query_input]
            else:
                is_single_input = False
                query_texts = query_input

            # 确保输入不为空
            if not query_texts or all(not text.strip() for text in query_texts):
                print("错误：查询文本为空")
                return None

            # 准备请求载荷
            payload = {
                "model": "BAAI/bge-m3",
                "input": query_texts,
                "encoding_format": "float"
            }

            # 随机选择API令牌
            BGE_API_TOKEN = random.choice(["sk-nmkifxhohubaoezcbfafmzojukdokvyvcekystkcolzxcyrc",
                                           "sk-xlmslbkpkremrpniafrcdpetyzjbxbohlqgpyqbyrhlqahod"])

            headers = {
                "Authorization": f"Bearer {BGE_API_TOKEN}",
                "Content-Type": "application/json"
            }

            # 发送API请求
            api_request_start = time.time()
            response = requests.post(BGE_API_URL, json=payload, headers=headers)
            api_request_end = time.time()
            print(
                f"[时间戳] BGE API 请求耗时: {(api_request_end - api_request_start):.3f}秒 (输入数量: {len(query_texts)})")

            # 处理响应
            if response.status_code == 200:
                json_start = time.time()
                data = response.json()
                json_end = time.time()
                print(f"[时间戳] API 响应JSON解析耗时: {(json_end - json_start):.3f}秒")

                if "data" in data and len(data["data"]) > 0:
                    # 处理所有返回的向量
                    norm_start = time.time()
                    vectors = []

                    for item in data["data"]:
                        if "embedding" in item:
                            vector = item["embedding"]
                            # 归一化向量
                            norm = np.linalg.norm(vector)
                            if norm > 0:
                                vector = [x / norm for x in vector]
                            vectors.append(vector)

                    norm_end = time.time()
                    print(f"[时间戳] 向量归一化耗时: {(norm_end - norm_start):.3f}秒")

                    # 根据输入类型返回相应的结果
                    embedding_end = time.time()
                    print(f"[时间戳] 向量嵌入总耗时: {(embedding_end - embedding_start):.3f}秒")

                    if is_single_input and vectors:
                        return vectors[0]  # 如果是单个输入，返回单个向量
                    else:
                        return vectors  # 如果是多个输入，返回向量列表
                else:
                    print("API返回值中没有找到向量数据")
                    embedding_end = time.time()
                    print(f"[时间戳] 向量嵌入总耗时(无数据): {(embedding_end - embedding_start):.3f}秒")
                    return None
            else:
                print(f"API错误: {response.status_code}, {response.text}")
                embedding_end = time.time()
                print(f"[时间戳] 向量嵌入总耗时(API错误): {(embedding_end - embedding_start):.3f}秒")
                return None
        except Exception as e:
            embedding_end = time.time()
            print(f"[时间戳] 向量嵌入总耗时(发生异常): {(embedding_end - embedding_start):.3f}秒")
            print(f"获取向量嵌入时出错: {e}")
            return None

    def tokenize_with_jieba(self, text: str) -> str:
        """使用jieba分词处理查询文本"""
        jieba_start = time.time()
        try:
            # 使用jieba进行分词
            tokens = jieba.cut(text, cut_all=False)
            # 将分词结果组合成空格分隔的字符串，这是ES的标准格式
            tokenized_text = " ".join(tokens)
            jieba_end = time.time()
            print(f"[时间戳] Jieba分词耗时: {(jieba_end - jieba_start):.3f}秒")
            return tokenized_text
        except Exception as e:
            jieba_end = time.time()
            print(f"[时间戳] Jieba分词出错: {(jieba_end - jieba_start):.3f}秒")
            print(f"Jieba分词时出错: {e}")
            return text  # 出错时返回原始文本

    def _build_journal_filter(self, journals: List[str]) -> List[Dict]:
        """构建期刊过滤条件，支持keyword字段"""
        if not journals:
            return []

        # 直接使用terms查询处理多个期刊
        journal_filter = {
            "terms": {
                "journal.keyword": journals
            }
        }

        return [journal_filter]

    def _build_author_filter(self, authors: Union[str, List[str]]) -> List[Dict]:
        """优化后的作者过滤方法"""
        if not authors:
            return []

        # 统一处理输入格式
        author_list = [authors] if isinstance(authors, str) else authors
        valid_authors = [a.strip() for a in author_list if a and a.strip()]

        if not valid_authors:
            return []

        # 合并多个作者的查询条件到单个bool/should中
        should_clauses = []
        for author in valid_authors:
            should_clauses.extend([
                {"match": {"authors": {"query": author, "operator": "and"}}}
            ])

        return [{
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }]

    def hybrid_search(
            self,
            query_text: str,
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            vector_weight: float = DEFAULT_VECTOR_WEIGHT,
            text_weight: float = DEFAULT_TEXT_WEIGHT,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = DEFAULT_CHUNK_TYPE,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """执行两阶段混合检索的统一入口函数"""
        if not self.es and not self.connect():
            return [], 0

        search_start = time.time()
        print(f"[时间戳] 开始混合搜索: {time.strftime('%H:%M:%S', time.localtime())}")

        try:
            # ========== 预处理阶段 ==========
            tokenized_query = self.tokenize_with_jieba(query_text)
            # 移除空字符串
            tokenized_query = ' '.join([token for token in tokenized_query.split() if token.strip()])
            print(f"原始查询: '{query_text}' -> 分词后: '{tokenized_query}'")


            query_vector = self.get_vector_embedding(query_text)
            if not query_vector:
                print("向量生成失败，退回纯文本搜索")
                return [], time.time() - search_start

            # ========== 统一构建过滤条件 ==========
            must_clauses = []

            # 添加知识库ID筛选
            if kb_id:
                print(f"应用知识库ID筛选: {kb_id}")
                must_clauses.append({"term": {"kb_id": kb_id}})

            # 添加文档类型筛选
            if chunk_type and len(chunk_type) > 0:
                print(f"应用文档类型筛选: {chunk_type}")
                must_clauses.append({"terms": {"chunk_type.keyword": chunk_type}})

            # 添加文档名称关键词筛选
            if docnm_kwds and len(docnm_kwds) > 0:
                print(f"应用文档名称筛选: {docnm_kwds}")
                must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

            # 添加期刊过滤条件
            if journals:
                print(f"应用期刊筛选条件: {journals}")
                journal_filters = self._build_journal_filter(journals)
                if len(journal_filters) == 1:
                    must_clauses.append(journal_filters[0])
                elif len(journal_filters) > 1:
                    # 当有多个期刊时，使用should组合
                    must_clauses.append({
                        "bool": {
                            "should": journal_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加作者过滤条件
            if authors:
                print(f"应用作者筛选条件: {authors}")
                author_filters = self._build_author_filter(authors)
                if len(author_filters) == 1:
                    must_clauses.append(author_filters[0])
                elif len(author_filters) > 1:
                    # 当有多个作者时，使用should组合
                    must_clauses.append({
                        "bool": {
                            "should": author_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加年份范围过滤条件
            if year_range and len(year_range) == 2:
                print(f"应用年份范围筛选条件: {year_range}")
                # 年份字段为long类型
                must_clauses.append({
                    "range": {
                        "year": {
                            "gte": year_range[0],
                            "lte": year_range[1]
                        }
                    }
                })

            # 添加文献语言
            if language:
                print(f"应用文献语言筛选: {language}")
                must_clauses.append({"term": {"language.keyword": language}})

            # ========== 第一阶段：文本检索 ==========
            text_query = {
                "_source": False,
                "size": 100,
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "should": [
                            {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 10}}},
                            {"match_phrase": {"content_ltks": {"query": query_text, "boost": text_weight * 5}}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            text_results = self.es.search(index=index_name, body=text_query, request_timeout=30)
            text_ids = [hit["_id"] for hit in text_results.get("hits", {}).get("hits", [])]
            print(f"文本阶段获取候选文档数: {len(text_ids)}")

            # ========== 第二阶段：向量检索 ==========
            vector_query = {
                "_source": ["title", "abstract", "authors", "journal", "year", "vO", "issue",
                            "page_range", "doc_id", "kb_id", "chunk_type", "content_with_weight",
                            "pdf_url", "level", "subject", "impact_factor", "reference", "docnm_kwd","translated_abstract","language"],
                "size": top_k,
                "knn": {
                    "field": "q_1024_vec",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": min(top_k * 2, 50),
                    "filter": {
                        "bool": {
                            "must": must_clauses + [{"terms": {"_id": text_ids}}]
                        }
                    }
                }
            }
            vector_results = self.es.search(index=index_name, body=vector_query, request_timeout=30)
            vector_hits = vector_results.get("hits", {}).get("hits", [])

            # ========== 结果融合 ==========
            text_rank_dict = {doc_id: idx + 1 for idx, doc_id in enumerate(text_ids)}
            final_results = []

            for vector_rank, hit in enumerate(vector_hits, 1):
                doc_id = hit["_id"]
                source = hit.get("_source", {})
                text_rank = text_rank_dict.get(doc_id, 0)

                # RRF分数计算
                text_rrf = 1 / (60 + text_rank) if text_rank else 0
                vector_rrf = 1 / (60 + vector_rank)
                combined_score = (text_rrf * text_weight) + (vector_rrf * vector_weight)

                final_results.append({
                    "id": doc_id,
                    "score": combined_score,
                    **{k: source.get(k, "") for k in [
                        'title', 'abstract', 'authors', 'journal', 'year', 'vO',
                        'issue', 'page_range', 'doc_id', 'kb_id', 'chunk_type',
                        'content_with_weight', 'pdf_url', 'level', 'subject',
                        'impact_factor', 'reference', 'docnm_kwd',"translated_abstract","language"
                    ]}
                })

            final_results.sort(key=lambda x: x["score"], reverse=True)
            elapsed_time = time.time() - search_start
            print(f"最终返回结果数: {len(final_results)}, 耗时: {elapsed_time:.2f}s")
            return final_results[:top_k], elapsed_time

        except Exception as e:
            error_time = time.time() - search_start
            print(f"混合搜索异常: {str(e)}")
            return [], error_time

    def multi_hybrid_search(
            self,
            query_texts: List[str],
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            vector_weight: float = DEFAULT_VECTOR_WEIGHT,
            text_weight: float = DEFAULT_TEXT_WEIGHT,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = DEFAULT_CHUNK_TYPE,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Dict[str, Tuple[List[Dict[str, Any]], float]]:
        """
        执行多个查询的两阶段混合检索，使用msearch批量处理

        Args:
            query_texts: 需要搜索的多个查询文本
            index_name: 索引名称
            top_k: 每个查询返回的搜索结果数量
            vector_weight: 向量搜索的权重
            text_weight: 文本搜索的权重
            kb_id: 知识库ID过滤
            chunk_type: 文档类型过滤列表
            docnm_kwds: 文档名称关键词过滤列表
            journals: 期刊过滤条件
            authors: 作者过滤条件
            year_range: 年份范围过滤，格式为 [start_year, end_year]
            language: 文献语言过滤

        Returns:
            查询文本到(搜索结果列表, 耗时)的映射词典
        """
        if not self.es and not self.connect():
            return {query: ([], 0) for query in query_texts}

        # 过滤空查询
        valid_queries = [q for q in query_texts if q.strip()]
        if not valid_queries:
            print("没有有效的查询可以搜索")
            return {}

        search_start = time.time()
        print(f"[时间戳] 开始多查询混合搜索: {time.strftime('%H:%M:%S', time.localtime())}")
        print(f"将为 {len(valid_queries)} 个查询执行批量搜索")

        results = {}

        try:
            # ========== 预处理阶段 ==========
            # 为每个查询构建向量和分词结果
            query_data = {}

            for query in valid_queries:
                tokenized_query = self.tokenize_with_jieba(query)
                query_vector = self.get_vector_embedding(query)

                if not query_vector:
                    print(f"查询 '{query}' 向量生成失败，跳过")
                    results[query] = ([], time.time() - search_start)
                    continue

                query_data[query] = {
                    "tokenized": tokenized_query,
                    "vector": query_vector
                }

            if not query_data:
                print("所有查询的向量生成均失败")
                return {query: ([], time.time() - search_start) for query in valid_queries}

            # ========== 统一构建过滤条件 ==========
            must_clauses = []

            # 添加知识库ID筛选
            if kb_id:
                print(f"应用知识库ID筛选: {kb_id}")
                must_clauses.append({"term": {"kb_id": kb_id}})

            # 添加文档类型筛选
            if chunk_type and len(chunk_type) > 0:
                print(f"应用文档类型筛选: {chunk_type}")
                must_clauses.append({"terms": {"chunk_type.keyword": chunk_type}})

            # 添加文档名称关键词筛选
            if docnm_kwds and len(docnm_kwds) > 0:
                print(f"应用文档名称筛选: {docnm_kwds}")
                must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

            # 添加期刊过滤条件
            if journals:
                print(f"应用期刊筛选条件: {journals}")
                journal_filters = self._build_journal_filter(journals)
                if len(journal_filters) == 1:
                    must_clauses.append(journal_filters[0])
                elif len(journal_filters) > 1:
                    # 当有多个期刊时，使用should组合
                    must_clauses.append({
                        "bool": {
                            "should": journal_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加作者过滤条件
            if authors:
                print(f"应用作者筛选条件: {authors}")
                author_filters = self._build_author_filter(authors)
                if len(author_filters) == 1:
                    must_clauses.append(author_filters[0])
                elif len(author_filters) > 1:
                    # 当有多个作者时，使用should组合
                    must_clauses.append({
                        "bool": {
                            "should": author_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加年份范围过滤条件
            if year_range and len(year_range) == 2:
                print(f"应用年份范围筛选条件: {year_range}")
                # 年份字段为long类型
                must_clauses.append({
                    "range": {
                        "year": {
                            "gte": year_range[0],
                            "lte": year_range[1]
                        }
                    }
                })

            # 添加文献语言
            if language:
                print(f"应用文献语言筛选: {language}")
                must_clauses.append({"term": {"language.keyword": language}})

            # ========== 第一阶段：文本检索（批量） ==========
            msearch_body = []

            for query, data in query_data.items():
                # 添加索引信息
                msearch_body.append({"index": index_name})

                # 构建文本检索查询
                text_query = {
                    "_source": False,
                    "size": 1000,
                    "query": {
                        "bool": {
                            "must": must_clauses,
                            "should": [
                                {"match": {"content_ltks": {"query": data["tokenized"], "boost": text_weight * 10}}},
                                {"match_phrase": {"content_ltks": {"query": query, "boost": text_weight * 5}}}
                            ],
                            "minimum_should_match": 1
                        }
                    }
                }
                msearch_body.append(text_query)

            # 执行批量文本搜索
            print(f"第一阶段：执行批量文本搜索，共 {len(query_data)} 个查询")
            text_search_start = time.time()
            text_msearch_results = self.es.msearch(body=msearch_body, request_timeout=30)
            text_search_time = time.time() - text_search_start
            print(f"第一阶段文本搜索完成，耗时: {text_search_time:.2f}秒")

            # 处理文本搜索结果
            text_search_data = {}
            for query, response in zip(query_data.keys(), text_msearch_results.get("responses", [])):
                # 提取文档ID
                text_ids = [hit["_id"] for hit in response.get("hits", {}).get("hits", [])]
                text_search_data[query] = {
                    "ids": text_ids,
                    "id_rank": {doc_id: idx + 1 for idx, doc_id in enumerate(text_ids)}
                }
                print(f"查询 '{query}' 文本阶段获取候选文档数: {len(text_ids)}")

            # ========== 第二阶段：向量检索（批量） ==========
            vector_msearch_body = []

            for query, data in query_data.items():
                # 获取该查询的文本搜索结果ID
                text_result = text_search_data.get(query, {"ids": []})
                if not text_result["ids"]:
                    print(f"查询 '{query}' 没有文本搜索结果，跳过向量搜索")
                    results[query] = ([], time.time() - search_start)
                    continue

                # 添加索引信息
                vector_msearch_body.append({"index": index_name})

                # 构建向量搜索查询
                vector_query = {
                    "_source": ["title", "abstract", "authors", "journal", "year", "vO", "issue",
                                "page_range", "doc_id", "kb_id", "chunk_type", "content_with_weight",
                                "pdf_url", "level", "subject", "impact_factor", "reference", "docnm_kwd",
                                "translated_abstract", "language"],
                    "size": top_k,
                    "knn": {
                        "field": "q_1024_vec",
                        "query_vector": data["vector"],
                        "k": top_k,
                        "num_candidates": min(top_k * 3, 100),
                        "filter": {
                            "bool": {
                                "must": must_clauses + [{"terms": {"_id": text_result["ids"]}}]
                            }
                        }
                    }
                }
                vector_msearch_body.append(vector_query)

            # 执行批量向量搜索
            if vector_msearch_body:
                print(f"第二阶段：执行批量向量搜索，共 {len(vector_msearch_body) // 2} 个查询")
                vector_search_start = time.time()
                vector_msearch_results = self.es.msearch(body=vector_msearch_body, request_timeout=30)
                vector_search_time = time.time() - vector_search_start
                print(f"第二阶段向量搜索完成，耗时: {vector_search_time:.2f}秒")

                # 处理向量搜索结果并融合
                result_idx = 0
                for query in query_data.keys():
                    # 跳过没有文本搜索结果的查询
                    if query not in text_search_data or not text_search_data[query]["ids"]:
                        continue

                    vector_response = vector_msearch_results.get("responses", [])[result_idx]
                    result_idx += 1

                    vector_hits = vector_response.get("hits", {}).get("hits", [])
                    if not vector_hits:
                        print(f"查询 '{query}' 没有向量搜索结果")
                        results[query] = ([], time.time() - search_start)
                        continue

                    # 融合结果
                    text_rank_dict = text_search_data[query]["id_rank"]
                    final_results = []

                    for vector_rank, hit in enumerate(vector_hits, 1):
                        doc_id = hit["_id"]
                        source = hit.get("_source", {})
                        text_rank = text_rank_dict.get(doc_id, 0)

                        # RRF分数计算
                        text_rrf = 1 / (60 + text_rank) if text_rank else 0
                        vector_rrf = 1 / (60 + vector_rank)
                        combined_score = (text_rrf * text_weight) + (vector_rrf * vector_weight)

                        final_results.append({
                            "id": doc_id,
                            "score": combined_score,
                            **{k: source.get(k, "") for k in [
                                'title', 'abstract', 'authors', 'journal', 'year', 'vO',
                                'issue', 'page_range', 'doc_id', 'kb_id', 'chunk_type',
                                'content_with_weight', 'pdf_url', 'level', 'subject',
                                'impact_factor', 'reference', 'docnm_kwd', "translated_abstract", "language"
                            ]}
                        })

                    # 按得分排序
                    final_results.sort(key=lambda x: x["score"], reverse=True)
                    results[query] = (final_results[:top_k], time.time() - search_start)
                    print(f"查询 '{query}' 最终返回结果数: {len(final_results[:top_k])}")

            # 添加没有处理的查询（向量生成失败的）
            for query in valid_queries:
                if query not in results:
                    results[query] = ([], time.time() - search_start)

            total_elapsed = time.time() - search_start
            print(f"多查询混合搜索完成，共处理 {len(valid_queries)} 个查询，总耗时: {total_elapsed:.2f}秒")

            return results

        except Exception as e:
            error_time = time.time() - search_start
            print(f"多查询混合搜索异常: {str(e)}")
            traceback.print_exc()

            # 返回空结果
            return {query: ([], error_time) for query in valid_queries}

    def multi_text_search(
            self,
            query_texts: List[str],
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = DEFAULT_CHUNK_TYPE,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Dict[str, Tuple[List[Dict[str, Any]], float]]:
        """
        执行多个查询的文本搜索，使用msearch批量处理

        Args:
            query_texts: 需要搜索的多个查询文本
            index_name: 索引名称
            top_k: 每个查询返回的搜索结果数量
            kb_id: 知识库ID过滤
            chunk_type: 文档类型过滤列表
            docnm_kwds: 文档名称关键词过滤列表
            journals: 期刊过滤条件
            authors: 作者过滤条件
            year_range: 年份范围过滤，格式为 [start_year, end_year]
            language: 文献语言过滤

        Returns:
            查询文本到(搜索结果列表, 耗时)的映射词典
        """
        if not self.es and not self.connect():
            return {query: ([], 0) for query in query_texts}

        # 过滤空查询
        valid_queries = [q for q in query_texts if q.strip()]
        if not valid_queries:
            print("没有有效的查询可以搜索")
            return {}

        search_start = time.time()
        print(f"[时间戳] 开始多查询文本搜索: {time.strftime('%H:%M:%S', time.localtime())}")
        print(f"将为 {len(valid_queries)} 个查询执行批量搜索")

        results = {}

        try:
            # ========== 预处理阶段 ==========
            # 为每个查询构建分词结果
            query_data = {}

            for query in valid_queries:
                tokenized_query = self.tokenize_with_jieba(query)
                query_data[query] = {
                    "tokenized": tokenized_query,
                    "original": query
                }

            # ========== 统一构建过滤条件 ==========
            must_clauses = []

            # 添加知识库ID筛选
            if kb_id:
                print(f"应用知识库ID筛选: {kb_id}")
                must_clauses.append({"term": {"kb_id": kb_id}})

            # 添加文档类型筛选
            if chunk_type and len(chunk_type) > 0:
                print(f"应用文档类型筛选: {chunk_type}")
                must_clauses.append({"terms": {"chunk_type.keyword": chunk_type}})

            # 添加文档名称关键词筛选
            if docnm_kwds and len(docnm_kwds) > 0:
                print(f"应用文档名称筛选: {docnm_kwds}")
                must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

            # 添加期刊过滤条件
            if journals:
                print(f"应用期刊筛选条件: {journals}")
                journal_filters = self._build_journal_filter(journals)
                if len(journal_filters) == 1:
                    must_clauses.append(journal_filters[0])
                elif len(journal_filters) > 1:
                    # 当有多个期刊时，使用should组合
                    must_clauses.append({
                        "bool": {
                            "should": journal_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加作者过滤条件
            if authors:
                print(f"应用作者筛选条件: {authors}")
                author_filters = self._build_author_filter(authors)
                if len(author_filters) == 1:
                    must_clauses.append(author_filters[0])
                elif len(author_filters) > 1:
                    # 当有多个作者时，使用should组合
                    must_clauses.append({
                        "bool": {
                            "should": author_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加年份范围过滤条件
            if year_range and len(year_range) == 2:
                print(f"应用年份范围筛选条件: {year_range}")
                # 年份字段为long类型
                must_clauses.append({
                    "range": {
                        "year": {
                            "gte": year_range[0],
                            "lte": year_range[1]
                        }
                    }
                })

            # 添加文献语言
            if language:
                print(f"应用文献语言筛选: {language}")
                must_clauses.append({"term": {"language.keyword": language}})

            # ========== 文本搜索（批量） ==========
            msearch_body = []

            for query, data in query_data.items():
                # 添加索引信息
                msearch_body.append({"index": index_name})

                # 构建文本检索查询
                text_query = {
                    "_source": ["title", "abstract", "authors", "journal", "year", "vO", "issue",
                                "page_range", "doc_id", "kb_id", "chunk_type", "content_with_weight",
                                "pdf_url", "level", "subject", "impact_factor", "reference", "docnm_kwd",
                                "translated_abstract", "language"],
                    "size": top_k,
                    "query": {
                        "bool": {
                            "must": must_clauses,
                            "should": [
                                {"match": {"content_ltks": {"query": data["tokenized"], "boost": 2.0}}},
                                {"match_phrase": {"content_ltks": {"query": data["original"], "boost": 1.5}}},
                                {"match": {"title": {"query": data["original"], "boost": 1.2}}},
                                {"match": {"abstract": {"query": data["original"], "boost": 1.0}}}
                            ],
                            "minimum_should_match": 1
                        }
                    }
                }
                msearch_body.append(text_query)

            # 执行批量文本搜索
            print(f"执行批量文本搜索，共 {len(query_data)} 个查询")
            text_search_start = time.time()
            text_msearch_results = self.es.msearch(body=msearch_body, request_timeout=30)
            text_search_time = time.time() - text_search_start
            print(f"文本搜索完成，耗时: {text_search_time:.2f}秒")

            # 处理文本搜索结果
            for query, response in zip(query_data.keys(), text_msearch_results.get("responses", [])):
                hits = response.get("hits", {}).get("hits", [])

                if not hits:
                    print(f"查询 '{query}' 没有搜索结果")
                    results[query] = ([], time.time() - search_start)
                    continue

                # 构建最终结果
                final_results = []
                for hit in hits:
                    source = hit.get("_source", {})
                    final_results.append({
                        "id": hit["_id"],
                        "score": hit["_score"],
                        **{k: source.get(k, "") for k in [
                            'title', 'abstract', 'authors', 'journal', 'year', 'vO',
                            'issue', 'page_range', 'doc_id', 'kb_id', 'chunk_type',
                            'content_with_weight', 'pdf_url', 'level', 'subject',
                            'impact_factor', 'reference', 'docnm_kwd', "translated_abstract", "language"
                        ]}
                    })

                results[query] = (final_results, time.time() - search_start)
                print(f"查询 '{query}' 返回结果数: {len(final_results)}")

            total_elapsed = time.time() - search_start
            print(f"多查询文本搜索完成，共处理 {len(valid_queries)} 个查询，总耗时: {total_elapsed:.2f}秒")

            return results

        except Exception as e:
            error_time = time.time() - search_start
            print(f"多查询文本搜索异常: {str(e)}")
            traceback.print_exc()

            # 返回空结果
            return {query: ([], error_time) for query in valid_queries}

    def vector_search(
            self,
            query_text: str,
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = DEFAULT_CHUNK_TYPE,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """纯向量检索函数"""
        if not self.es and not self.connect():
            return [], 0

        search_start = time.time()
        print(f"[时间戳] 单篇阅读开始向量搜索: {time.strftime('%H:%M:%S', time.localtime())}")

        try:
            # ========== 预处理阶段 ==========
            query_vector = self.get_vector_embedding(query_text)
            if not query_vector:
                print("向量生成失败，返回空结果")
                return [], time.time() - search_start

            # ========== 统一构建过滤条件 ==========
            must_clauses = []

            # 添加知识库ID筛选
            if kb_id:
                print(f"应用知识库ID筛选: {kb_id}")
                must_clauses.append({"term": {"kb_id": kb_id}})

            # 添加文档类型筛选
            if chunk_type and len(chunk_type) > 0:
                print(f"应用文档类型筛选: {chunk_type}")
                must_clauses.append({"terms": {"chunk_type.keyword": chunk_type}})

            # 添加文档名称关键词筛选
            if docnm_kwds and len(docnm_kwds) > 0:
                print(f"应用文档名称筛选: {docnm_kwds}")
                must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

            # 添加期刊过滤条件
            if journals:
                print(f"应用期刊筛选条件: {journals}")
                journal_filters = self._build_journal_filter(journals)
                if len(journal_filters) == 1:
                    must_clauses.append(journal_filters[0])
                elif len(journal_filters) > 1:
                    must_clauses.append({
                        "bool": {
                            "should": journal_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加作者过滤条件
            if authors:
                print(f"应用作者筛选条件: {authors}")
                author_filters = self._build_author_filter(authors)
                if len(author_filters) == 1:
                    must_clauses.append(author_filters[0])
                elif len(author_filters) > 1:
                    must_clauses.append({
                        "bool": {
                            "should": author_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加年份范围过滤条件
            if year_range and len(year_range) == 2:
                print(f"应用年份范围筛选条件: {year_range}")
                must_clauses.append({
                    "range": {
                        "year": {
                            "gte": year_range[0],
                            "lte": year_range[1]
                        }
                    }
                })

            # 添加文献语言
            if language:
                print(f"应用文献语言筛选: {language}")
                must_clauses.append({"term": {"language.keyword": language}})

            # ========== 向量检索 ==========
            vector_query = {
                "_source": ["title", "abstract", "authors", "journal", "year", "vO", "issue",
                            "page_range", "doc_id", "kb_id", "chunk_type", "content_with_weight",
                            "pdf_url", "level", "subject", "impact_factor", "reference", "docnm_kwd",
                            "translated_abstract","language"],
                "size": top_k,
                "knn": {
                    "field": "q_1024_vec",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": min(top_k * 3, 100),
                    "filter": {
                        "bool": {
                            "must": must_clauses
                        }
                    }
                }
            }
            vector_results = self.es.search(index=index_name, body=vector_query, request_timeout=30)
            vector_hits = vector_results.get("hits", {}).get("hits", [])

            # 格式化结果
            final_results = []
            for hit in vector_hits:
                doc_id = hit["_id"]
                source = hit.get("_source", {})
                score = hit.get("_score", 0.0)

                final_results.append({
                    "id": doc_id,
                    "score": score,
                    **{k: source.get(k, "") for k in [
                        'title', 'abstract', 'authors', 'journal', 'year', 'vO',
                        'issue', 'page_range', 'doc_id', 'kb_id', 'chunk_type',
                        'content_with_weight', 'pdf_url', 'level', 'subject',
                        'impact_factor', 'reference', 'docnm_kwd', 'translated_abstract'
                    ]}
                })

            elapsed_time = time.time() - search_start
            print(f"最终返回结果数: {len(final_results)}, 耗时: {elapsed_time:.2f}s")
            return final_results, elapsed_time

        except Exception as e:
            error_time = time.time() - search_start
            print(f"向量搜索异常: {str(e)}")
            return [], error_time

    def search_literature(
            self,
            query: str,
            index_name: str = DEFAULT_INDEX,
            year_range: Optional[List[int]] = None,
            kb_id: Optional[str] = "3dcd9e360c6811f081000242ac120004",  # 写死KB_ID
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            language: Optional[str] = None,
            levels: Optional[list[str]] = None,
            page: int = 1,
            page_size: int = 10,
            text_weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        对外暴露的搜索函数，使用简单的文本检索，支持基本分页

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回结果数量
            year_range: 年份范围列表 [from_year, to_year]
            kb_id: 知识库ID过滤，固定值
            chunk_type: 文档类型过滤
            docnm_kwds: 文章列表，格式为[docnm_kwd1, docnm_kwd2, ...]
            journals: 期刊名称，字符串或字符串列表
            authors: 作者名称，字符串或字符串列表
            language: 文献语言
            level: 层级过滤
            page: 当前页码，从1开始
            page_size: 每页显示结果数
            text_weight: 文本权重

        Returns:
            包含检索结果和分页信息的字典
        """
        total_start = time.time()
        print(f"\n[时间戳] === 开始搜索文档 ===")
        print(f"[时间戳] 查询: '{query}', 知识库ID: {kb_id}, 分页信息: 第{page}页，每页{page_size}条")

        # 确保连接到ES
        if not self.es:
            es_connect_start = time.time()
            if not self.connect():
                total_end = time.time()
                print(f"[时间戳] 搜索总耗时(连接失败): {(total_end - total_start):.3f}秒")
                return {
                    "results": [],
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False
                }
            es_connect_end = time.time()
            print(f"[时间戳] ES连接函数调用耗时: {(es_connect_end - es_connect_start):.3f}秒")

        # 计算from参数
        from_param = (page - 1) * page_size

        # 对查询文本进行分词处理
        tokenized_query = self.tokenize_with_jieba(query)

        # 构建基本查询条件
        must_clauses = []

        # 添加固定的KB_ID过滤
        must_clauses.append({"term": {"kb_id": kb_id}})


        # 添加文档名称关键词筛选
        if docnm_kwds and len(docnm_kwds) > 0:
            print(f"[时间戳] 应用文档名称筛选: {docnm_kwds}")
            must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

        # 添加期刊过滤条件
        if journals:
            print(f"[时间戳] 应用期刊筛选条件: {journals}")
            journal_filters = self._build_journal_filter(journals)
            if len(journal_filters) == 1:
                must_clauses.append(journal_filters[0])
            elif len(journal_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": journal_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加作者过滤条件
        if authors:
            print(f"[时间戳] 应用作者筛选条件: {authors}")
            author_filters = self._build_author_filter(authors)
            if len(author_filters) == 1:
                must_clauses.append(author_filters[0])
            elif len(author_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": author_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加年份范围过滤条件
        if year_range and len(year_range) == 2:
            print(f"[时间戳] 应用年份范围筛选条件: {year_range}")
            must_clauses.append({
                "range": {
                    "year": {
                        "gte": year_range[0],
                        "lte": year_range[1]
                    }
                }
            })

        # 添加文献语言
        if language:
            print(f"[时间戳] 应用文献语言筛选: {language}")
            must_clauses.append({"term": {"language.keyword": language}})

        # 添加层级过滤
        if levels:
            print(f"[时间戳] 应用层级筛选: {levels}")
            must_clauses.append({"terms": {"level.keyword": levels}})

        # 构建查询体，使用与示例一致的结构
        query_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 10}}},
                        {"match_phrase": {"content_ltks": {"query": query, "boost": text_weight * 5}}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": [
                "title", "abstract", "authors", "journal", "year", "vO", "issue",
                "page_range", "pdf_url", "level", "subject", "impact_factor",
                "reference", "docnm_kwd", "translated_abstract", "language","content_with_weight"
            ],
            "track_scores": True,
            "track_total_hits": True,
            "sort": [
                {"_score": {"order": "desc"}}
            ],
            "size": page_size,
            "from": from_param
        }

        try:
            # 执行查询
            search_start = time.time()
            response = self.es.search(index=index_name, body=query_body)
            search_end = time.time()

            hits = response["hits"]["hits"]
            total_hits = response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else \
            response["hits"]["total"]

            # 计算分页信息
            total_pages = math.ceil(total_hits / page_size)
            has_next = page < total_pages
            has_prev = page > 1

            search_time = search_end - search_start
            total_end = time.time()
            total_time = total_end - total_start

            print(f"[时间戳] 文本检索耗时: {search_time:.3f}秒，总耗时: {total_time:.3f}秒")
            print(f"[时间戳] 找到结果总数: {total_hits}，当前页结果数: {len(hits)}")
            print(f"[时间戳] 当前页: {page}/{total_pages}")

            return {
                "results": hits,
                "total": total_hits,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }

        except Exception as e:
            print(f"[时间戳] 搜索出错: {str(e)}")

            total_end = time.time()
            print(f"[时间戳] 搜索总耗时(出错): {(total_end - total_start):.3f}秒")

            return {
                "results": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            }

    def hybrid_search_indicator(
            self,
            query_text: str,
            top_k: int = 3,
            vector_weight: float = DEFAULT_VECTOR_WEIGHT,
            text_weight: float = DEFAULT_TEXT_WEIGHT,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行结合向量和关键词的混合搜索，使用jieba分词优化中文查询
        """
        search_start = time.time()
        print(f"[时间] 开始混合搜索(jieba分词): {time.strftime('%H:%M:%S', time.localtime())}")

        # 使用jieba进行中文分词
        seg_list = jieba.cut(query_text, cut_all=False)
        segmented_query = " ".join(seg_list)
        print(f"分词后的查询: {segmented_query}")

        # 获取查询的向量嵌入
        query_vector = self.get_vector_embedding(query_text)

        if not query_vector:
            print("无法获取查询向量，将仅使用关键词搜索")
            return []

        # 构建筛选条件
        must_clauses = []

        # 添加文本搜索查询，使用分词后的查询
        text_match = {
            "match": {
                "content_ltks": {
                    "query": segmented_query,
                    "boost": text_weight * 10
                }
            }
        }

        # 添加筛选条件
        if filters:
            print(f"应用的筛选条件: {filters}")
            for field, value in filters.items():
                if isinstance(value, list):
                    must_clauses.append({"terms": {field: value}})
                elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                    must_clauses.append({"range": {field: value}})
                else:
                    must_clauses.append({"term": {field: value}})

        # 文本查询
        text_query = {
            "_source": ["docnm_kwd", "content_with_weight", "Statistical unit", "Source"],
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [text_match],
                    "minimum_should_match": 1
                }
            }
        }

        # 向量查询
        vector_query = {
            "_source": ["docnm_kwd", "content_with_weight", "Statistical unit", "Source"],
            "size": top_k,
            "knn": {
                "field": "q_1024_vec",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": min(top_k * 3, 100),
                "filter": {
                    "bool": {
                        "must": must_clauses
                    }
                }
            }
        }

        try:
            # 执行两种搜索
            text_results = self.es.search(index=DEFAULT_INDEX, body=text_query, request_timeout=30)
            vector_results = self.es.search(index=DEFAULT_INDEX, body=vector_query, request_timeout=30)

            # 提取结果
            text_hits = text_results.get("hits", {}).get("hits", [])
            vector_hits = vector_results.get("hits", {}).get("hits", [])

            # 合并结果（倒数排名融合）
            results_by_id = {}

            # 处理文本搜索结果
            for rank, hit in enumerate(text_hits, 1):
                doc_id = hit["_id"]
                source = hit.get("_source", {})
                text_score = hit.get("_score", 0)

                results_by_id[doc_id] = {
                    "id": doc_id,
                    "text_score": text_score,
                    "text_rank": rank,
                    "vector_score": 0,
                    "vector_rank": 0,
                    "source": source
                }

            # 处理向量搜索结果
            for rank, hit in enumerate(vector_hits, 1):
                doc_id = hit["_id"]
                source = hit.get("_source", {})
                vector_score = hit.get("_score", 0)

                if doc_id in results_by_id:
                    results_by_id[doc_id]["vector_score"] = vector_score
                    results_by_id[doc_id]["vector_rank"] = rank
                else:
                    results_by_id[doc_id] = {
                        "id": doc_id,
                        "text_score": 0,
                        "text_rank": 0,
                        "vector_score": vector_score,
                        "vector_rank": rank,
                        "source": source
                    }

            # 计算RRF分数（k=60是常用默认值）
            k = 60
            for doc_id, item in results_by_id.items():
                text_rrf = 0 if item["text_rank"] == 0 else 1.0 / (k + item["text_rank"])
                vector_rrf = 0 if item["vector_rank"] == 0 else 1.0 / (k + item["vector_rank"])

                # 加权合并
                item["score"] = (text_rrf * text_weight) + (vector_rrf * vector_weight)

            # 按合并分数排序
            sorted_results = sorted(results_by_id.values(), key=lambda x: x["score"], reverse=True)

            # 只保留top_k个结果
            final_results = sorted_results[:top_k]

            # 格式化结果
            results = []
            for item in final_results:
                source = item["source"]
                # 如果存在，从docnm_kwd中移除.txt
                docnm_kwd = source.get("docnm_kwd", "")
                if docnm_kwd.endswith(".txt"):
                    docnm_kwd = docnm_kwd[:-4]

                result = {
                    "id": item["id"],
                    "score": item["score"],
                    "docnm_kwd": docnm_kwd,
                    "content_with_weight": source.get("content_with_weight", ""),
                    "Statistical unit": source.get("Statistical unit", ""),
                    "Source": source.get("Source", ""),
                    "doc_id": source.get("doc_id", ""),
                    "kb_id": source.get("kb_id", "")
                }
                results.append(result)

            search_end = time.time()
            elapsed_time = search_end - search_start
            print(f"搜索完成，找到 {len(results)} 个结果，用时: {elapsed_time:.3f}秒")

            return results

        except Exception as e:
            print(f"混合搜索错误: {str(e)}")
            return []

    def search_documents(
            self,
            query: str,
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            vector_weight: float = 0.9,
            text_weight: float = 0.1,
            year_range: Optional[List[int]] = None,
            kb_id: Optional[str] = None,
            chunk_type: Optional[list[str]] = DEFAULT_CHUNK_TYPE,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        对外暴露的搜索函数，可以在其他模块中调用

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回结果数量
            vector_weight: 向量搜索权重(0-1)
            text_weight: 文本搜索权重(0-1)
            year_range: 年份范围列表 [from_year, to_year]
            kb_id: 知识库ID过滤
            chunk_type: 文档类型过滤
            docnm_kwds: 文章列表，格式为[docnm_kwd1, docnm_kwd2, ...]
            journals: 期刊名称，字符串或字符串列表
            authors: 作者名称，字符串或字符串列表

        Returns:
            检索结果列表
        """
        total_start = time.time()
        print(f"\n[时间戳] === 开始搜索文档 ===")
        print(f"[时间戳] 查询: '{query}', 知识库ID: {kb_id}, 文档类型: {chunk_type}")
        if journals:
            print(f"[时间戳] 期刊筛选: {journals}")
        if authors:
            print(f"[时间戳] 作者筛选: {authors}")
        if year_range:
            print(f"[时间戳] 年份范围: {year_range}")
        if docnm_kwds:
            print(f"[时间戳] 文档名称: {docnm_kwds}")
        if language:
            print(f"[时间戳] 文献语言: {language}")

        # 确保连接到ES
        if not self.es:
            es_connect_start = time.time()
            if not self.connect():
                total_end = time.time()
                print(f"[时间戳] 搜索总耗时(连接失败): {(total_end - total_start):.3f}秒")
                return []
            es_connect_end = time.time()
            print(f"[时间戳] ES连接函数调用耗时: {(es_connect_end - es_connect_start):.3f}秒")

        # 直接调用修改后的hybrid_search函数，传递所有筛选参数
        results, search_time = self.hybrid_search(
            query_text=query,
            index_name=index_name,
            top_k=top_k,
            vector_weight=vector_weight,
            text_weight=text_weight,
            kb_id=kb_id,
            chunk_type=chunk_type,
            docnm_kwds=docnm_kwds,
            journals=journals,
            authors=authors,
            year_range=year_range,
            language=language
        )

        total_end = time.time()
        total_time = total_end - total_start
        print(f"搜索完成，耗时: {search_time:.2f}秒，总耗时: {total_time:.2f}秒，找到结果数: {len(results)}")
        return results

    def search_documents_with_vector(
            self,
            query: str,
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            year_range: Optional[List[int]] = None,
            kb_id: Optional[str] = None,
            chunk_type: Optional[list[str]] = DEFAULT_CHUNK_TYPE,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        对外暴露的搜索函数，可以在其他模块中调用

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回结果数量
            vector_weight: 向量搜索权重(0-1)
            text_weight: 文本搜索权重(0-1)
            year_range: 年份范围列表 [from_year, to_year]
            kb_id: 知识库ID过滤
            chunk_type: 文档类型过滤
            docnm_kwds: 文章列表，格式为[docnm_kwd1, docnm_kwd2, ...]
            journals: 期刊名称，字符串或字符串列表
            authors: 作者名称，字符串或字符串列表

        Returns:
            检索结果列表
        """
        total_start = time.time()
        print(f"\n[时间戳] === 开始搜索文档 ===")
        print(f"[时间戳] 查询: '{query}', 知识库ID: {kb_id}, 文档类型: {chunk_type}")
        if journals:
            print(f"[时间戳] 期刊筛选: {journals}")
        if authors:
            print(f"[时间戳] 作者筛选: {authors}")
        if year_range:
            print(f"[时间戳] 年份范围: {year_range}")
        if docnm_kwds:
            print(f"[时间戳] 文档名称: {docnm_kwds}")
        if language:
            print(f"[时间戳] 文献语言: {language}")

        # 确保连接到ES
        if not self.es:
            es_connect_start = time.time()
            if not self.connect():
                total_end = time.time()
                print(f"[时间戳] 搜索总耗时(连接失败): {(total_end - total_start):.3f}秒")
                return []
            es_connect_end = time.time()
            print(f"[时间戳] ES连接函数调用耗时: {(es_connect_end - es_connect_start):.3f}秒")

        # 直接调用修改后的hybrid_search函数，传递所有筛选参数
        results, search_time = self.vector_search(
            query_text=query,
            index_name=index_name,
            top_k=top_k,
            kb_id=kb_id,
            chunk_type=None,
            docnm_kwds=docnm_kwds,
            journals=journals,
            authors=authors,
            year_range=year_range,
            language=language
        )

        total_end = time.time()
        total_time = total_end - total_start
        print(f"搜索完成，耗时: {search_time:.2f}秒，总耗时: {total_time:.2f}秒，找到结果数: {len(results)}")
        return results

    def hybrid_search_v3(
            self,
            query_text: str,
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            vector_weight: float = DEFAULT_VECTOR_WEIGHT,
            text_weight: float = DEFAULT_TEXT_WEIGHT,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = None,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], float]:
        """
        执行多阶段混合检索，采用分步骤筛选策略
        向量生成成功时：使用向量+文本混合检索
        向量生成失败时：退回纯文本检索
        """
        # 硬编码知识库ID

        if not self.es and not self.connect():
            return {"chunks": [], "doc_aggs": []}, 0

        search_start = time.time()
        print(f"[时间戳] 开始混合搜索: {time.strftime('%H:%M:%S', time.localtime())}")

        try:
            # ========== 共用预处理阶段 ==========
            tokenized_query = self.tokenize_with_jieba(query_text)
            print(f"原始查询: '{query_text}' -> 分词后: '{tokenized_query}'")

            # ========== 共用过滤条件处理 ==========
            must_clauses_without_kb_and_chunk_type = []  # 不包含kb_id和chunk_type的must条件
            chunk_type_condition = None  # 用户指定的chunk_type条件

            # 保存用户指定的chunk_type条件
            if chunk_type and len(chunk_type) > 0:
                print(f"保存文档类型筛选: {chunk_type}，将在后续检索中应用")
                chunk_type_condition = {"terms": {"chunk_type.keyword": chunk_type}}
            else:
                print("未指定文档类型，将在向量检索中排除chunk_type为raw的文档")
                chunk_type_condition = None

            # 添加文档名称关键词筛选
            if docnm_kwds and len(docnm_kwds) > 0:
                print(f"应用文档名称筛选: {docnm_kwds}")
                must_clauses_without_kb_and_chunk_type.append({"terms": {"docnm_kwd": docnm_kwds}})

            # 添加期刊过滤条件
            if journals:
                print(f"应用期刊筛选条件: {journals}")
                journal_filters = self._build_journal_filter(journals)
                if len(journal_filters) == 1:
                    must_clauses_without_kb_and_chunk_type.append(journal_filters[0])
                elif len(journal_filters) > 1:
                    must_clauses_without_kb_and_chunk_type.append({
                        "bool": {
                            "should": journal_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加作者过滤条件
            if authors:
                print(f"应用作者筛选条件: {authors}")
                author_filters = self._build_author_filter(authors)
                if len(author_filters) == 1:
                    must_clauses_without_kb_and_chunk_type.append(author_filters[0])
                elif len(author_filters) > 1:
                    must_clauses_without_kb_and_chunk_type.append({
                        "bool": {
                            "should": author_filters,
                            "minimum_should_match": 1
                        }
                    })

            # 添加年份范围过滤条件
            if year_range and len(year_range) == 2:
                print(f"应用年份范围筛选条件: {year_range}")
                must_clauses_without_kb_and_chunk_type.append({
                    "range": {
                        "year": {
                            "gte": year_range[0],
                            "lte": year_range[1]
                        }
                    }
                })

            # 添加文献语言
            if language:
                print(f"应用文献语言筛选: {language}")
                must_clauses_without_kb_and_chunk_type.append({"term": {"language.keyword": language}})

            # ========== 共用第一阶段：在KB_ID_PAPER中文本检索，获取docnm_kwd ==========
            first_stage_must_clauses = must_clauses_without_kb_and_chunk_type.copy()
            first_stage_must_clauses.append({"term": {"kb_id": KB_ID_PAPER}})

            sm_text_query = {
                "_source": ["docnm_kwd"],  # 只获取docnm_kwd
                "size": 100,  # 直接增大size，一次获取足够多的结果
                "query": {
                    "bool": {
                        "must": first_stage_must_clauses,
                        "should": [
                            {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 10}}},
                            {"match_phrase": {"content_ltks": {"query": query_text, "boost": text_weight * 5}}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                # 按相关性排序，提高准确性
                "sort": [
                    {"_score": {"order": "desc"}}
                ],
                # 增加预排序提高效率
                "track_scores": True,
                "track_total_hits": True
            }

            sm_text_search_start = time.time()
            sm_text_results = self.es.search(index=index_name, body=sm_text_query, request_timeout=10)
            sm_text_hits = sm_text_results.get("hits", {}).get("hits", [])
            sm_text_search_end = time.time()

            # 提取docnm_kwd列表
            docnm_kwd_list = []
            for hit in sm_text_hits:
                source = hit.get("_source", {})
                docnm_kwd = source.get("docnm_kwd", "")
                if docnm_kwd:
                    docnm_kwd_list.append(docnm_kwd)

            # 去重
            docnm_kwd_list = list(set(docnm_kwd_list))

            print(
                f"第一阶段(kb_id_paper)文本检索获取候选文档数: {len(docnm_kwd_list)}, 耗时: {sm_text_search_end - sm_text_search_start:.2f}s")

            if not docnm_kwd_list:
                print("第一阶段检索未找到结果，结束搜索")
                return {"chunks": [], "doc_aggs": []}, time.time() - search_start

            # ========== 共用字段定义 ==========
            content_fields = ["kb_id", "chunk_type", "content_with_weight", "docnm_kwd"]

            # ========== 分支逻辑：尝试生成向量 ==========
            query_vector = self.get_vector_embedding(query_text)

            if not query_vector:
                print("向量生成失败，退回纯文本搜索")
                # ========== 纯文本搜索分支 ==========

                # 构建纯文本检索的过滤条件
                text_must_clauses = [
                    {"term": {"kb_id": KB_ID_CHUNK}},
                    {"terms": {"docnm_kwd": docnm_kwd_list}}
                ]

                # 应用chunk_type条件
                text_must_not_clauses = []
                if chunk_type_condition:
                    text_must_clauses.append(chunk_type_condition)
                else:
                    # 排除raw类型的chunk
                    text_must_not_clauses.append({"term": {"chunk_type.keyword": "raw"}})

                text_search_query = {
                    "_source": content_fields + ["chunk_id"],
                    "size": top_k * 2,  # 获取更多结果用于后续过滤
                    "query": {
                        "bool": {
                            "must": text_must_clauses,
                            "must_not": text_must_not_clauses,
                            "should": [
                                {"match": {
                                    "content_with_weight": {"query": tokenized_query, "boost": text_weight * 10}}},
                                {"match_phrase": {
                                    "content_with_weight": {"query": query_text, "boost": text_weight * 5}}}
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "sort": [
                        {"_score": {"order": "desc"}}
                    ]
                }

                text_search_start = time.time()
                text_results = self.es.search(index=index_name, body=text_search_query, request_timeout=10)
                text_search_end = time.time()
                text_hits = text_results.get("hits", {}).get("hits", [])
                print(f"纯文本检索获取结果数: {len(text_hits)}, 耗时: {text_search_end - text_search_start:.2f}s")

                # 处理纯文本搜索结果
                chunks = []
                for hit in text_hits:
                    source = hit.get("_source", {})
                    chunk_content = {k: source.get(k, "") for k in content_fields}
                    chunk_content["id"] = hit["_id"]
                    chunk_content["chunk_id"] = source.get("chunk_id")
                    chunks.append(chunk_content)

            else:
                print("向量生成成功，执行向量+文本混合检索")
                # ========== 向量+文本混合检索分支 ==========

                # 第二阶段：在KB_ID_CHUNK中进行向量检索
                vector_must_clauses = [
                    {"term": {"kb_id": KB_ID_CHUNK}},
                    {"terms": {"docnm_kwd": docnm_kwd_list}}
                ]

                # 应用chunk_type条件
                vector_must_not_clauses = []
                if chunk_type_condition:
                    vector_must_clauses.append(chunk_type_condition)
                else:
                    # 排除raw类型的chunk
                    vector_must_not_clauses.append({"term": {"chunk_type.keyword": "raw"}})

                vector_query = {
                    "_source": content_fields + ["chunk_id"],
                    "size": 10,
                    "knn": {
                        "field": "q_1024_vec",
                        "query_vector": query_vector,
                        "k": 10,
                        "num_candidates": 100,  # 增加候选数以获得更好的向量检索结果
                        "filter": {
                            "bool": {
                                "must": vector_must_clauses,
                                "must_not": vector_must_not_clauses
                            }
                        }
                    }
                }

                vector_search_start = time.time()
                vector_results = self.es.search(index=index_name, body=vector_query, request_timeout=10)
                vector_search_end = time.time()
                vector_hits = vector_results.get("hits", {}).get("hits", [])
                print(f"向量检索获取前10个非raw结果，耗时: {vector_search_end - vector_search_start:.2f}s")

                # 第三阶段：在KB_ID_CHUNK中查找raw chunks (使用multi-search API)
                def extract_fields(source, field_list):
                    return {k: source.get(k, "") for k in field_list}

                # 生成chunks列表及收集需要获取raw chunks的docnm_kwd
                chunks = []
                unique_docnm_kwds = set()
                chunk_ids = []  # 存储所有chunk_id用于后续获取相邻chunk

                for hit in vector_hits:
                    source = hit.get("_source", {})
                    chunk_id = source.get("chunk_id")
                    docnm_kwd = source.get("docnm_kwd", "")
                    non_raw_content = extract_fields(source, content_fields)
                    non_raw_content["id"] = hit["_id"]
                    non_raw_content["chunk_id"] = chunk_id

                    # 非raw chunk添加到结果
                    chunks.append(non_raw_content)

                    # 收集所有unique的docnm_kwd
                    if docnm_kwd:
                        unique_docnm_kwds.add(docnm_kwd)

                # 使用multi-search API查询所有raw chunks
                raw_search_start = time.time()
                raw_chunks = []

                if unique_docnm_kwds:
                    # 构建msearch请求体
                    msearch_body = []
                    for docnm_kwd in unique_docnm_kwds:
                        # 添加空头部(使用默认索引)
                        msearch_body.append({})

                        # 添加查询部分
                        raw_query = {
                            "_source": content_fields + ["chunk_id"],  # 添加chunk_id字段
                            "size": 1,  # 每个docnm_kwd最多返回1个raw chunks
                            "knn": {
                                "field": "q_1024_vec",
                                "query_vector": query_vector,
                                "k": 3,
                                "num_candidates": 5,
                                "filter": {
                                    "bool": {
                                        "must": [
                                            {"term": {"kb_id": KB_ID_CHUNK}},
                                            {"term": {"docnm_kwd": docnm_kwd}},
                                            {"term": {"chunk_type.keyword": "raw"}}
                                        ]
                                    }
                                }
                            }
                        }
                        msearch_body.append(raw_query)

                    # 执行msearch请求
                    msearch_results = self.es.msearch(body=msearch_body, index=index_name, request_timeout=10)
                    responses = msearch_results.get("responses", [])

                    # 处理每个响应
                    for i, response in enumerate(responses):
                        if i >= len(unique_docnm_kwds):
                            break

                        raw_hits = response.get("hits", {}).get("hits", [])
                        if not raw_hits:
                            docnm_kwd = list(unique_docnm_kwds)[i]
                            print(f"未找到raw chunks (docnm_kwd={docnm_kwd})")
                            continue

                        # 处理找到的raw chunks
                        for raw_hit in raw_hits:
                            raw_source = raw_hit.get("_source", {})
                            chunk_id = raw_source.get("chunk_id")
                            raw_content = extract_fields(raw_source, content_fields)
                            raw_content["id"] = raw_hit["_id"]
                            raw_content["chunk_id"] = chunk_id
                            chunks.append(raw_content)
                            raw_chunks.append(raw_content)

                            # 收集chunk_id (只收集有效的chunk_id)
                            if chunk_id is not None:
                                chunk_ids.append(chunk_id)

                raw_search_end = time.time()
                print(f"获取raw chunks数量: {len(raw_chunks)}, 耗时: {raw_search_end - raw_search_start:.2f}s")

                # 第四阶段：获取相邻chunks (通过chunk_id获取前一个chunk)
                adjacent_search_start = time.time()

                # 按docnm_kwd分组整理raw chunks和它们的chunk_id
                raw_chunks_by_docnm = defaultdict(list)
                for raw_chunk in raw_chunks:
                    if "chunk_id" not in raw_chunk or raw_chunk["chunk_id"] is None or "docnm_kwd" not in raw_chunk:
                        continue
                    docnm_kwd = raw_chunk.get("docnm_kwd", "")
                    if docnm_kwd:
                        raw_chunks_by_docnm[docnm_kwd].append(raw_chunk)

                # 计算前一个chunk的ID (处理纯数字Long格式)
                adjacent_chunk_ids_by_docnm = defaultdict(list)  # 按docnm_kwd分组存储相邻chunk_id
                chunk_id_map = {}  # 用于存储chunk_id到其对应的前一个chunk_id的映射

                for docnm_kwd, chunks_in_doc in raw_chunks_by_docnm.items():
                    for raw_chunk in chunks_in_doc:
                        chunk_id = raw_chunk.get("chunk_id")
                        if chunk_id is None:
                            continue

                        # 处理chunk_id为long型数字
                        try:
                            num = int(chunk_id)

                            # 计算前一个chunk的ID
                            prev_chunk_id = num - 1 if num > 1 else None  # 当ID为1时不获取前一个

                            # 存储需要查询的前一个chunk_id，并关联到docnm_kwd
                            if prev_chunk_id is not None:
                                adjacent_chunk_ids_by_docnm[docnm_kwd].append(prev_chunk_id)

                            # 存储映射关系（包含docnm_kwd信息）
                            key = (num, docnm_kwd)  # 使用(chunk_id, docnm_kwd)作为key
                            chunk_id_map[key] = {
                                "prev": prev_chunk_id,
                                "docnm_kwd": docnm_kwd
                            }
                        except (ValueError, TypeError):
                            # 如果chunk_id格式不是数字，跳过
                            continue

                # 使用msearch批量获取所有前一个chunks
                if adjacent_chunk_ids_by_docnm:
                    adjacent_msearch_body = []

                    # 为每个docnm_kwd构建一个查询
                    for docnm_kwd, adj_ids in adjacent_chunk_ids_by_docnm.items():
                        # 去重
                        unique_adj_ids = list(set(adj_ids))

                        # 每个docnm_kwd最多处理50个ID，避免查询过大
                        for i in range(0, len(unique_adj_ids), 50):
                            batch_ids = unique_adj_ids[i:i + 50]

                            # 添加空头部
                            adjacent_msearch_body.append({})

                            # 添加查询 (使用terms查询批量获取，同时限制在相同docnm_kwd内)
                            adjacent_query = {
                                "_source": content_fields + ["chunk_id"],
                                "size": len(batch_ids),
                                "query": {
                                    "bool": {
                                        "must": [
                                            {"term": {"kb_id": KB_ID_CHUNK}},
                                            {"term": {"docnm_kwd": docnm_kwd}},  # 限制在相同docnm_kwd内
                                            {"terms": {"chunk_id": batch_ids}}
                                        ]
                                    }
                                }
                            }
                            adjacent_msearch_body.append(adjacent_query)

                    # 执行msearch请求
                    if adjacent_msearch_body:
                        adjacent_msearch_results = self.es.msearch(
                            body=adjacent_msearch_body,
                            index=index_name,
                            request_timeout=10
                        )
                        adjacent_responses = adjacent_msearch_results.get("responses", [])

                        # 处理相邻chunk结果
                        adjacent_chunks_by_key = {}  # 使用(chunk_id, docnm_kwd)作为key

                        for response in adjacent_responses:
                            adj_hits = response.get("hits", {}).get("hits", [])

                            for adj_hit in adj_hits:
                                adj_source = adj_hit.get("_source", {})
                                adj_chunk_id = adj_source.get("chunk_id")
                                adj_docnm_kwd = adj_source.get("docnm_kwd", "")

                                if adj_chunk_id is not None and adj_docnm_kwd:
                                    try:
                                        adj_chunk_id_int = int(adj_chunk_id)
                                        adj_content = extract_fields(adj_source, content_fields)
                                        adj_content["id"] = adj_hit["_id"]
                                        adj_content["chunk_id"] = adj_chunk_id
                                        # 使用复合键(chunk_id, docnm_kwd)
                                        key = (adj_chunk_id_int, adj_docnm_kwd)
                                        adjacent_chunks_by_key[key] = adj_content
                                    except (ValueError, TypeError):
                                        continue

                        # 处理raw chunks，为每个添加其前一个chunk
                        additional_chunks = []
                        for raw_chunk in raw_chunks:
                            if "chunk_id" not in raw_chunk or raw_chunk["chunk_id"] is None:
                                continue

                            docnm_kwd = raw_chunk.get("docnm_kwd", "")
                            if not docnm_kwd:
                                continue

                            try:
                                chunk_id_int = int(raw_chunk["chunk_id"])
                                key = (chunk_id_int, docnm_kwd)

                                if key in chunk_id_map:
                                    adj_info = chunk_id_map[key]

                                    # 添加前一个chunk的信息 (如果chunk_id > 1)
                                    prev_id = adj_info.get("prev")
                                    if prev_id is not None:
                                        prev_key = (prev_id, docnm_kwd)
                                        if prev_key in adjacent_chunks_by_key:
                                            prev_chunk = adjacent_chunks_by_key[prev_key].copy()  # 创建副本避免修改原对象
                                            additional_chunks.append(prev_chunk)
                            except (ValueError, TypeError):
                                continue

                        # 将找到的前一个chunks添加到结果中
                        chunks.extend(additional_chunks)

                adjacent_search_end = time.time()
                print(f"获取前一个相邻chunks耗时: {adjacent_search_end - adjacent_search_start:.2f}s")

            # ========== 共用后处理阶段 ==========

            # 获取元数据
            meta_search_start = time.time()
            doc_aggs = []
            used_docnm_kwds = set()

            # 收集所有最终结果中出现的docnm_kwd
            for chunk in chunks:
                docnm_kwd = chunk.get("docnm_kwd", "")
                if docnm_kwd:
                    used_docnm_kwds.add(docnm_kwd)

            if used_docnm_kwds:
                meta_fields = ["title", "abstract", "authors", "journal", "year", "vO", "issue",
                               "page_range", "pdf_url", "level", "subject", "impact_factor",
                               "reference", "docnm_kwd", "translated_abstract", "language"]

                # 一次性获取所有元数据
                meta_query = {
                    "_source": meta_fields,
                    "size": len(used_docnm_kwds),
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"kb_id": KB_ID_PAPER}},
                                {"terms": {"docnm_kwd": list(used_docnm_kwds)}}
                            ]
                        }
                    }
                }

                meta_results = self.es.search(index=index_name, body=meta_query, request_timeout=10)
                meta_hits = meta_results.get("hits", {}).get("hits", [])

                for hit in meta_hits:
                    source = hit.get("_source", {})
                    meta_data = {k: source.get(k, "") for k in meta_fields}
                    meta_data["id"] = hit["_id"]
                    doc_aggs.append(meta_data)

            meta_search_end = time.time()
            print(f"获取元数据耗时: {meta_search_end - meta_search_start:.2f}s")

            # 对结果进行排序和过滤
            # 过滤掉content_with_weight中数字过多或包含"作者简介"的chunk
            filtered_chunks = []
            for chunk in chunks:
                content = chunk.get("content_with_weight", "")

                # 检查是否包含"作者简介"
                if "作者简介" in content:
                    continue

                # 统计内容中的数字字符比例
                digit_count = sum(1 for c in content if c.isdigit())
                total_count = len(content) if content else 1  # 避免除零错误
                digit_ratio = digit_count / total_count

                # 如果数字比例小于30%，则保留此chunk
                if digit_ratio < 0.3:
                    filtered_chunks.append(chunk)

            # 按照docnm_kwd和chunk_id排序chunks
            def sort_key_for_chunks(chunk):
                docnm_kwd = chunk.get("docnm_kwd", "")
                chunk_id_str = chunk.get("chunk_id", "0")
                try:
                    chunk_id = int(chunk_id_str)
                except (ValueError, TypeError):
                    chunk_id = 0
                return (docnm_kwd, chunk_id)

            filtered_chunks.sort(key=sort_key_for_chunks)

            # 获取过滤后保留的docnm_kwd集合
            remaining_docnm_kwds = set()
            for chunk in filtered_chunks:
                docnm_kwd = chunk.get("docnm_kwd", "")
                if docnm_kwd:
                    remaining_docnm_kwds.add(docnm_kwd)
            print(f"过滤后保留的docnm_kwd: {remaining_docnm_kwds}")

            # 根据保留的docnm_kwd过滤doc_aggs
            filtered_doc_aggs = []
            for doc in doc_aggs:
                doc_kwd = doc.get("docnm_kwd", "")
                if doc_kwd in remaining_docnm_kwds:
                    filtered_doc_aggs.append(doc)

            print(f"过滤后doc_aggs数量: {len(filtered_doc_aggs)}")

            # 按照docnm_kwd排序doc_aggs
            filtered_doc_aggs.sort(key=lambda doc: doc.get("docnm_kwd", ""))

            # 按照top_k限制返回结果
            final_chunks = filtered_chunks[:top_k] if len(filtered_chunks) > top_k else filtered_chunks

            result = {
                "chunks": final_chunks,
                "doc_aggs": filtered_doc_aggs
            }

            elapsed_time = time.time() - search_start
            search_type = "混合检索" if query_vector else "纯文本检索"
            print(
                f"{search_type}完成: chunks数: {len(final_chunks)}, doc_aggs数: {len(filtered_doc_aggs)}, 总耗时: {elapsed_time:.2f}s")

            return result, elapsed_time

        except Exception as e:
            print(f"混合搜索异常: {str(e)}")
            traceback.print_exc()
            return {"chunks": [], "doc_aggs": []}, time.time() - search_start

    def _build_filter_conditions(
            self,
            chunk_type: Optional[List[str]] = None,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """
        构建过滤条件

        Args:
            chunk_type: 文档类型筛选条件
            docnm_kwds: 文档名称关键词筛选条件
            journals: 期刊筛选条件
            authors: 作者筛选条件
            year_range: 年份范围筛选条件
            language: 语言筛选条件

        Returns:
            Tuple[List[Dict], Optional[Dict]]: 返回must过滤条件列表和chunk_type条件
        """
        must_clauses = []  # 不包含kb_id和chunk_type的must条件
        chunk_type_condition = None  # 用户指定的chunk_type条件

        # 保存用户指定的chunk_type条件
        if chunk_type and len(chunk_type) > 0:
            print(f"保存文档类型筛选: {chunk_type}，将在检索中应用")
            chunk_type_condition = {"terms": {"chunk_type.keyword": chunk_type}}
        else:
            print("未指定文档类型")
            chunk_type_condition = None

        # 添加文档名称关键词筛选
        if docnm_kwds and len(docnm_kwds) > 0:
            print(f"应用文档名称筛选: {docnm_kwds}")
            must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

        # 添加期刊过滤条件
        if journals:
            print(f"应用期刊筛选条件: {journals}")
            journal_filters = self._build_journal_filter(journals)
            if len(journal_filters) == 1:
                must_clauses.append(journal_filters[0])
            elif len(journal_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": journal_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加作者过滤条件
        if authors:
            print(f"应用作者筛选条件: {authors}")
            author_filters = self._build_author_filter(authors)
            if len(author_filters) == 1:
                must_clauses.append(author_filters[0])
            elif len(author_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": author_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加年份范围过滤条件
        if year_range and len(year_range) == 2:
            print(f"应用年份范围筛选条件: {year_range}")
            must_clauses.append({
                "range": {
                    "year": {
                        "gte": year_range[0],
                        "lte": year_range[1]
                    }
                }
            })

        # 添加文献语言
        if language:
            print(f"应用文献语言筛选: {language}")
            must_clauses.append({"term": {"language.keyword": language}})

        return must_clauses, chunk_type_condition

    def hybrid_search_v4(
            self,
            query_text: str,
            translated_text: Optional[str] = None,  # 添加翻译后文本参数
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            vector_weight: float = DEFAULT_VECTOR_WEIGHT,
            text_weight: float = DEFAULT_TEXT_WEIGHT,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = None,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], float]:
        """
        执行多阶段混合检索，采用分步骤筛选策略
        第一阶段：在kb_id_paper中检索，获取docnm_kwd和元数据，同时使用原始文本和翻译文本
        第二阶段：多线程处理，一个线程获取符合条件的非raw chunks，另一个线程混合检索获取每篇文献的raw chunks
        第三阶段：获取相邻chunks，将它们的内容合并到对应raw chunk的content_with_weight字段中
        第四阶段：处理结果并返回

        返回格式：
        {
            "chunks": [...],  # chunk数据
            "doc_aggs": [...]  # 元数据
        }
        """
        # 硬编码知识库ID

        if not self.es and not self.connect():
            return {"chunks": [], "doc_aggs": []}, 0

        search_start = time.time()
        print(f"[时间戳] 开始混合搜索: {time.strftime('%H:%M:%S', time.localtime())}")

        try:
            # ========== 预处理阶段 ==========
            # 处理原始查询文本
            # tokenized_query = self.tokenize_with_jieba(query_text)
            tokenized_query=query_text
            # 移除空字符串
            # tokenized_query = ' '.join([token for token in tokenized_query.split() if token.strip()])
            # print(f"原始查询: '{query_text}' -> 分词后: '{tokenized_query}'")

            # 处理翻译文本（如果提供）
            if translated_text:
                tokenized_translated=translated_text
            # tokenized_translated = ""
            # if translated_text:
            #     tokenized_translated = self.tokenize_with_jieba(translated_text)
            #     # 移除空字符串
            #     tokenized_translated = ' '.join([token for token in tokenized_translated.split() if token.strip()])
            #     print(f"翻译查询: '{translated_text}' -> 分词后: '{tokenized_translated}'")

            # ========== 处理过滤条件 ==========
            must_clauses, chunk_type_condition = self._build_filter_conditions(
                chunk_type=chunk_type,
                docnm_kwds=docnm_kwds,
                journals=journals,
                authors=authors,
                year_range=year_range,
                language=language
            )

            # ========== 第一阶段：在KB_ID_PAPER中文本检索，获取docnm_kwd和元数据 ==========
            # 使用filter而非must
            filter_clauses = must_clauses.copy()
            filter_clauses.append({"term": {"kb_id": KB_ID_PAPER}})

            # 构建查询条件，同时使用原始文本和翻译文本
            should_clauses = [
                {"match": {"text_ltks": {"query": tokenized_query, "boost": text_weight * 10}}}
            ]

            # 如果有翻译文本，添加翻译文本的查询条件
            if translated_text and tokenized_translated:
                should_clauses.extend([
                    {"match": {"text_ltks": {"query": tokenized_translated, "boost": text_weight * 10}}}
                ])

            # 定义元数据字段
            meta_fields = ["title", "abstract", "authors", "journal", "year", "vO", "issue",
                           "page_range", "pdf_url", "level", "subject", "impact_factor",
                           "reference", "docnm_kwd", "translated_abstract", "language"]

            sm_text_query = {
                "_source": meta_fields,  # 获取元数据字段
                "size": 10,  # 获取前10篇文献
                "query": {
                    "bool": {
                        "filter": filter_clauses,  # 使用filter代替must
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                }
            }

            sm_text_search_start = time.time()
            sm_text_results = self.es.search(index=index_name, body=sm_text_query, request_timeout=10)
            sm_text_hits = sm_text_results.get("hits", {}).get("hits", [])
            sm_text_search_end = time.time()

            # 提取docnm_kwd列表和保存元数据
            docnm_kwd_list = []
            doc_aggs = []

            for hit in sm_text_hits:
                source = hit.get("_source", {})
                docnm_kwd = source.get("docnm_kwd", "")
                if docnm_kwd:
                    docnm_kwd_list.append(docnm_kwd)

                    # 保存元数据
                    meta_data = {k: source.get(k, "") for k in meta_fields}
                    meta_data["id"] = hit["_id"]
                    doc_aggs.append(meta_data)

            # 去重
            docnm_kwd_list = list(set(docnm_kwd_list))

            print(
                f"第一阶段(kb_id_paper)文本检索获取候选文档数: {len(docnm_kwd_list)}, 耗时: {sm_text_search_end - sm_text_search_start:.2f}s")

            if not docnm_kwd_list:
                print("第一阶段检索未找到结果，结束搜索")
                return {"chunks": [], "doc_aggs": []}, time.time() - search_start

            # 定义内容字段
            content_fields = ["kb_id", "chunk_type", "content_with_weight", "docnm_kwd"]

            # ========== 第二阶段：多线程处理 ==========
            def extract_fields(source, field_list):
                return {k: source.get(k, "") for k in field_list}

            # 初始化结果容器
            chunks = []
            raw_chunks = []

            # 使用多线程进行两种不同的检索
            import threading
            from queue import Queue

            result_queue = Queue()

            # 线程1：获取符合chunk_type条件的非raw chunks
            def fetch_non_raw_chunks():
                non_raw_chunks_result = []
                try:
                    # 使用multi-search API查询每个docnm_kwd下的非raw chunks
                    msearch_body = []
                    for docnm_kwd in docnm_kwd_list:
                        # 添加空头部(使用默认索引)
                        msearch_body.append({})

                        # 构建文本查询部分
                        text_should_clauses = [
                            {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 5}}}
                        ]

                        # 如果有翻译文本，添加翻译文本的查询条件
                        if translated_text and tokenized_translated:
                            text_should_clauses.extend([
                                {"match": {"content_ltks": {"query": tokenized_translated, "boost": text_weight * 1}}}
                            ])

                        # 构建filter条件
                        filter_clauses = [
                            {"term": {"kb_id": KB_ID_CHUNK}},
                            {"term": {"docnm_kwd": docnm_kwd}}
                        ]

                        # 添加chunk_type条件
                        if chunk_type and len(chunk_type) > 0:
                            filter_clauses.append({"terms": {"chunk_type.keyword": chunk_type}})
                        else:
                            # 如果没有指定chunk_type，则排除raw类型
                            filter_clauses.append({"bool": {"must_not": [{"term": {"chunk_type.keyword": "raw"}}]}})

                        # 构建非raw查询
                        non_raw_query = {
                            "_source": content_fields + ["chunk_id"],
                            "size": 1,  # 每个docnm_kwd返回1个non-raw chunk
                            "min_score": 0.5,  # 添加相关性下限，0.5是示例值，根据你的需求调整
                            "query": {
                                "bool": {
                                    "filter": filter_clauses,
                                    "should": text_should_clauses,
                                    "minimum_should_match": 1  # 确保至少匹配一个should子句
                                }
                            }
                        }

                        msearch_body.append(non_raw_query)

                    # 执行msearch请求
                    msearch_results = self.es.msearch(body=msearch_body, index=index_name, request_timeout=10)
                    responses = msearch_results.get("responses", [])

                    # 处理每个响应
                    for i, response in enumerate(responses):
                        if i >= len(docnm_kwd_list):
                            break

                        non_raw_hits = response.get("hits", {}).get("hits", [])
                        if not non_raw_hits:
                            docnm_kwd = docnm_kwd_list[i]
                            print(f"未找到non-raw chunks (docnm_kwd={docnm_kwd})")
                            continue

                        # 处理找到的non-raw chunks
                        for hit in non_raw_hits:
                            source = hit.get("_source", {})
                            chunk_id = source.get("chunk_id")
                            content = extract_fields(source, content_fields)
                            content["id"] = hit["_id"]
                            content["chunk_id"] = chunk_id
                            non_raw_chunks_result.append(content)

                    print(f"获取非raw chunks数量: {len(non_raw_chunks_result)}")
                except Exception as e:
                    print(f"获取非raw chunks异常: {str(e)}")

                # 将结果放入队列
                result_queue.put(("non_raw", non_raw_chunks_result))

            # 线程2：混合检索获取raw chunks
            def fetch_raw_chunks():
                raw_chunks_result = []
                try:
                    # 使用multi-search API查询所有raw chunks，只采用文本检索
                    msearch_body = []
                    for docnm_kwd in docnm_kwd_list:
                        # 添加空头部(使用默认索引)
                        msearch_body.append({})

                        # 构建文本查询部分
                        text_should_clauses = [
                            {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 5}}}
                        ]

                        # 如果有翻译文本，添加翻译文本的查询条件
                        if translated_text and tokenized_translated:
                            text_should_clauses.extend([
                                {"match": {"content_ltks": {"query": tokenized_translated, "boost": text_weight * 5}}}
                            ])

                        # 构建纯文本查询 - 使用filter替代must
                        text_query = {
                            "_source": content_fields + ["chunk_id"],
                            "size": 1,  # 每个docnm_kwd返回1个raw chunk
                            "min_score": 0.5,  # 添加相关性下限，0.5是示例值，根据你的需求调整
                            "query": {
                                "bool": {
                                    "filter": [
                                        {"term": {"kb_id": KB_ID_CHUNK}},
                                        {"term": {"docnm_kwd": docnm_kwd}},
                                        {"term": {"chunk_type.keyword": "raw"}}
                                    ],
                                    "should": text_should_clauses,
                                    "minimum_should_match": 1
                                }
                            }
                        }

                        msearch_body.append(text_query)

                    # 执行msearch请求
                    msearch_results = self.es.msearch(body=msearch_body, index=index_name, request_timeout=10)
                    responses = msearch_results.get("responses", [])

                    # 处理每个响应
                    for i, response in enumerate(responses):
                        if i >= len(docnm_kwd_list):
                            break

                        raw_hits = response.get("hits", {}).get("hits", [])
                        if not raw_hits:
                            docnm_kwd = docnm_kwd_list[i]
                            print(f"未找到raw chunks (docnm_kwd={docnm_kwd})")
                            continue

                        # 处理找到的raw chunks
                        for raw_hit in raw_hits:
                            raw_source = raw_hit.get("_source", {})
                            chunk_id = raw_source.get("chunk_id")
                            raw_content = extract_fields(raw_source, content_fields)
                            raw_content["id"] = raw_hit["_id"]
                            raw_content["chunk_id"] = chunk_id
                            raw_chunks_result.append(raw_content)

                    print(f"获取raw chunks数量: {len(raw_chunks_result)}")
                except Exception as e:
                    print(f"获取raw chunks异常: {str(e)}")

                # 将结果放入队列
                result_queue.put(("raw", raw_chunks_result))

            # 创建并启动线程
            thread_non_raw = threading.Thread(target=fetch_non_raw_chunks)
            thread_raw = threading.Thread(target=fetch_raw_chunks)

            second_stage_start = time.time()

            thread_non_raw.start()
            thread_raw.start()

            # 等待两个线程完成
            thread_non_raw.join()
            thread_raw.join()

            # 从队列获取结果
            non_raw_chunks = []
            while not result_queue.empty():
                chunk_type, chunk_results = result_queue.get()
                if chunk_type == "non_raw":
                    non_raw_chunks = chunk_results
                else:  # raw
                    raw_chunks = chunk_results

            second_stage_end = time.time()
            print(f"第二阶段多线程检索耗时: {second_stage_end - second_stage_start:.2f}s")

            # ========== 第三阶段：获取相邻chunks并合并内容 ==========
            adjacent_search_start = time.time()

            # 按docnm_kwd分组整理raw chunks和它们的chunk_id
            raw_chunks_by_docnm = defaultdict(list)
            for raw_chunk in raw_chunks:
                if "chunk_id" not in raw_chunk or raw_chunk["chunk_id"] is None or "docnm_kwd" not in raw_chunk:
                    continue
                docnm_kwd = raw_chunk.get("docnm_kwd", "")
                if docnm_kwd:
                    raw_chunks_by_docnm[docnm_kwd].append(raw_chunk)

            # 记录需要查询的相邻chunk的ID
            adjacent_chunk_ids_to_fetch = defaultdict(set)  # 按docnm_kwd分组存储需要获取的chunk_id
            chunk_id_map = {}  # 存储原始chunk的映射关系

            # 首先识别所有原始chunks的ID并创建映射
            for docnm_kwd, chunks_in_doc in raw_chunks_by_docnm.items():
                # 收集这个文档中已有的chunk_id
                existing_chunk_ids = set()
                for chunk in chunks_in_doc:
                    try:
                        chunk_id = int(chunk.get("chunk_id"))
                        existing_chunk_ids.add(chunk_id)
                        # 将原始chunk保存到映射中
                        chunk_id_map[(chunk_id, docnm_kwd)] = chunk
                    except (ValueError, TypeError):
                        continue

                # 确定需要查询的相邻chunk_id
                for chunk_id in existing_chunk_ids:
                    prev_id = chunk_id - 1
                    next_id = chunk_id + 1

                    # 只获取还不在结果集中的chunk
                    if prev_id > 0 and prev_id not in existing_chunk_ids:
                        adjacent_chunk_ids_to_fetch[docnm_kwd].add(prev_id)

                    if next_id not in existing_chunk_ids:
                        adjacent_chunk_ids_to_fetch[docnm_kwd].add(next_id)

            # 使用msearch批量获取所有需要的相邻chunks
            adjacent_msearch_body = []
            for docnm_kwd, adj_ids in adjacent_chunk_ids_to_fetch.items():
                # 转为列表并分批处理
                adj_ids_list = list(adj_ids)
                for i in range(0, len(adj_ids_list), 50):  # 每批最多50个ID
                    batch_ids = adj_ids_list[i:i + 50]

                    # 添加空头部
                    adjacent_msearch_body.append({})

                    # 添加查询体
                    adjacent_query = {
                        "_source": content_fields + ["chunk_id"],
                        "size": len(batch_ids),
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"kb_id": KB_ID_CHUNK}},
                                    {"term": {"docnm_kwd": docnm_kwd}},
                                    {"terms": {"chunk_id": [str(id) for id in batch_ids]}}  # 确保ID是字符串格式
                                ]
                            }
                        }
                    }
                    adjacent_msearch_body.append(adjacent_query)

            # 存储查询到的相邻chunks
            adjacent_chunks_by_key = {}  # 以(chunk_id, docnm_kwd)为键

            # 执行msearch请求
            if adjacent_msearch_body:
                adjacent_msearch_results = self.es.msearch(
                    body=adjacent_msearch_body,
                    index=index_name,
                    request_timeout=10
                )
                adjacent_responses = adjacent_msearch_results.get("responses", [])

                # 处理响应
                for response in adjacent_responses:
                    adj_hits = response.get("hits", {}).get("hits", [])

                    for adj_hit in adj_hits:
                        adj_source = adj_hit.get("_source", {})
                        adj_chunk_id = adj_source.get("chunk_id")
                        adj_docnm_kwd = adj_source.get("docnm_kwd", "")

                        if adj_chunk_id is not None and adj_docnm_kwd:
                            try:
                                adj_chunk_id_int = int(adj_chunk_id)
                                adj_content = extract_fields(adj_source, content_fields)
                                adj_content["id"] = adj_hit["_id"]
                                adj_content["chunk_id"] = adj_chunk_id
                                adjacent_chunks_by_key[(adj_chunk_id_int, adj_docnm_kwd)] = adj_content
                            except (ValueError, TypeError):
                                continue

            # 现在为每个原始chunk合并前后相邻chunk的内容
            for (chunk_id, docnm_kwd), raw_chunk in chunk_id_map.items():
                original_content = raw_chunk.get("content_with_weight", "")
                merged_content = original_content

                # 添加前一个chunk的内容
                prev_id = chunk_id - 1
                prev_key = (prev_id, docnm_kwd)

                # 先检查是否在原始chunks中
                if prev_key in chunk_id_map:
                    prev_chunk = chunk_id_map[prev_key]
                    prev_content = prev_chunk.get("content_with_weight", "")
                    if prev_content:
                        merged_content = prev_content + "\n\n" + merged_content
                # 再检查是否在查询到的相邻chunks中
                elif prev_key in adjacent_chunks_by_key:
                    prev_chunk = adjacent_chunks_by_key[prev_key]
                    prev_content = prev_chunk.get("content_with_weight", "")
                    if prev_content:
                        merged_content = prev_content + "\n\n" + merged_content

                # 添加后一个chunk的内容
                next_id = chunk_id + 1
                next_key = (next_id, docnm_kwd)

                # 先检查是否在原始chunks中
                if next_key in chunk_id_map:
                    next_chunk = chunk_id_map[next_key]
                    next_content = next_chunk.get("content_with_weight", "")
                    if next_content:
                        merged_content = merged_content + "\n\n" + next_content
                # 再检查是否在查询到的相邻chunks中
                elif next_key in adjacent_chunks_by_key:
                    next_chunk = adjacent_chunks_by_key[next_key]
                    next_content = next_chunk.get("content_with_weight", "")
                    if next_content:
                        merged_content = merged_content + "\n\n" + next_content

                # 更新原始chunk的内容
                raw_chunk["content_with_weight"] = merged_content

            # 添加一个时间统计
            adjacent_search_end = time.time()
            print(f"相邻chunk合并耗时: {adjacent_search_end - adjacent_search_start:.4f}秒")


            # 将处理后的非raw和raw chunks合并到结果列表
            chunks = non_raw_chunks + raw_chunks

            adjacent_search_end = time.time()
            print(f"获取相邻chunks并合并内容耗时: {adjacent_search_end - adjacent_search_start:.2f}s")

            # ========== 第四阶段：处理结果并返回 ==========
            elapsed_time = time.time() - search_start
            print(f"检索完成: chunks数: {len(chunks)}, doc_aggs数: {len(doc_aggs)}, 总耗时: {elapsed_time:.2f}s")

            # ========== 对结果进行排序和过滤 ==========
            # 过滤掉content_with_weight中数字过多或包含"作者简介"的chunk
            filtered_chunks = []
            for chunk in chunks:
                content = chunk.get("content_with_weight", "")

                # 检查是否包含"作者简介"
                if "作者简介" in content:
                    continue

                # 统计内容中的数字字符比例
                digit_count = sum(1 for c in content if c.isdigit())
                total_count = len(content) if content else 1  # 避免除零错误
                digit_ratio = digit_count / total_count

                # 如果数字比例小于30%，则保留此chunk
                if digit_ratio < 0.3:
                    filtered_chunks.append(chunk)

            # 按照docnm_kwd和chunk_id排序chunks
            def sort_key_for_chunks(chunk):
                docnm_kwd = chunk.get("docnm_kwd", "")
                chunk_id_str = chunk.get("chunk_id", "0")
                try:
                    chunk_id = int(chunk_id_str)
                except (ValueError, TypeError):
                    chunk_id = 0
                return (docnm_kwd, chunk_id)

            filtered_chunks.sort(key=sort_key_for_chunks)

            # 获取过滤后保留的docnm_kwd集合 - 确保这一步骤被正确执行
            remaining_docnm_kwds = set()
            for chunk in filtered_chunks:
                docnm_kwd = chunk.get("docnm_kwd", "")
                if docnm_kwd:
                    remaining_docnm_kwds.add(docnm_kwd)
            print(f"过滤后保留的docnm_kwd: {remaining_docnm_kwds}")

            # 显式打印doc_aggs中的docnm_kwd以便调试
            print(f"原始doc_aggs中的docnm_kwd: {[doc.get('docnm_kwd', '') for doc in doc_aggs]}")

            # 根据保留的docnm_kwd过滤doc_aggs - 使用更明确的方式
            filtered_doc_aggs = []
            for doc in doc_aggs:
                doc_kwd = doc.get("docnm_kwd", "")
                if doc_kwd in remaining_docnm_kwds:
                    filtered_doc_aggs.append(doc)
                else:
                    print(f"移除doc_agg: {doc_kwd} (不在保留的docnm_kwd中)")

            print(f"过滤后doc_aggs数量: {len(filtered_doc_aggs)}")

            # 按照docnm_kwd排序doc_aggs
            filtered_doc_aggs.sort(key=lambda doc: doc.get("docnm_kwd", ""))

            # 按照top_k限制返回结果
            result = {
                "chunks": filtered_chunks,
                "doc_aggs": filtered_doc_aggs
            }
            return result


        except Exception as e:
            print(f"搜索异常: {str(e)}")
            traceback.print_exc()
            return {"chunks": [], "doc_aggs": []}