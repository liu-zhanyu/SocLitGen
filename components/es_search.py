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

# 默认搜索参数
DEFAULT_INDEX = "ragflow_b7a5a24ce9ba11efa0ca0242ac120006"  # 替换为您的索引名称
# DEFAULT_INDEX = "ragflow_optimized_vectors"  # 替换为您的索引名称
DEFAULT_TOP_K = 30
DEFAULT_VECTOR_WEIGHT = 0.7  # 向量搜索的权重
DEFAULT_TEXT_WEIGHT = 0.3  # 文本搜索的权重
DEFAULT_CHUNK_TYPE = ["raw"]  # 默认的chunk_type值


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
