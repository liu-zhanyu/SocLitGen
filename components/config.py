from pymongo import MongoClient
from dotenv import load_dotenv
import os
from elasticsearch import Elasticsearch
# 加载环境变量
load_dotenv()

# MongoDB 连接参数
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
DB= os.getenv("DB")
COLLECTION= os.getenv("COLLECTION")
# Elasticsearch 连接参数
ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
DEFAULT_INDEX=os.getenv("DEFAULT_INDEX")

# Knowledge Base IDs
KB_ID_SUMMARY = os.getenv("KB_ID_SUMMARY")

DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K")
DEFAULT_VECTOR_WEIGHT = os.getenv("DEFAULT_VECTOR_WEIGHT")
DEFAULT_TEXT_WEIGHT = os.getenv("DEFAULT_TEXT_WEIGHT")
DEFAULT_CHUNK_TYPE = os.getenv("DEFAULT_CHUNK_TYPE")

BGE_API_URL = os.getenv("BGE_API_URL")
BGE_API_KEY=[
    os.getenv("BGE_API_TOKEN_1"),
    os.getenv("BGE_API_TOKEN_2")
]
