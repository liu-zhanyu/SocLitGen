from pymongo import MongoClient
from components.es_search import ElasticsearchService

# MongoDB 连接参数
MONGO_HOST = "43.134.113.96"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_PASSWORD = "mongo321654987"

# 初始化 MongoDB 连接
try:
    mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
    # 测试连接
    mongo_client.server_info()
    print(f"成功连接到MongoDB")
except Exception as e:
    print(f"连接MongoDB时出错: {e}")
    mongo_client = None

es_service = ElasticsearchService()
es_service.connect()