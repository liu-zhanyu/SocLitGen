import anthropic
import oss2
from openai import OpenAI
from pymongo import MongoClient
from es_search import ElasticsearchService

# MongoDB 连接参数
MONGO_HOST = "43.134.113.96"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_PASSWORD = "mongo321654987"
MONGO_DB = "database_database"
MONGO_COLLECTION = "Data"

# Claude API设置
CLAUDE_API_KEY = "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"

# OpenAI API设置
OPENAI_API_KEY = "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"  # 请替换为你的API密钥
# 配置OSS连接
ENDPOINT = "oss-cn-qingdao.aliyuncs.com"
BUCKET_NAME = "hentre-user-upload"
OSS_ACCESS_KEY_ID = "LTAI5tEcTV9KxipuT6awaE86"
OSS_ACCESS_KEY_SECRET = "4w7OH3GKQnTtEEOqhDyXbEt6oIJmgM"


# 初始化 Claude 客户端
claude_client = anthropic.Client(api_key=CLAUDE_API_KEY)

# 初始化 OpenAI 客户端
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 初始化 MongoDB 连接
try:
    mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
    # 测试连接
    mongo_client.server_info()
    print(f"成功连接到MongoDB")
except Exception as e:
    print(f"连接MongoDB时出错: {e}")
    mongo_client = None

# 初始化 OSS 连接
try:
    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    oss_bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)
    print(f"成功初始化OSS Bucket连接")
except Exception as e:
    print(f"初始化OSS Bucket时出错: {e}")
    oss_bucket = None

es_service = ElasticsearchService()
es_service.connect()