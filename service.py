import logging
from agent.literature_review_system import *
from components.es_search import *

from datetime import datetime
from components.check_state import *
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
        review_system = LiteratureReviewSystem(
            user_id=user_id,
            task_id=task_id,
            kb_id=kb_id,
            mode=mode,
            structure=structure
        )

        review_system.generate_review(
            main_topic=query,
            language=language,
            chinese_weight=chinese_weight
        )


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
