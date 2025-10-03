from components.config import *
import logging

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

# 查询任务状态服务函数
def get_task_state_service(user_id, task_id, task_type="review"):
    """
    查询任务的状态和结果，支持不同类型的任务

    Args:
        user_id: 用户ID
        task_id: 任务ID
        task_type: 任务类型，默认为"review"，可选值包括"review"、"research_data"、"hypothesis"、"introduction"、"ai_data"和"batch_download"

    Returns:
        dict: 包含任务状态和结果信息的字典
    """
    logger.info(f"查询任务状态: 用户 {user_id}, 任务 {task_id}, 类型 {task_type}")

    try:
        # 直接创建MongoDB连接
        mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}")
        db = mongo_client["Newbit"]

        # 根据任务类型选择不同的集合
        if task_type == "research_data":
            collection = db["research_data"]
        elif task_type == "hypothesis":
            collection = db["hypothesis"]
        elif task_type == "introduction":
            collection = db["introduction"]
        elif task_type == "ai_data":
            collection = db["ai_data"]
            # ai_data类型额外排除questionnaire和results字段
            projection = {"_id": 0, "user_id": 0, "task_id": 0, "questionnaire": 0, "results": 0}
        elif task_type == "full_paper":
            collection = db["full_paper"]
        elif task_type == "papers":
            collection = db["papers"]
        elif task_type == "batch_download":
            collection = db["download_tasks"]
        elif task_type=="risk_diagnose":
            collection = db["risk_diagnose"]
        elif task_type == "quote_match":
            collection = db["quote_match"]
        elif task_type == "info_extraction":
            collection = db["info_extraction"]
        else:  # 默认为 "review"
            collection = db["review"]

        # 设置默认投影字段（除ai_data外其他任务类型使用）
        if task_type != "ai_data":
            projection = {"_id": 0, "user_id": 0, "task_id": 0}

        # 查询任务
        task = collection.find_one(
            {"user_id": user_id, "task_id": task_id},
            projection
        )

        if task:
            # 统一的状态映射
            status_map = {0: "失败", 1: "成功", 2: "进行中"}
            status_code = task.get("state", -1)

            # 根据任务类型获取查询字段名称
            if task_type == "hypothesis":
                query_field = "hypothesis"
            elif task_type=="papers":
                query_field = "paper_title"
            else:
                query_field = "query"

            # 构建基本返回结果
            result = {
                "status": status_map.get(status_code, "未知"),
                "status_code": status_code,
                "query": task.get(query_field, ""),
                "update_time": task.get("update_time", ""),
                "task_type": task_type
            }

            # 根据状态添加不同信息
            if status_code == 0:  # 失败
                result["error"] = task.get("error", "")
                result["message"] = task.get("message", "")
            else:  # 成功
                if task_type == "research_data":
                    # 如果数据存在且不为空
                    if "research_data" in task and task["research_data"] != "{}":
                        result["research_data"] = task.get("research_data", "{}")
                        result["meta_data"] = task.get("meta_data", [])
                    else:
                        result["message"] = task.get("message", "未找到相关研究数据")
                        result["meta_data"] = task.get("meta_data", [])
                elif task_type == "hypothesis":
                    if "cot" in task:
                        result["cot"] = task.get("cot", "")
                    if "draft" in task:
                        result["draft"] = task.get("draft", "")
                    result["main_text"] = task.get("main_text", "")
                    result["references"] = task.get("references", [])
                    result["hypothesis_text"] = task.get("hypothesis_text", "")
                    result["pdf_urls"] = task.get("pdf_urls", [])
                elif task_type == "introduction":
                    result["query"] = task.get("research_topic", "")
                    if "paragraph1" in task:
                        result["paragraph1"] = task.get("paragraph1", "")
                    if "paragraph2" in task:
                        result["paragraph2"] = task.get("paragraph2", "")
                    if "paragraph3" in task:
                        result["paragraph3"] = task.get("paragraph3", "")
                    result["paragraph2"] = task.get("paragraph2", "")
                    result["references2"] = task.get("references2", [])
                    result["references3"] = task.get("references3", [])
                    result["pdf_urls2"] = task.get("pdf_urls2", [])
                    result["pdf_urls3"] = task.get("pdf_urls3", [])
                    result["main_text"] = task.get("main_text", "")
                    result["complete_text"]=task.get("complete_text","")
                elif task_type == "ai_data":
                    result["excel_url"] = task.get("excel_url", "")
                    result["markdown_table"] = task.get("markdown_table", "")
                    result["successful_count"] = task.get("successful_count", [])
                elif task_type == "full_paper":
                    result["hypotheses"] = task.get("hypotheses", {})
                    result["introduction"] = task.get("introduction", {})
                    result["literature_review"] = task.get("literature_review", {})
                    result["content"] = task.get("content", [])
                    result["pdf_urls"] = task.get("pdf_urls", [])
                elif task_type == "papers":
                    result["full_paper_text"] = task.get("full_paper_text", {})
                    result["all_references"] = task.get("all_references", {})
                    result["all_pdf_urls"] = task.get("all_pdf_urls", {})
                elif task_type == "batch_download":
                    # 批量下载任务只需要这两个关键字段
                    result["download_url"] = task.get("download_url", "")
                    result["total_files"] = task.get("total_files", 0)
                    result["message"] = task.get("message", "")
                elif task_type == "risk_diagnose":
                    # 批量下载任务只需要这两个关键字段
                    result["errors"] = task.get("errors", "")
                    result["error_paragraphs"] = task.get("error_paragraphs", 0)
                elif task_type == "quote_match":
                    # 批量下载任务只需要这两个关键字段
                    result["results"] = task.get("results", "")
                    result["matched_quotes"] = task.get("matched_quotes", 0)
                elif task_type == "info_extraction":
                    # 批量下载任务只需要这两个关键字段
                    result["result"] = task.get("result", "")
                else:  # review
                    result["review_text"] = task.get("review_text", {})
                    if "processed_review_text" in task:
                        result["processed_review_text"] = task.get("processed_review_text", "")

            return result
        else:
            return {"status": "未找到", "status_code": -1, "task_type": task_type}

    except Exception as e:
        logger.error(f"查询任务状态出错: 用户 {user_id}, 任务 {task_id}, 类型 {task_type}, 错误: {str(e)}")
        return {"status": "查询出错", "error": str(e), "status_code": -2, "task_type": task_type}

