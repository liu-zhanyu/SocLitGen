from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query, Response
import uvicorn
from datetime import datetime
import uuid
from service import *
from model import *
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import io
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorClient
from dateutil import parser
from fastapi.responses import StreamingResponse
from components.llm_call import *
import tempfile

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

app = FastAPI(title="Newbit服务", description="Newbit产品API接口")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或者改成指定域名 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # 允许 GET、POST、OPTIONS 等请求方法
    allow_headers=["*"],  # 允许所有请求头
)


# 添加一个根路径处理函数
@app.get("/")
def read_root():
    logger.info("访问根路径")
    return {"message": "SocLitGen服务已启动，请访问 /docs 查看API文档"}


# 生成文献综述API
@app.post("/api/generate_review", summary="生成文献综述")
def api_generate_review(request: ReviewRequest, background_tasks: BackgroundTasks):
    """
    异步生成文献综述

    Args:
        request: 包含查询文本和用户ID的请求
        background_tasks: FastAPI的后台任务管理器

    Returns:
        任务ID和状态信息
    """
    if request.language:
        request.language=request.language
    else:
        request.language=""
    logger.info(f"API - 用户 {request.user_id} 发起{request.language}文献综述生成，输入为：{request.query}")
    # 生成任务ID
    task_id = f"review_{uuid.uuid4()}"

    # 添加生成任务到后台
    background_tasks.add_task(
        async_generate_literature_review,
        query=request.query,
        user_id=request.user_id,
        task_id=task_id,
        language=request.language,
        mode=request.mode,
        structure=request.structure,
        chinese_weight=request.chinese_weight
    )

    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat()
    }


# 查询任务状态API
@app.post("/api/get_task_state", summary="查询任务状态")
def get_task_state_api(request: StatusRequest):
    """
    查询任务的状态和结果

    Args:
        request: 包含用户ID、任务ID和任务类型的请求

    Returns:
        任务状态和结果信息
    """
    user_id = request.user_id
    task_id = request.task_id

    # 从任务ID推断任务类型
    if task_id.startswith("data_"):
        task_type = "research_data"
    elif task_id.startswith("review_"):
        task_type = "review"
    elif task_id.startswith("hypothesis_"):
        task_type = "hypothesis"
    elif task_id.startswith("questionnaire_data_"):
        task_type = "ai_data"
    elif task_id.startswith("download_"):
        task_type = "batch_download"
    else:
        task_type = "unknown"

    # 如果请求中明确指定了任务类型，则使用请求中的类型
    if hasattr(request, 'task_type') and request.task_type:
        task_type = request.task_type

    logger.info(f"API - 用户 {user_id} 查询任务状态, 任务ID: {task_id}, 任务类型: {task_type}")

    # 调用服务函数获取任务状态
    result = get_task_state_service(user_id, task_id, task_type)

    # 处理可能的错误
    if result["status_code"] == -1:
        logger.warning(f"API - 用户 {user_id} 请求的任务未找到, 任务ID: {task_id}, 任务类型: {task_type}")
        raise HTTPException(status_code=404, detail="任务未找到")

    if result["status_code"] == -2:
        logger.error(f"API - 查询出错: {result.get('error', '')}")
        raise HTTPException(status_code=500, detail=f"查询出错: {result.get('error', '')}")

    # 记录状态信息
    logger.info(f"API - 返回任务状态: {result['status']}, 用户: {user_id}, 任务ID: {task_id}, 任务类型: {task_type}")

    return result


# 主函数
def main():
    """启动FastAPI服务器"""
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()