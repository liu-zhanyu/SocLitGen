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
from llm_call import *
from pre_search_service import classify_query_type
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
    return {"message": "Newbit服务已启动，请访问 /docs 查看API文档"}


@app.post("/api/classify_query", summary="分类用户查询类型")
def classify_query_api(request: ClassifyRequest):
    """
    使用大模型判断用户查询类型
    Args:
        request: 包含用户ID和查询内容的请求

    Returns:
        分类结果（0: QA型，1: 数据型，2: 混合型）
    """
    user_id = request.user_id
    query = request.query

    logger.info(f"API - 用户 {user_id} 发起查询分类，查询内容: {query}")

    try:
        result = classify_query_type(query)
        logger.info(f"API - 用户 {user_id} 查询分类结果: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"API - 用户 {user_id} 查询分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询分类失败: {e}")


# 结合路由搜索相关文档API
@app.post("/api/search_with_router_v3", summary="结合路由搜索相关文档V3")
def search_with_router_v3(request: SearchRequest):
    """
    结合路由搜索相关文档API

    Args:
        request: 包含查询文本和期望结果数量的请求

    Returns:
        搜索结果列表
    """
    # data = search_documents_simple(query=request.query, top_k=request.top_k,docnm_kwds=request.docnm_kwds)
    results,extracted_info = search_with_extracted_params_v3(
        es_service=es_service,
        user_query=request.query,
        top_k=request.top_k,
        additional_filters={"kb_id": "7750e714049611f08aa20242ac120003", "docnm_kwds": request.docnm_kwds}
    )
    print("路由参数：", extracted_info)

    for doc_agg in results["doc_aggs"]:
        doc_agg["pdf_url"] = doc_agg["pdf_url"].replace("http://hentre-admin-upload.oss-cn-qingdao.aliyuncs.com/",
                                                        "https://oss.gtpa.cloud/")
    return {"results": results, "count": len(results["chunks"])}


# 生成追问问题API
@app.post("/api/generate_followup", summary="生成追问问题")
def generate_followup(request: FollowupRequest):
    """
    结合路由搜索相关文档API

    Args:
        dialogue_dict: {
            "user": "合并后的所有用户发言内容",
            "assistant": "合并后的所有助手回复内容"
        }

    Returns:
        包含追问的字符串列表，如:
        ["追问内容1?", "追问内容2?"]
    """
    user_id = request.user_id
    dialogue = request.dialogue

    # 输入验证
    if not all(key in dialogue for key in ["user", "assistant"]):
        logger.warning(f"API - 用户 {user_id} 提交无效对话格式")
        raise HTTPException(
            status_code=400,
            detail="对话必须包含user和assistant字段"
        )

    logger.info(f"API - 用户 {user_id} 发起追问生成，对话长度: "
                f"用户{len(dialogue['user'])}字，助手{len(dialogue['assistant'])}字")

    try:

        questions = generate_followup_questions(dialogue)

        logger.info(f"API - 用户 {user_id} 生成追问成功: {questions}")
        return {
            "status": "success",
            "questions": questions,
        }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "questions": [],
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} 追问生成异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "questions": [],
        }


@app.post("/api/search/documents",
             summary="智能搜索文献",
             description="结合向量搜索和文本搜索，支持多种过滤条件的高级文档搜索接口")
def search_documents_api(request: SearchDocumentsRequest):
    """
    修改后的高级文档搜索API，返回格式与search_with_router保持一致

    Args:
        request: 包含搜索查询和各种过滤条件的请求对象

    Returns:
        {
            "results": {
                "chunks": [{"content_with_weight", "chunk_type", "kb_id", "score"}],
                "doc_aggs": [完整文档信息]
            },
            "count": 结果数量
        }
    """
    # 调用原始搜索函数
    raw_results = es_service.search_documents(
        query=request.query,
        top_k=request.top_k,
        year_range=request.year_range,
        kb_id=request.kb_id,
        journals=request.journals,
        authors=request.authors,
        language=request.language,
        chunk_type=["summary"]
    )

    # 按照search_with_router的格式处理结果
    results = {
        "chunks": [{
            "content_with_weight": item.get("content_with_weight", item.get("content", "")),
            "chunk_type": item.get("chunk_type", ""),
            "kb_id": item.get("kb_id", ""),
            "score": item.get("score", 0.0)
        } for item in raw_results],

        "doc_aggs": list({
                             item.get("docnm_kwd", item.get("title", "")): {
                                 "title": item.get("title", ""),
                                 "abstract": item.get("abstract", ""),
                                 "authors": item.get("authors", ""),
                                 "journal": item.get("journal", ""),
                                 "year": item.get("year", ""),
                                 "vO": item.get("vO", ""),
                                 "issue": item.get("issue", ""),
                                 "page_range": item.get("page_range", ""),
                                 "doc_id": item.get("doc_id", ""),
                                 "kb_id": item.get("kb_id", ""),
                                 "chunk_type": item.get("chunk_type", ""),
                                 "content_with_weight": item.get("content_with_weight", ""),
                                 "pdf_url": item.get("pdf_url", ""),
                                 "level": item.get("level", ""),
                                 "subject": item.get("subject", ""),
                                 "impact_factor": item.get("impact_factor", ""),
                                 "reference": item.get("reference", ""),
                                 "docnm_kwd": item.get("docnm_kwd", ""),
                                 "translated_abstract": item.get("translated_abstract", ""),
                                 "language": item.get("language", "")
                             } for item in raw_results
                         }.values())
    }

    return {
        "results": results,
        "count": len(results["chunks"])
    }


@app.post("/api/search_literature",summary="搜索文献",description="文献检索搜索接口")
def search_literature_api(request: SearchLiteratureRequest):
    """
    修改后的高级文档搜索API，返回格式与search_with_router保持一致

    Args:
        request: 包含搜索查询和各种过滤条件的请求对象

    Returns:
        {
            "results": {
                "chunks": [{"content_with_weight", "chunk_type", "kb_id", "score"}],
                "doc_aggs": [完整文档信息]
            },
            "count": 结果数量
        }
    """
    # 调用原始搜索函数
    raw_results = es_service.search_literature(
        query=request.query,
        year_range=request.year_range,
        journals=request.journals,
        authors=request.authors,
        language=request.language,
        levels=request.levels,
        page_size=int(request.page_size),
        page=int(request.page)
    )

    return raw_results

# 查询改写API端点
@app.post("/api/rewrite_query", summary="重写并消除查询歧义")
def rewrite_query_api(request: QueryRewriteRequest):
    """
    使用大模型对用户查询进行改写和消除歧义，使其更适合检索使用

    Args:
        user_id: 用户ID
        history: 历史对话记录，格式为[{"content": "文本内容", "role": "user/assistant"}]
        query: 当前用户查询内容

    Returns:
        包含原始查询和改写后查询的字典:
        {
            "status": "success/fail",
            "original_query": "原始查询",
            "rewritten_query": "改写后的查询"
        }
    """
    user_id = request.user_id
    history = request.history
    query = request.query

    logger.info(f"API - 用户 {user_id} 发起查询改写，原始查询: '{query}'，历史消息数: {len(history)}")

    try:
        # 调用改写函数
        rewritten_query = rewrite_and_disambiguate_query(history, query)

        if rewritten_query != query:
            logger.info(f"API - 用户 {user_id} 查询改写成功: '{rewritten_query}'")
            return {
                "status": "success",
                "original_query": query,
                "rewritten_query": rewritten_query
            }
        else:
            logger.info(f"API - 用户 {user_id} 查询未改写，返回原查询")
            return {
                "status": "success",
                "original_query": query,
                "rewritten_query": query,
                "message": "查询无需改写或改写失败，返回原始查询"
            }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "original_query": query,
            "rewritten_query": query,
            "message": f"输入验证失败: {str(e)}"
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} 查询改写异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "original_query": query,
            "rewritten_query": query,
            "message": f"查询改写失败: {str(e)}"
        }


@app.post("/api/summarize_abstracts", summary="总结多篇研究摘要")
def summarize_abstracts_api(request: AbstractsSummarizeRequest):
    """
    使用大模型对多篇研究摘要进行总结，突出核心观点并比较不同的研究视角

    Args:
        user_id: 用户ID
        abstracts: 研究摘要列表，格式为[{"author": "作者", "year": "年份", "abstract": "摘要内容"}]

    Returns:
        包含原始摘要和总结结果的字典:
        {
            "status": "success/fail",
            "abstracts_count": 摘要数量,
            "summary": "总结内容",
            "message": "可选的消息"
        }
    """
    user_id = request.user_id
    abstracts = request.abstracts

    logger.info(f"API - 用户 {user_id} 发起摘要总结请求，摘要数量: {len(abstracts)}")

    try:
        # 输入验证
        if not abstracts or not isinstance(abstracts, list):
            raise ValueError("摘要列表不能为空且必须是数组格式")

        for item in abstracts:
            if not isinstance(item, dict):
                raise ValueError("摘要项必须是字典格式")
            if "abstract" not in item or not item["abstract"]:
                raise ValueError("摘要内容不能为空")
            if "authors" not in item or not item["authors"]:
                raise ValueError("作者信息不能为空")
            if "year" not in item or not item["year"]:
                raise ValueError("年份信息不能为空")

        # 调用总结函数
        summary = summarize_abstracts(abstracts)

        if summary:
            logger.info(f"API - 用户 {user_id} 摘要总结成功，总结长度: {len(summary)}")
            return {
                "status": "success",
                "abstracts_count": len(abstracts),
                "summary": summary
            }
        else:
            logger.warning(f"API - 用户 {user_id} 摘要总结结果为空")
            return {
                "status": "success",
                "abstracts_count": len(abstracts),
                "summary": "",
                "message": "总结生成为空，请检查输入的摘要内容"
            }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "abstracts_count": len(abstracts) if isinstance(abstracts, list) else 0,
            "summary": "",
            "message": f"输入验证失败: {str(e)}"
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} 摘要总结异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "abstracts_count": len(abstracts) if isinstance(abstracts, list) else 0,
            "summary": "",
            "message": f"摘要总结失败: {str(e)}"
        }



@app.post("/api/generate_title", summary="生成对话标题")
def generate_title_api(request: DialogueRequest):
    """
    基于用户和助手的对话内容生成标题

    Args:
        request: 包含用户ID和对话内容的请求

    Returns:
        生成的对话标题
    """
    user_id = request.user_id
    dialogue = request.dialogue

    logger.info(f"API - 用户 {user_id} 发起标题生成请求")

    try:
        # 验证对话格式
        if not isinstance(dialogue, dict) or "user" not in dialogue or "assistant" not in dialogue:
            logger.error(f"API - 用户 {user_id} 请求中的对话格式不正确")
            raise HTTPException(status_code=400, detail="对话格式不正确，必须包含'user'和'assistant'字段")

        # 调用标题生成函数
        title = generate_dialogue_title(dialogue)
        logger.info(f"API - 用户 {user_id} 标题生成成功: {title}")

        return {
            "status": "success",
            "user_id": user_id,
            "title": title
        }
    except Exception as e:
        logger.error(f"API - 用户 {user_id} 标题生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"标题生成失败: {str(e)}")

# 搜索文档API
@app.post("/api/search", summary="搜索相关文档")
def api_search_documents(request: SearchRequest):
    """
    搜索文档API

    Args:
        request: 包含查询文本和期望结果数量的请求

    Returns:
        搜索结果列表
    """
    data = search_documents_simple(query=request.query, top_k=request.top_k, docnm_kwds=request.docnm_kwds)

    results = {
        "chunks": [{k: v for k, v in item.items() if k in ['content_with_weight', 'chunk_type', 'kb_id', 'score']} for
                   item in data],
        "doc_aggs": list({item.get("docnm_kwd", item["title"]): {k: v for k, v in item.items() if
                                                                 k not in ['content_with_weight', 'chunk_type', 'kb_id',
                                                                           'score']} for item in
                          data}.values())
    }
    return {"results": results, "count": len(results["chunks"])}


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


# 处理AIGC API
@app.post("/api/process_aigc", summary="处理文献综述AIGC率")
def api_process_aigc(request: AIGCProcessRequest, background_tasks: BackgroundTasks):
    """
    异步处理文献综述AIGC率降低

    Args:
        request: 包含用户ID和任务ID的请求
        background_tasks: FastAPI的后台任务管理器

    Returns:
        任务状态信息
    """
    logger.info(f"API - 用户 {request.user_id} 发起AIGC率降低处理，任务ID：{request.task_id}")

    # 添加处理任务到后台
    background_tasks.add_task(
        process_aigc,
        user_id=request.user_id,
        task_id=request.task_id,
        task_type=request.task_type if hasattr(request, 'task_type') else "review"
    )

    return {
        "task_id": request.task_id,
        "user_id": request.user_id,
        "status": "AIGC处理已提交",
        "submit_time": datetime.now().isoformat()
    }

# 生成研究假设API
@app.post("/api/generate_hypothesis", summary="生成研究假设")
def api_generate_hypothesis(request: HypothesisRequest, background_tasks: BackgroundTasks):
    """
    异步生成研究假设

    Args:
        request: 包含研究假设和用户ID的请求
        background_tasks: FastAPI的后台任务管理器

    Returns:
        任务ID和状态信息
    """
    logger.info(f"API - 用户 {request.user_id} 发起研究假设生成，输入为：{request.hypothesis}")
    # 生成任务ID
    task_id = f"hypothesis_{uuid.uuid4()}"

    # 添加生成任务到后台
    background_tasks.add_task(
        async_generate_hypothesis,
        hypothesis=request.hypothesis,
        user_id=request.user_id,
        task_id=task_id
    )

    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat()
    }

# 生成引言API
@app.post("/api/generate_introduction", summary="生成引言")
def api_generate_introduction(request: IntroductionRequest,background_tasks: BackgroundTasks):
    """
    生成完整研究引言
    参数:
    - user_id: 用户ID
    - query：选题

    返回:
    - 即时生成的完整引言内容
    """
    logger.info(f"API - 用户 {request.user_id} 发起引言生成，输入为：{request.query}")
    # 自动生成task_id
    task_id = f"intro_{uuid.uuid4()}"

    # 添加生成任务到后台
    background_tasks.add_task(
        async_generate_introduction,
        research_topic=request.query,
        user_id=request.user_id,
        task_id=task_id
    )

    # 直接返回生成的完整内容
    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat()
    }


# 生成全文API
@app.post("/api/generate_full_paper", summary="生成全文")
def api_generate_full_paper(request: FullPaperRequest, background_tasks: BackgroundTasks):
    """
    生成全文
    参数:
    - user_id: 用户ID
    - query：选题

    返回:
    - 即时生成的完整引言内容
    """
    logger.info(f"API - 用户 {request.user_id} 发起全文生成，输入为：{request.query}")
    # 自动生成task_id
    task_id = f"full_paper_{uuid.uuid4()}"

    # 添加生成任务到后台
    background_tasks.add_task(
        async_generate_full_paper,
        topic=request.query,
        user_id=request.user_id,
        task_id=task_id,
        mode=request.mode
    )

    # 直接返回生成的完整内容
    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat()
    }


# 生成全文API
@app.post("/api/generate_full_paper_v2", summary="生成全文")
def api_generate_full_paper_v2(request: FullPaperRequestV2, background_tasks: BackgroundTasks):
    """
    生成全文
    参数:
    - user_id: 用户ID
    - query：选题

    返回:
    - 即时生成的完整引言内容
    """
    logger.info(f"API - 用户 {request.user_id} 发起全文生成，输入为：{request.query}")
    # 自动生成task_id
    task_id = f"papers_{uuid.uuid4()}"

    # 添加生成任务到后台
    background_tasks.add_task(
        async_generate_full_paper_v2,
        topic=request.query,
        user_id=request.user_id,
        task_id=task_id
    )

    # 直接返回生成的完整内容
    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat()
    }


# 生成问卷数据API
@app.post("/api/process_questionnaire", summary="生成问卷数据")
def api_process_questionnaire(request: QuestionnaireRequest, background_tasks: BackgroundTasks):
    """
    异步处理问卷数据

    Args:
        request: 包含问卷信息和用户ID的请求
        background_tasks: FastAPI的后台任务管理器

    Returns:
        任务ID和状态信息
    """
    logger.info(f"API - 用户 {request.user_id} 发起问卷数据生成请求，样本量: {request.sample_size}")

    # 生成任务ID
    task_id = f"questionnaire_data_{uuid.uuid4()}"

    # 添加处理任务到后台
    background_tasks.add_task(
        process_questionnaire_data,
        demo_list=request.demo_list,
        questionnaire=request.questionnaire,
        sample_size=request.sample_size,
        user_id=request.user_id,
        task_id=task_id,
        country_of_origin=request.country_of_origin,
        timestamp=request.timestamp,
        subgroup=request.subgroup,
        experimental_group=request.experimental_group
    )

    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat()
    }
# 生成研究数据API
@app.post("/api/generate_research_data", summary="查找研究数据")
def api_generate_research_data(request: ReviewRequest, background_tasks: BackgroundTasks):
    """
    异步生成研究数据

    Args:
        request: 包含查询文本和用户ID的请求
        background_tasks: FastAPI的后台任务管理器

    Returns:
        任务ID和状态信息
    """
    logger.info(f"API - 用户 {request.user_id} 发起找数据，输入为：{request.query}")

    # 生成任务ID
    task_id = f"data_{uuid.uuid4()}"

    # 添加生成任务到后台
    background_tasks.add_task(
        async_generate_research_data,
        query=request.query,
        user_id=request.user_id,
        task_id=task_id
    )

    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat(),
        "task_type": "research_data"
    }


# 设置文件大小限制（25MB）
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes


@app.post("/api/risk_diagnose")
async def risk_diagnose_api(
        file: UploadFile = File(...),
        user_id: str = Form(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    上传文件进行学术风险诊断

    参数:
    - file: 要诊断的文件（PDF、DOCX）
    - user_id: 用户ID

    返回:
    - 任务状态信息
    """
    # 检查文件
    if not file:
        raise HTTPException(status_code=400, detail="没有提供文件")

    # 获取文件名和扩展名
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    # 验证文件类型
    allowed_extensions = ['.pdf', '.docx', '.doc']
    file_ext = os.path.splitext(filename)[1].lower()

    if not file_ext:
        raise HTTPException(status_code=400, detail="文件没有扩展名")

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。只接受以下格式: {', '.join(allowed_extensions)}"
        )

    # 生成唯一任务ID
    task_id = f"risk_{uuid.uuid4()}"

    # 创建带扩展名的临时文件
    temp_file_path = os.path.join(tempfile.gettempdir(), f"{task_id}{file_ext}")

    # 读取上传文件内容
    content = await file.read()

    # 检查文件大小
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="上传的文件内容为空")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过25MB限制: {len(content)} 字节, 最大允许大小: {MAX_FILE_SIZE} 字节"
        )

    # 写入临时文件
    with open(temp_file_path, "wb") as f:
        f.write(content)

    # 确认文件已成功写入
    if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
        raise HTTPException(status_code=500, detail="文件保存失败，临时文件不存在或为空")

    # 记录任务信息
    logger.info(
        f"API - 用户 {user_id} 发起学术风险诊断请求，文件: {filename}，临时文件: {temp_file_path}，任务ID: {task_id}")

    # 在后台处理诊断任务
    background_tasks.add_task(
        async_risk_diagnose_service,
        file_path=temp_file_path,
        user_id=user_id,
        task_id=task_id
    )

    # 返回任务信息
    return {
        "status_code": 0,
        "message": "任务已提交，请稍后查询结果",
        "task_id": task_id,
        "task_type": "risk_diagnose",
        "user_id": user_id,
        "file_name": filename,
        "file_type": file_ext[1:]  # 移除扩展名前的点
    }


@app.post("/api/info_extraction")
async def info_extraction_api(
        entity_name: str = Form(...),
        field_descriptions: str = Form(...),  # JSON字符串格式
        files: Optional[List[UploadFile]] = File(None),
        urls: Optional[List[str]] = Form(None),
        input_text: Optional[str] = Form(None),
        user_id: str = Form(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    信息提取API - 支持多个文件、URL和文本输入

    参数:
    - entity_name: 要提取信息的实体名称
    - field_descriptions: 字段描述（JSON字符串格式）
    - files: 要处理的文件列表（PDF、DOCX、DOC、TXT）
    - urls: URL列表
    - input_text: 直接输入的文本内容
    - user_id: 用户ID

    返回:
    - 任务状态信息
    """

    # 验证至少提供一种输入源
    if not files and not urls and not input_text:
        raise HTTPException(
            status_code=400,
            detail="必须提供至少一种输入源：文件、URL或文本"
        )

    # 解析field_descriptions
    try:
        field_descriptions_parsed = json.loads(field_descriptions)

        # 验证数据结构
        if not isinstance(field_descriptions_parsed, list):
            raise ValueError("field_descriptions必须是数组格式")

        for item in field_descriptions_parsed:
            if not isinstance(item, dict):
                raise ValueError("field_descriptions中的每个项目必须是对象")

            required_keys = ['info_name', 'info_des', 'field_list']
            if not all(key in item for key in required_keys):
                raise ValueError(f"field_descriptions中的对象必须包含: {required_keys}")

            if not isinstance(item['field_list'], list):
                raise ValueError("field_list必须是数组格式")

            for field in item['field_list']:
                if not isinstance(field, dict) or not all(key in field for key in ['field_name', 'field_des']):
                    raise ValueError("field_list中的每个字段必须包含field_name和field_des")

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="field_descriptions必须是有效的JSON格式"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"field_descriptions格式错误: {str(e)}"
        )

    # 处理URLs
    urls_list = urls if urls else []

    # 验证文件类型和大小
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
    temp_file_paths = []

    if files:
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="存在文件名为空的文件")

            # 验证文件类型
            file_ext = os.path.splitext(file.filename)[1].lower()
            if not file_ext:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {file.filename} 没有扩展名"
                )

            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型: {file_ext}。只接受以下格式: {', '.join(allowed_extensions)}"
                )

    # 生成唯一任务ID
    task_id = f"info_extract_{uuid.uuid4()}"

    # 处理上传的文件
    if files:
        for file in files:
            # 读取文件内容
            content = await file.read()

            # 检查文件大小
            if len(content) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {file.filename} 内容为空"
                )

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件 {file.filename} 大小超过25MB限制"
                )

            # 创建临时文件
            file_ext = os.path.splitext(file.filename)[1].lower()
            temp_file_path = os.path.join(
                tempfile.gettempdir(),
                f"{task_id}_{file.filename}"
            )

            # 写入临时文件
            with open(temp_file_path, "wb") as f:
                f.write(content)

            # 确认文件已成功写入
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"文件 {file.filename} 保存失败"
                )

            temp_file_paths.append(temp_file_path)

    # 构建sources列表
    sources = []

    # 添加文件路径
    sources.extend(temp_file_paths)

    # 添加URLs
    sources.extend(urls_list)

    # 添加文本输入
    if input_text:
        sources.append({
            "text": input_text,
            "source": "manual_input"
        })

    # 记录任务信息
    logger.info(
        f"API - 用户 {user_id} 发起信息提取请求，实体: {entity_name}，"
        f"文件数: {len(temp_file_paths)}，URL数: {len(urls_list)}，"
        f"文本输入: {'是' if input_text else '否'}，任务ID: {task_id}"
    )

    # 在后台处理信息提取任务
    background_tasks.add_task(
        async_info_extraction_service,
        entity_name=entity_name,
        field_descriptions=field_descriptions_parsed,
        sources=sources,
        user_id=user_id,
        task_id=task_id
    )

    # 返回任务信息
    return {
        "status_code": 0,
        "message": "信息提取任务已提交，请稍后查询结果",
        "task_id": task_id,
        "task_type": "info_extraction",
        "user_id": user_id,
        "entity_name": entity_name,
        "sources_count": {
            "files": len(temp_file_paths),
            "urls": len(urls_list),
            "text_input": 1 if input_text else 0
        },
        "total_sources": len(sources)
    }


@app.post("/api/quote_match")
async def quote_match_api(
       file: UploadFile = File(...),
       user_id: str = Form(...),
       background_tasks: BackgroundTasks = BackgroundTasks()
):
   """
   上传文件进行语录匹配

   参数:
   - file: 要匹配的文件（PDF、DOCX、MD）
   - user_id: 用户ID

   返回:
   - 任务状态信息
   """
   # 检查文件
   if not file:
       raise HTTPException(status_code=400, detail="没有提供文件")

   # 获取文件名和扩展名
   filename = file.filename
   if not filename:
       raise HTTPException(status_code=400, detail="文件名为空")

   # 验证文件类型
   allowed_extensions = ['.pdf', '.docx', '.doc', '.md']
   file_ext = os.path.splitext(filename)[1].lower()

   if not file_ext:
       raise HTTPException(status_code=400, detail="文件没有扩展名")

   if file_ext not in allowed_extensions:
       raise HTTPException(
           status_code=400,
           detail=f"不支持的文件类型: {file_ext}。只接受以下格式: {', '.join(allowed_extensions)}"
       )

   # 生成唯一任务ID
   task_id = f"quote_{uuid.uuid4()}"

   # 创建带扩展名的临时文件
   temp_file_path = os.path.join(tempfile.gettempdir(), f"{task_id}{file_ext}")

   # 读取上传文件内容
   content = await file.read()

   # 检查文件大小
   if len(content) == 0:
       raise HTTPException(status_code=400, detail="上传的文件内容为空")

   if len(content) > MAX_FILE_SIZE:
       raise HTTPException(
           status_code=400,
           detail=f"文件大小超过25MB限制: {len(content)} 字节, 最大允许大小: {MAX_FILE_SIZE} 字节"
       )

   # 写入临时文件
   with open(temp_file_path, "wb") as f:
       f.write(content)

   # 确认文件已成功写入
   if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
       raise HTTPException(status_code=500, detail="文件保存失败，临时文件不存在或为空")

   # 记录任务信息
   logger.info(
       f"API - 用户 {user_id} 发起语录匹配请求，文件: {filename}，临时文件: {temp_file_path}，任务ID: {task_id}")

   # 在后台处理语录匹配任务
   background_tasks.add_task(
       async_quote_match_service,
       file_path=temp_file_path,
       user_id=user_id,
       task_id=task_id
   )

   # 返回任务信息
   return {
       "status_code": 0,
       "message": "任务已提交，请稍后查询结果",
       "task_id": task_id,
       "task_type": "quote_match",
       "user_id": user_id,
       "file_name": filename,
       "file_type": file_ext[1:]  # 移除扩展名前的点
   }

@app.post("/api/chat_with_data", summary="基于数据回答用户查询")
def chat_with_data_api(request: ChatWithDataRequest):
    """
    使用用户提供的数据回答查询问题

    Args:
        request: 包含用户ID、查询内容、数据JSON和元数据的请求

    Returns:
        基于数据分析的回答结果
    """
    user_id = request.user_id
    query = request.query
    data_json = request.data_json
    format_type = request.format_type
    metadata = request.metadata

    logger.info(f"API - 用户 {user_id} 发起数据对话，查询内容: {query}")

    try:
        # 将JSON字符串转换回DataFrame
        if format_type == "records":
            # 字典列表格式
            df = pd.DataFrame(json.loads(data_json))
        elif format_type == "split":
            # 拆分格式
            df = pd.read_json(data_json, orient='split')
        elif format_type == "csv":
            # CSV字符串格式
            df = pd.read_csv(io.StringIO(data_json))
        else:
            # 默认使用orient参数
            df = pd.read_json(data_json, orient=format_type)

        logger.info(f"API - 成功将JSON数据转换为DataFrame，形状: {df.shape}")

        # 调用你的函数处理查询
        result = chat_with_data(query, df, metadata)

        logger.info(f"API - 用户 {user_id} 数据对话成功完成")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"API - 用户 {user_id} 数据对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据对话处理失败: {str(e)}")


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


# 支付相关API
APP_ID = "wx9f172f9adf518b65"
MCH_ID = "1714570600"
API_KEY = "B6oyan04qiyao14wEnXi9antONG08ai1"
NOTIFY_URL = "https://pay.bitflows.cn/notifyUrl"


@app.post("/api/pay/create")
async def create_wechat_order(data: PayRequest):
    nonce_str = uuid.uuid4().hex[:32]
    params = {
        "appid": APP_ID,
        "mch_id": MCH_ID,
        "nonce_str": nonce_str,
        "body": data.body,
        "out_trade_no": data.out_trade_no,
        "total_fee": str(data.total_fee),
        "spbill_create_ip": "43.134.113.96",
        "notify_url": NOTIFY_URL,
        "trade_type": "NATIVE",
    }

    sign = generate_sign(params, API_KEY)
    params["sign"] = sign
    xml = dict_to_xml(params)

    await create_order_record(data.out_trade_no, data.user_id, data.total_fee)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url="https://api.mch.weixin.qq.com/pay/unifiedorder",
            content=xml.encode("utf-8"),
            headers={"Content-Type": "application/xml"}
        )

    resp_data = xml_to_dict(response.text)
    if resp_data.get("return_code") == "SUCCESS" and resp_data.get("result_code") == "SUCCESS":
        return {
            "code_url": resp_data.get("code_url"),
            "order_id": data.out_trade_no
        }
    else:
        return {
            "error": resp_data.get("return_msg", "统一下单失败"),
            "detail": resp_data
        }


@app.post("/notifyUrl")
async def wechat_notify(request: Request):
    client = AsyncIOMotorClient(
        f"mongodb://admin:{MONGO_PASSWORD}@{MONGO_HOST}:27017",
    )
    db = client.get_database("Newbit")
    orders_collection = db["orders"]
    raw_body = await request.body()
    xml_data = raw_body.decode("utf-8")
    data = xml_to_dict(xml_data)
    print("微信回调原始数据：", data)

    if not data:
        return Response(content=notify_response("FAIL", "参数解析失败"), media_type="application/xml")

    # 校验签名
    sign = data.pop("sign", None)
    expected_sign = generate_sign(data, API_KEY)
    if sign != expected_sign:
        print("签名验证失败")
        return Response(content=notify_response("FAIL", "签名错误"), media_type="application/xml")

    print("签名验证通过")

    if data.get("return_code") == "SUCCESS" and data.get("result_code") == "SUCCESS":
        out_trade_no = data.get("out_trade_no")
        transaction_id = data.get("transaction_id")
        print("开始更新订单：", out_trade_no)

        res = await orders_collection.update_one(
            {"order_id": out_trade_no},
            {"$set": {
                "status": "SUCCESS",
                "transaction_id": transaction_id,
                "paid_at": datetime.utcnow()
            }},
        )
        print("更新成功数量：", res.modified_count)
        return Response(content=notify_response("SUCCESS", "OK"), media_type="application/xml")
    else:
        print("支付状态异常")
        return Response(content=notify_response("FAIL", "支付失败"), media_type="application/xml")


def notify_response(code: str, msg: str) -> str:
    return f"""
    <xml>
      <return_code><![CDATA[{code}]]></return_code>
      <return_msg><![CDATA[{msg}]]></return_msg>
    </xml>
    """.strip()


@app.get("/api/pay/status")
async def get_status(order_id: str = Query(..., alias="orderId")):
    return await get_order_status(order_id)




# API endpoints
@app.post("/api/analytics/events", response_model=AnalyticsResponse)
async def capture_events(events_request: EventsRequest, request: Request):
    """
    接收前端发送的事件数据
    """
    start_time = time.time()

    try:
        # 在API中创建MongoDB连接
        client = AsyncIOMotorClient(
        f"mongodb://admin:{MONGO_PASSWORD}@{MONGO_HOST}:27017",
    )
        db = client.get_database('Newbit')
        events_collection = db["events"]
        users_collection = db["users"]
        sessions_collection = db["sessions"]

        # 提取请求元数据
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        processed_count = 0

        # 处理每个事件
        for event in events_request.events:
            # 解析ISO日期字符串为Python datetime对象
            try:
                timestamp = parser.parse(event.timestamp)
            except ValueError:
                timestamp = datetime.utcnow()

            # 构建事件文档
            event_doc = {
                "event_id": event.id,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "category": event.category,
                "action": event.action,
                "timestamp": timestamp,
                "url": event.url,
                "page": event.page,
                "data": event.data,
                "metadata": {
                    "ip": client_ip,
                    "user_agent": user_agent,
                    "received_at": datetime.utcnow()
                }
            }

            # 插入事件文档到events集合
            await events_collection.insert_one(event_doc)
            processed_count += 1

            # 更新用户信息到users集合
            await update_user_info(users_collection, event, timestamp)

            # 更新会话信息到sessions集合
            await update_session_info(sessions_collection, event, timestamp)

        processing_time = time.time() - start_time
        logger.info(f"Processed {processed_count} events in {processing_time:.2f}s")

        return AnalyticsResponse(
            success=True,
            message=f"Successfully processed {processed_count} events",
            data={"processed_count": processed_count}
        )

    except Exception as e:
        logger.error(f"Error processing events: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error processing events: {str(e)}"}
        )


async def update_user_info(users_collection, event: EventData, timestamp: datetime):
    """更新用户信息"""
    # 更新用户文档
    await users_collection.update_one(
        {"user_id": event.user_id},
        {
            "$setOnInsert": {
                "user_id": event.user_id,
                "first_seen": timestamp,
                "created_at": datetime.utcnow()
            },
            "$set": {
                "last_activity": timestamp
            },
            "$inc": {
                "event_count": 1
            }
        },
        upsert=True
    )

    # 如果是页面访问事件，更新页面访问计数
    if event.category == "page" and event.action == "page_view":
        await users_collection.update_one(
            {"user_id": event.user_id},
            {"$inc": {"page_view_count": 1}}
        )

    # 如果是点击事件，更新点击计数
    if event.category == "click":
        await users_collection.update_one(
            {"user_id": event.user_id},
            {"$inc": {"click_count": 1}}
        )


async def update_session_info(sessions_collection, event: EventData, timestamp: datetime):
    """更新会话信息"""
    # 更新会话文档
    session_update = {
        "$setOnInsert": {
            "session_id": event.session_id,
            "user_id": event.user_id,
            "start_time": timestamp,
            "user_agent": "unknown"  # 这应该从请求头中获取，此处简化
        },
        "$set": {
            "last_activity": timestamp
        },
        "$inc": {
            "event_count": 1
        }
    }

    # 如果是页面离开事件，记录停留时间
    if event.category == "page" and event.action == "page_leave" and "duration" in event.data:
        duration = event.data.get("duration", 0)

        # 更新总停留时间
        session_update["$inc"]["total_duration"] = duration

        # 记录页面停留时间
        page_name = event.data.get("page", "unknown")
        stay_time_field = f"page_durations.{page_name}"
        session_update["$inc"][stay_time_field] = duration

    # 如果是页面访问事件，更新页面访问计数
    if event.category == "page" and event.action == "page_view":
        session_update["$inc"]["page_view_count"] = 1

    # 如果是会话结束事件，设置会话结束时间和总持续时间
    if event.category == "session" and event.action == "session_end":
        session_duration = event.data.get("duration", 0)
        session_update["$set"]["end_time"] = timestamp
        session_update["$set"]["duration"] = session_duration

    await sessions_collection.update_one(
        {"session_id": event.session_id},
        session_update,
        upsert=True
    )


@app.get("/api/analytics/stats", response_model=AnalyticsResponse)
async def get_stats():
    """获取基本统计数据"""
    try:
        # 在API中创建MongoDB连接
        client = AsyncIOMotorClient(
            f"mongodb://admin:{MONGO_PASSWORD}@{MONGO_HOST}:27017",
        )
        db = client.get_database('Newbit')
        events_collection = db["events"]
        users_collection = db["users"]
        sessions_collection = db["sessions"]

        # 获取基本计数
        event_count = await events_collection.count_documents({})
        user_count = await users_collection.count_documents({})
        session_count = await sessions_collection.count_documents({})

        # 获取页面访问统计
        page_views = await events_collection.aggregate([
            {"$match": {"category": "page", "action": "page_view"}},
            {"$group": {"_id": "$page", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]).to_list(10)

        # 获取按钮点击统计
        button_clicks = await events_collection.aggregate([
            {"$match": {"category": "click"}},
            {"$group": {"_id": "$action", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]).to_list(10)

        # 获取平均页面停留时间
        avg_durations = await sessions_collection.aggregate([
            {"$match": {"page_durations": {"$exists": True}}},
            {"$project": {"durations": {"$objectToArray": "$page_durations"}}},
            {"$unwind": "$durations"},
            {"$group": {"_id": "$durations.k", "avg_duration": {"$avg": "$durations.v"}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]).to_list(10)

        return AnalyticsResponse(
            success=True,
            data={
                "event_count": event_count,
                "user_count": user_count,
                "session_count": session_count,
                "page_views": page_views,
                "button_clicks": button_clicks,
                "avg_durations": avg_durations
            }
        )
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching statistics: {str(e)}"}
        )



@app.post("/api/summarize_data", summary="总结AI相关指标")
def summarize_data_api(request: MetricsAnalysisRequest):
    """
    分析AI相关指标数据，生成关于各指标所代表变量的总结报告

    Args:
        user_id: 用户ID
        metadata: 指标元数据，格式为{'meta data': [{变量信息字典},...]}

    Returns:
        包含原始元数据和总结结果的字典:
        {
            "status": "success/fail",
            "indicators_count": 指标数量,
            "summary": "总结内容",
            "message": "可选的消息"
        }
    """
    user_id = request.user_id
    metadata = request.metadata

    logger.info(f"API - 用户 {user_id} 发起AI指标总结请求，指标数量: {len(metadata)}")

    try:
        # 调用总结函数
        summary = summarize_data(metadata)

        if summary:
            logger.info(f"API - 用户 {user_id} AI指标总结成功，总结长度: {len(summary)}")
            return {
                "status": "success",
                "indicators_count": len(metadata),
                "summary": summary
            }
        else:
            logger.warning(f"API - 用户 {user_id} AI指标总结结果为空")
            return {
                "status": "success",
                "indicators_count": len(metadata),
                "summary": "",
                "message": "总结生成为空，请检查输入的指标数据"
            }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "indicators_count": len(metadata) if isinstance(metadata,list) else 0,
            "summary": "",
            "message": f"输入验证失败: {str(e)}"
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} AI指标总结异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "indicators_count": len(metadata) if isinstance(metadata,list) else 0,
            "summary": "",
            "message": f"AI指标总结失败: {str(e)}"
        }

@app.post("/api/batch_download", summary="批量下载PDF")
async def batch_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """
    批量下载PDF文件，打包为ZIP并返回下载链接

    Args:
        request: 包含用户ID和PDF链接列表的请求

    Returns:
        任务ID和状态信息
    """
    user_id = request.user_id
    pdf_urls = request.pdf_urls

    # 检查PDF链接列表是否为空
    if not pdf_urls:
        raise HTTPException(status_code=400, detail="PDF链接列表不能为空")

    # 生成任务ID
    task_id = f"download_{uuid.uuid4()}"

    logger.info(f"API - 用户 {user_id} 发起批量下载请求，共 {len(pdf_urls)} 个文件，任务ID: {task_id}")

    # 在后台处理下载任务
    background_tasks.add_task(async_process_download, pdf_urls, user_id, task_id)

    # 返回任务ID和初始状态
    return {
        "status_code": 0,
        "message": "任务已提交，请稍后查询结果",
        "task_id": task_id,
        "task_type": "batch_download"
    }

# 对话接口
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Unified endpoint for chat interactions with multiple LLM providers
    """
    if request.stream:
        # Streaming response
        generator = handler.call_llm(
            provider=request.provider,
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
            system_prompt=request.system_prompt,
            history=request.history
        )

        # Return a StreamingResponse
        return StreamingResponse(
            generator,
            media_type="text/plain"
        )
    else:
        # Standard response
        result = handler.call_llm(
            provider=request.provider,
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
            system_prompt=request.system_prompt,
            history=request.history
        )

        return {"result": result}

# 主函数
def main():
    """启动FastAPI服务器"""
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()