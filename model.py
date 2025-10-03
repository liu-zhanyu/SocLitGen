from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict,Union


class ClassifyRequest(BaseModel):
    """意图识别请求模型"""
    user_id: str
    query: str


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索查询文本")
    top_k: int = Field(30, description="返回结果数量")
    docnm_kwds: Optional[List[str]] = Field(None, description="文献名称关键词列表，用于筛选文献范围")


class ReviewRequest(BaseModel):
    """文献综述生成请求模型"""
    query: str = Field(..., description="研究主题")
    user_id: str = Field(..., description="用户ID")
    language:Optional[str] =Field(None, description="文献语言")
    mode: Optional[int]=Field(None, description="模式")
    structure:Optional[int]=Field(0,description="综述结构")
    chinese_weight:Optional[float]=Field(None, description="中文文献比例")


class HypothesisRequest(BaseModel):
    """文献综述生成请求模型"""
    hypothesis: str = Field(..., description="研究假设")
    user_id: str = Field(..., description="用户ID")


class StatusRequest(BaseModel):
    """任务状态查询请求模型"""
    user_id: str = Field(..., description="用户ID")
    task_id: str = Field(..., description="任务ID")
    task_type: Optional[str] = Field(None,
                                     description="任务类型，可选值为'review'或'research_data'，如不提供则根据任务ID推断")


class SearchResultItem(BaseModel):
    """搜索结果项模型"""
    id: str
    score: float
    title: str = ""
    abstract: str = ""
    authors: str = ""
    journal: str = ""
    year: Any = None
    doc_id: str = ""
    kb_id: str = ""
    chunk_type: str = ""
    content_with_weight: str = ""


class SearchResponse(BaseModel):
    """搜索响应模型"""
    results: List[SearchResultItem]
    count: int


class ReviewTaskStatus(BaseModel):
    """文献综述任务状态响应模型"""
    status: str
    status_code: int
    query: Optional[str] = None
    update_time: Optional[Any] = None
    task_type: str = "review"
    error: Optional[str] = None
    review_text: Optional[str] = None
    research_data: Optional[str] = None
    meta_data: Optional[List[dict]] = None
    message: Optional[str] = None


class ReviewTaskSubmission(BaseModel):
    """任务提交响应模型"""
    task_id: str
    user_id: str
    status: str
    submit_time: str
    task_type: str = "review"


# 请求模型
class ChatWithDataRequest(BaseModel):
    user_id: str
    query: str
    data_json: str  # DataFrame转换成的JSON字符串
    format_type: str = "records"  # 默认为records格式
    metadata: List[Dict[str, Any]]  # 元数据列表


class PayRequest(BaseModel):
    user_id: str
    body: str
    out_trade_no: str
    total_fee: int


class FollowupRequest(BaseModel):
    """请求数据模型"""
    user_id: str
    dialogue: dict  # 格式: {"user": "内容", "assistant": "内容"}


class IntroductionRequest(BaseModel):
    user_id: str
    query:str

class FullPaperRequest(BaseModel):
    user_id: str
    query:str
    mode:int

class FullPaperRequestV2(BaseModel):
    user_id: str
    query:str

# 请求模型
class DialogueRequest(BaseModel):
    user_id: str
    dialogue: dict  # 包含"user"和"assistant"的字典

class QueryRewriteRequest(BaseModel):
    user_id: str
    history: List[Dict]
    query: str


# 数据模型
class EventData(BaseModel):
    """单个事件的数据模型"""
    id: str = Field(..., description="事件唯一ID")
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    category: str = Field(..., description="事件类别")
    action: str = Field(..., description="事件动作")
    timestamp: str = Field(..., description="事件发生时间")
    url: str = Field(..., description="页面URL")
    page: str = Field(..., description="页面名称")
    data: Dict[str, Any] = Field(default_factory={}, description="事件附加数据")

class EventsRequest(BaseModel):
    """事件请求的数据模型"""
    events: List[EventData] = Field(..., description="事件列表")

class AnalyticsResponse(BaseModel):
    """API响应模型"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(default="", description="响应消息")
    data: Optional[Dict[str, Any]] = Field(default=None, description="响应数据")

class AbstractsSummarizeRequest(BaseModel):
    """摘要总结请求模型"""
    user_id: str
    abstracts: List[Dict]

# 请求模型
class MetricsAnalysisRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    metadata: list = Field(..., description="指标元数据，格式为[{...}]")


class QuestionnaireRequest(BaseModel):
    user_id: str
    demo_list: List[Dict]
    questionnaire: List[Dict]
    sample_size: int
    country_of_origin: Optional[str] = None
    timestamp: Optional[str] = None
    subgroup: Optional[str] = None
    experimental_group: Optional[list] = None

class DownloadRequest(BaseModel):
    user_id: str
    pdf_urls: List[str]

class SearchDocumentsRequest(BaseModel):
    query: str  # 唯一必填参数
    journals: Optional[Union[str, List[str]]] = None
    authors: Optional[Union[str, List[str]]] = None
    language: Optional[str] = None
    year_range: Optional[List[int]] = None
    kb_id: Optional[str] = "3dcd9e360c6811f081000242ac120004"
    top_k: Optional[int] = None

class SearchLiteratureRequest(BaseModel):
    query: str  # 唯一必填参数
    journals: List[str] = None
    authors: List[str] = None
    language: Optional[str] = None
    year_range: Optional[List[int]] = None
    levels:Optional[List[str]] = None
    page_size: int=10
    page: int=1

class AIGCProcessRequest(BaseModel):
    user_id: str
    task_id: str
    task_type: Optional[str] = "review"

class ChatRequest(BaseModel):
    provider: str
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    system_prompt: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None