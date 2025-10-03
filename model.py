from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict,Union


class ReviewRequest(BaseModel):
    """文献综述生成请求模型"""
    query: str = Field(..., description="研究主题")
    user_id: str = Field(..., description="用户ID")
    language:Optional[str] =Field(None, description="文献语言")
    mode: Optional[int]=Field(None, description="模式")
    structure:Optional[int]=Field(0,description="综述结构")
    chinese_weight:Optional[float]=Field(None, description="中文文献比例")


class StatusRequest(BaseModel):
    """任务状态查询请求模型"""
    user_id: str = Field(..., description="用户ID")
    task_id: str = Field(..., description="任务ID")
    task_type: Optional[str] = Field(None,
                                     description="任务类型，可选值为'review'或'research_data'，如不提供则根据任务ID推断")


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

