from datetime import datetime
from typing import Any, List, Optional

# import pandas as pd
from pydantic import BaseModel

from app.dtos.base import BaseRequest, BaseResponse


class ContentComponent(BaseModel):
    category_type: Optional[str]
    content: Any
    error: Optional[str] or None
    showError: Optional[bool]


# class QuestionRequest(BaseRequest):
class QuestionRequest(BaseModel):
    chat_id: int
    user_id: str
    context_id: int
    question: str
    # additional_context: Optional[str]
    # partial_response: Optional[List[ContentComponent]]


# class QuestionResponse(BaseResponse):
class QuestionResponse(BaseModel):
    error: Optional[str]
    user_id: Optional[str]
    question_id: Optional[int]
    answer_id: Optional[int]
    category: Optional[str]
    # created_time: Optional[datetime]
    content: Optional[Any]
    # data: Optional[List[ContentComponent]]

    # class Config:
    #     orm_mode = True
    #     allow_population_by_field_name = True


class ChartRequest(BaseRequest):  # Can inherit QuestionRequest
    question: str
    additional_context: Optional[str]
    sql_query: str
    # output_table: Optional[pd.DataFrame]
    output_table_dict: Optional[dict]
