from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class GetDetailsResponse(BaseModel):
    chat_id: Optional[int]
    context_id: Optional[int]
    chat_name: Optional[str]
    is_active: Optional[bool]
    chat_history: Optional[Dict]


class CreateOrUpdateChatRequest(BaseModel):
    user_id: Union[int, str]


class CreateOrUpdateChatResponse(BaseModel):
    user_id: Optional[Union[int, str]]
    report_list: Optional[List]
    chat_details: GetDetailsResponse


class ResetRequest(BaseModel):
    user_id: Union[int, str]
    chat_id: int


class TopicRequest(BaseModel):
    user_id: Optional[str]
    chat_id: Optional[int]


class TopicResponse(BaseModel):
    context_id: Optional[int]


class FeedbackRequest(BaseModel):
    answer_id: int
    status: bool
    reason: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback: bool


class QuesByContextRequest(BaseModel):
    context_id: int


class QuesByContextResponse(BaseModel):
    ques_list: list
