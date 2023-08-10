import json
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel


class ErrorEntity(BaseModel):
    timestamp: datetime
    status: int
    error: str
    message: str
    path: str


class BaseRequest(BaseModel):
    context_id: str
    chat_id: str
    user_id: str
    session_id: str
    role: str  # user | assistant
    seq: int
    content: Any  # Request Entity will come here
    chat_history: List


class BaseResponse(BaseModel):
    success: Optional[bool]
    error: Optional[ErrorEntity]  # if success is false
    content: Optional[Any]  # Response Entity will come here


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
