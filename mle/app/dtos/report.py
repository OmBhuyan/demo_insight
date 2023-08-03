from typing import Any, List, Optional, Union

from pydantic import BaseModel


class GenerateReportRequest(BaseModel):
    user_id: Union[int, str]
    chat_id: int


class GenerateReportResponse(BaseModel):
    chat_id: Optional[int]
    report_id: Optional[int]
    report_name: Optional[str]
    user_id: Optional[str]
    report_url: Optional[str]
    ques_list: Optional[List[int]]
    chat_history: Optional[Any]
    created_time: Optional[str]
