from fastapi import APIRouter

from app.dtos.report import GenerateReportRequest, GenerateReportResponse
from app.services import report_service

router = APIRouter()


@router.post("/generate", response_model=GenerateReportResponse)
def generate_report(request: GenerateReportRequest):
    user_id = request.user_id
    chat_id = request.chat_id
    return report_service.generate_report(user_id, chat_id)
