from fastapi import APIRouter

from app.dtos.chats import (
    CreateOrUpdateChatRequest,
    CreateOrUpdateChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    QuesByContextRequest,
    QuesByContextResponse,
    ResetRequest,
    TopicRequest,
    TopicResponse,
)
from app.dtos.question import QuestionRequest, QuestionResponse
from app.services import chat_service

router = APIRouter()


@router.post("/initiation", response_model=CreateOrUpdateChatResponse)
def initiation_chat(request: CreateOrUpdateChatRequest):
    user_id = request.user_id
    data = chat_service.check_existing_user(user_id)
    return data


@router.post("/context", response_model=TopicResponse)
def new_topic(request: TopicRequest):
    user_id = request.user_id
    chat_id = request.chat_id
    result = TopicResponse()
    result.context_id = chat_service.create_new_context(chat_id, user_id)
    return result


@router.post("/question", response_model=QuestionResponse)
def question(request: QuestionRequest):
    chat_id = request.chat_id
    user_id = request.user_id
    context_id = request.context_id
    question = request.question

    return chat_service.create_new_question(chat_id, context_id, user_id, question)


@router.post("/reset", response_model=CreateOrUpdateChatResponse)
def reset_chat(request: ResetRequest):
    user_id = request.user_id
    return chat_service.reset_chat(user_id)


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    answer_id = request.answer_id
    status = request.status
    reason = request.reason
    return chat_service.feedback(answer_id, status, reason)


@router.post("/get_questions_by_context_id", response_model=QuesByContextResponse)
def get_questions_by_context_id(request: QuesByContextRequest):
    context_id = request.context_id  # get questions by context_id
    return chat_service.get_questions_by_context_id(context_id)
