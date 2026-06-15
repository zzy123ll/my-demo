"""用户反馈收集端点。"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from evaluation.feedback import FeedbackCollector

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])
_collector = FeedbackCollector()


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(ge=1, le=5)
    session_id: str = ""
    user_id: str = ""
    comment: str = ""
    question: str = ""
    answer: str = ""


@router.post("")
async def submit_feedback(req: FeedbackRequest):
    record = _collector.submit(
        message_id=req.message_id,
        rating=req.rating,
        session_id=req.session_id,
        user_id=req.user_id,
        comment=req.comment,
        question=req.question,
        answer=req.answer,
    )
    return {"status": "ok", "feedback_id": record.feedback_id}


@router.get("/stats")
async def feedback_stats():
    return _collector.get_stats()
