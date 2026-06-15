"""转接管理端点。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/escalate", tags=["escalate"])
_handler = None


def init_global_handler():
    global _handler
    from escalation_handler import EscalationHandler
    _handler = EscalationHandler()


class EscalateRequest(BaseModel):
    session_id: str
    user_id: str
    user_message: str
    user_name: str = ""
    department: str = ""


class ResolveRequest(BaseModel):
    ticket_id: str
    resolution: str
    solution_tags: list[str] = Field(default_factory=list)


@router.post("")
async def escalate(req: EscalateRequest):
    result = _handler.escalate(session_id=req.session_id, user_id=req.user_id,
        user_message=req.user_message, user_name=req.user_name, department=req.department)
    return {"escalated": result.escalated, "ticket_id": result.ticket_id,
            "trigger_reason": result.trigger_reason, "user_message": result.user_message}


@router.get("/pending")
async def list_pending(queue: str = None):
    return {"tickets": _handler.list_pending(queue)}


@router.post("/{ticket_id}/resolve")
async def resolve_ticket(ticket_id: str, req: ResolveRequest):
    result = _handler.resolve_ticket(ticket_id, req.resolution, req.solution_tags)
    if result["status"] == "not_found":
        raise HTTPException(404, "工单不存在")
    return result


@router.get("/stats")
async def queue_stats():
    return _handler.get_queue_stats()
