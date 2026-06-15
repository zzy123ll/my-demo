"""工单队列管理器 — 内存实现 (可替换为 Redis)。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TicketStatus(Enum):
    OPEN = "open"
    PENDING = "pending"          # 等待坐席接单
    IN_PROGRESS = "in_progress"  # 坐席处理中
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class Ticket:
    ticket_id: str
    session_id: str
    user_id: str
    user_name: str = ""
    user_department: str = ""
    queue: str = "general"
    priority: int = 3                # 1=紧急, 2=高, 3=普通
    trigger_reason: str = ""
    summary: str = ""
    conversation_summary: str = ""
    attempted_solutions: str = ""
    failure_reason: str = ""
    sentiment_assessment: str = ""
    user_query: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    status: TicketStatus = TicketStatus.OPEN
    assigned_agent: str = ""
    resolution: str = ""
    solution_tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    resolved_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "queue": self.queue,
            "priority": self.priority,
            "trigger_reason": self.trigger_reason,
            "summary": self.summary,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at,
        }


class TicketQueue:
    """内存工单队列。

    API:
    - POST /escalate    → create_ticket()
    - GET  /pending     → list_pending()
    - POST /resolve     → resolve_ticket()
    """

    def __init__(self, ttl_minutes: int = 60, max_size: int = 1000):
        self._tickets: dict[str, Ticket] = {}
        self._queue: dict[str, list[str]] = {}  # queue_name -> [ticket_ids]
        self.ttl_seconds = ttl_minutes * 60
        self.max_size = max_size

    def create(self, **kwargs) -> Ticket:
        self._cleanup_expired()

        if len(self._tickets) >= self.max_size:
            raise RuntimeError("工单队列已满")

        ticket = Ticket(
            ticket_id=f"TKT-{uuid.uuid4().hex[:8].upper()}",
            created_at=time.time(),
            updated_at=time.time(),
            status=TicketStatus.OPEN,
            **kwargs,
        )

        self._tickets[ticket.ticket_id] = ticket
        queue_name = ticket.queue or "general"
        self._queue.setdefault(queue_name, []).append(ticket.ticket_id)
        ticket.status = TicketStatus.PENDING

        return ticket

    def list_pending(self, queue: str = None, limit: int = 50) -> list[dict]:
        self._cleanup_expired()
        result = []

        queues = [queue] if queue else self._queue.keys()
        for q in queues:
            for tid in self._queue.get(q, []):
                t = self._tickets.get(tid)
                if t and t.status == TicketStatus.PENDING:
                    result.append(t.to_dict())
                    if len(result) >= limit:
                        return result

        return result

    def claim(self, ticket_id: str, agent_id: str) -> Ticket | None:
        t = self._tickets.get(ticket_id)
        if t and t.status == TicketStatus.PENDING:
            t.status = TicketStatus.IN_PROGRESS
            t.assigned_agent = agent_id
            t.updated_at = time.time()
            return t
        return None

    def resolve(self, ticket_id: str, resolution: str,
                solution_tags: list[str] = None) -> Ticket | None:
        t = self._tickets.get(ticket_id)
        if t and t.status in (TicketStatus.PENDING, TicketStatus.IN_PROGRESS):
            t.status = TicketStatus.RESOLVED
            t.resolution = resolution
            t.solution_tags = solution_tags or []
            t.resolved_at = time.time()
            t.updated_at = time.time()
            return t
        return None

    def get(self, ticket_id: str) -> Ticket | None:
        return self._tickets.get(ticket_id)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            tid for tid, t in self._tickets.items()
            if t.status == TicketStatus.PENDING
            and (now - t.created_at) > self.ttl_seconds
        ]
        for tid in expired:
            t = self._tickets[tid]
            t.status = TicketStatus.EXPIRED

    def stats(self) -> dict:
        return {
            "total": len(self._tickets),
            "pending": sum(1 for t in self._tickets.values() if t.status == TicketStatus.PENDING),
            "in_progress": sum(1 for t in self._tickets.values() if t.status == TicketStatus.IN_PROGRESS),
            "resolved": sum(1 for t in self._tickets.values() if t.status == TicketStatus.RESOLVED),
            "expired": sum(1 for t in self._tickets.values() if t.status == TicketStatus.EXPIRED),
        }


# 别名
TicketManager = TicketQueue
