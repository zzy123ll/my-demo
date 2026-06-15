"""会话状态管理器 — 转接后保留状态并支持恢复。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class SessionStatus(Enum):
    ACTIVE = "active"           # 正常机器人服务中
    WAITING_HUMAN = "waiting"   # 已转接，等待人工
    HUMAN_HANDLING = "handling" # 人工坐席处理中
    RESUMED = "resumed"         # 机器人恢复服务


@dataclass
class SessionState:
    session_id: str
    user_id: str
    user_name: str = ""
    user_department: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    conversation_history: list[dict] = field(default_factory=list)
    current_ticket_id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    human_agent_id: str = ""
    human_reply: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "ticket_id": self.current_ticket_id,
            "agent_id": self.human_agent_id,
            "rounds": len(self.conversation_history),
        }


class SessionManager:
    """内存会话管理器。

    转接后原会话保留并标记为 WAITING_HUMAN，
    坐席回复通过 inject_human_reply 注入对话历史，
    机器人通过 resume_session 恢复服务。
    """

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str, user_id: str,
               user_name: str = "", department: str = "") -> SessionState:
        sess = SessionState(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            user_department=department,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._sessions[session_id] = sess
        return sess

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str, user_id: str = "",
                      **kwargs) -> SessionState:
        sess = self._sessions.get(session_id)
        if sess:
            return sess
        return self.create(session_id, user_id, **kwargs)

    def add_turn(self, session_id: str, role: str,
                 content: str) -> None:
        sess = self._sessions.get(session_id)
        if sess:
            sess.conversation_history.append({
                "role": role,
                "content": content,
                "timestamp": time.time(),
            })
            sess.updated_at = time.time()

    def escalate(self, session_id: str, ticket_id: str) -> None:
        """标记会话为等待人工。"""
        sess = self._sessions.get(session_id)
        if sess:
            sess.status = SessionStatus.WAITING_HUMAN
            sess.current_ticket_id = ticket_id
            sess.updated_at = time.time()

    def inject_human_reply(self, session_id: str, agent_id: str,
                           reply: str) -> None:
        """坐席回复注入对话历史。"""
        sess = self._sessions.get(session_id)
        if sess:
            sess.human_agent_id = agent_id
            sess.human_reply = reply
            sess.status = SessionStatus.HUMAN_HANDLING
            sess.conversation_history.append({
                "role": "agent",
                "content": reply,
                "agent_id": agent_id,
                "timestamp": time.time(),
            })
            sess.updated_at = time.time()

    def resume(self, session_id: str) -> SessionState | None:
        """机器人恢复服务。"""
        sess = self._sessions.get(session_id)
        if sess:
            sess.status = SessionStatus.RESUMED
            sess.updated_at = time.time()
        return sess

    def list_waiting(self) -> list[SessionState]:
        return [s for s in self._sessions.values()
                if s.status == SessionStatus.WAITING_HUMAN]
