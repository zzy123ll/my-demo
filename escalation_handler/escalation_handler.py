"""Escalation Handler 主编排器。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import EscalationConfig, load_escalation_config
from .trigger_rules import TriggerEngine, TriggerResult, TriggerReason
from .sentiment_analyzer import SentimentAnalyzer
from .ticket_manager import TicketManager, Ticket, TicketStatus
from .session_manager import SessionManager, SessionState, SessionStatus

logger = logging.getLogger(__name__)


@dataclass
class EscalationResult:
    escalated: bool
    ticket_id: str = ""
    trigger_reason: str = ""
    user_message: str = ""
    ticket_summary: str = ""


TICKET_SUMMARY_PROMPT = """你是一个客服工单专员。请根据以下信息生成一个简短的工单摘要。

【用户信息】
用户: {user_name} | 部门: {department}

【触发原因】
{trigger_reason}

【对话历史（最近5轮）】
{conversation_history}

【系统诊断】
{diagnostics}

请生成摘要，包含:
1. 用户问题概述 (1-2句)
2. 已尝试的解决方案
3. 失败原因分析
4. 用户情绪评估

输出一段 100-150 字的摘要。"""


class EscalationHandler:
    """转接处理器。

    流程:
    1. TriggerEngine 评估是否触发转接
    2. SentimentAnalyzer 评估情绪
    3. 触发 → 调用 LLM 生成工单摘要
    4. TicketManager 创建工单
    5. SessionManager 标记会话状态
    """

    def __init__(self, config: EscalationConfig = None, llm=None):
        self.config = config or load_escalation_config()
        self.trigger = TriggerEngine(self.config)
        self.sentiment = SentimentAnalyzer()
        self.tickets = TicketManager(ttl_minutes=self.config.ticket_ttl_minutes)
        self.sessions = SessionManager()
        self.llm = llm

    async def evaluate(self, session_id: str, user_id: str,
                       user_message: str,
                       user_name: str = "",
                       department: str = "",
                       safety_verdict: dict = None,
                       hallucination_verdict: dict = None,
                       conversation_history: list[dict] = None
                       ) -> EscalationResult:
        """评估是否需要转接。

        Returns:
            EscalationResult: 转接结果
        """
        # 情感分析
        sentiment_score = self.sentiment.analyze(user_message)

        # 触发评估
        trigger_result = self.trigger.evaluate(
            session_id=session_id,
            user_message=user_message,
            safety_verdict=safety_verdict,
            hallucination_verdict=hallucination_verdict,
            sentiment_score=sentiment_score,
        )

        if not trigger_result.should_escalate:
            return EscalationResult(escalated=False)

        # 确保会话存在
        self.sessions.get_or_create(session_id, user_id,
                                     user_name=user_name, department=department)

        # 更新会话历史
        for turn in (conversation_history or []):
            self.sessions.add_turn(session_id, turn.get("role", "user"),
                                   turn.get("content", ""))

        # 生成工单摘要
        summary = await self._generate_summary(
            user_name, department, trigger_result.reason,
            conversation_history or [],
            safety_verdict, hallucination_verdict,
        )

        # 创建工单
        ticket = self.tickets.create(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            user_department=department,
            trigger_reason=trigger_result.reason,
            summary=summary,
            user_query=user_message,
            conversation_history=conversation_history or [],
            attempted_solutions="AI 自动应答",
            failure_reason=trigger_result.reason,
            sentiment_assessment=f"情感分数: {sentiment_score:.2f}",
            priority=self._map_priority(trigger_result),
        )

        # 标记会话状态
        self.sessions.escalate(session_id, ticket.ticket_id)

        return EscalationResult(
            escalated=True,
            ticket_id=ticket.ticket_id,
            trigger_reason=trigger_result.reason,
            user_message="已为您转接人工客服，请稍候...",
            ticket_summary=summary,
        )

    async def _generate_summary(self, user_name, department,
                                trigger_reason, history,
                                safety, hallucination) -> str:
        # LLM 不可用时的规则摘要
        history_text = "\n".join(
            f"[{t.get('role','')}]: {t.get('content','')[:200]}"
            for t in (history or [])[-5:]
        )
        diagnostics_parts = []
        if safety:
            diagnostics_parts.append(f"安全检测: {safety.get('decision','')}")
        if hallucination:
            diagnostics_parts.append(f"幻觉检测: {hallucination.get('verdict','')}")

        if self.llm and self.config.is_llm_configured():
            try:
                from langchain_core.messages import HumanMessage
                prompt = TICKET_SUMMARY_PROMPT.format(
                    user_name=user_name,
                    department=department,
                    trigger_reason=trigger_reason,
                    conversation_history=history_text,
                    diagnostics="; ".join(diagnostics_parts) or "无",
                )
                response = await asyncio.wait_for(
                    self.llm.ainvoke([HumanMessage(content=prompt)]),
                    timeout=5.0,
                )
                return response.content.strip()[:200]
            except Exception as e:
                logger.warning(f"LLM summary failed: {e}")

        return (
            f"用户 {user_name} ({department}) 触发转接。"
            f"原因: {trigger_reason}。"
            f"对话轮次: {len(history)}。"
        )

    def _map_priority(self, trigger: TriggerResult) -> int:
        if trigger.trigger_type == TriggerReason.SAFETY_APPEAL:
            return 1
        if trigger.trigger_type == TriggerReason.HALLUCINATION:
            return 2
        if trigger.trigger_type == TriggerReason.NEGATIVE_SENTIMENT:
            return 2
        return 3

    # ---- REST API 接口 ----

    def escalate(self, **kwargs) -> EscalationResult:
        """同步转接入口，兼容 async 和 sync 上下文。"""
        try:
            loop = asyncio.get_running_loop()
            # Async context: use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.evaluate(**kwargs))
                return future.result(timeout=10)
        except RuntimeError:
            # No running loop: create one
            try:
                loop = asyncio.new_event_loop()
                return loop.run_until_complete(self.evaluate(**kwargs))
            finally:
                loop.close()
        """同步转接入口。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        return loop.run_until_complete(self.evaluate(**kwargs))

    def list_pending(self, queue: str = None) -> list[dict]:
        return self.tickets.list_pending(queue)

    def resolve_ticket(self, ticket_id: str, resolution: str,
                       solution_tags: list[str] = None) -> dict:
        ticket = self.tickets.resolve(ticket_id, resolution, solution_tags)
        if ticket:
            # 恢复会话
            self.sessions.resume(ticket.session_id)
            return {"status": "resolved", "ticket_id": ticket_id}
        return {"status": "not_found", "ticket_id": ticket_id}

    def inject_human_reply(self, session_id: str, agent_id: str,
                           reply: str) -> dict:
        self.sessions.inject_human_reply(session_id, agent_id, reply)
        return {"status": "injected", "session_id": session_id}

    def get_session_state(self, session_id: str) -> dict | None:
        sess = self.sessions.get(session_id)
        return sess.to_dict() if sess else None

    def get_ticket(self, ticket_id: str) -> dict | None:
        t = self.tickets.get(ticket_id)
        return t.to_dict() if t else None

    def get_queue_stats(self) -> dict:
        return self.tickets.stats()
