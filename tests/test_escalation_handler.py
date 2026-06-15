"""Escalation Handler 模块的完整测试。"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from escalation_handler.config import load_escalation_config
from escalation_handler.trigger_rules import TriggerEngine, TriggerResult, TriggerReason
from escalation_handler.sentiment_analyzer import SentimentAnalyzer
from escalation_handler.ticket_manager import TicketManager, Ticket, TicketStatus
from escalation_handler.session_manager import SessionManager, SessionStatus


# ============================================================
class TestTriggerEngine:
    @pytest.fixture
    def engine(self):
        return TriggerEngine(load_escalation_config())

    def test_keyword_trigger(self, engine):
        r = engine.evaluate("s1", "我要转人工", sentiment_score=0.5)
        assert r.should_escalate
        assert r.trigger_type == TriggerReason.KEYWORD

    def test_keyword_variants(self, engine):
        for kw in ["人工客服", "转人工坐席", "找真人", "投诉", "叫你们经理"]:
            r = engine.evaluate("s1", f"帮我{kw}", sentiment_score=0.0)
            assert r.should_escalate, f"Should trigger: {kw}"

    def test_normal_query_no_trigger(self, engine):
        r = engine.evaluate("s1", "年假有几天？", sentiment_score=0.3)
        assert not r.should_escalate

    def test_hallucination_threshold(self, engine):
        for _ in range(2):
            r = engine.evaluate("s1", "问题",
                               hallucination_verdict={"verdict": "reject"},
                               sentiment_score=0.0)
        assert r.should_escalate
        assert r.trigger_type == TriggerReason.HALLUCINATION

    def test_negative_sentiment_threshold(self, engine):
        for _ in range(3):
            r = engine.evaluate("s1", "不满意",
                               sentiment_score=-0.9)
        assert r.should_escalate
        assert r.trigger_type == TriggerReason.NEGATIVE_SENTIMENT

    def test_safety_appeal(self, engine):
        r = engine.evaluate("s1", "为什么说我违规？这不合理",
                           safety_verdict={"decision": "BLOCK", "matched_category": "layoff"},
                           sentiment_score=-0.5)
        assert r.should_escalate
        assert r.trigger_type == TriggerReason.SAFETY_APPEAL

    def test_session_reset(self, engine):
        for _ in range(2):
            engine.evaluate("s1", "x", hallucination_verdict={"verdict": "reject"})
        engine.reset_session("s1")
        r = engine.evaluate("s1", "y", sentiment_score=0.0)
        assert not r.should_escalate


# ============================================================
class TestSentimentAnalyzer:
    def test_positive(self):
        sa = SentimentAnalyzer()
        assert sa.analyze("谢谢，很好很满意") > 0

    def test_negative(self):
        sa = SentimentAnalyzer()
        assert sa.analyze("太差了，失望，垃圾") < 0

    def test_negative_complaint(self):
        sa = SentimentAnalyzer()
        assert sa.analyze("我要投诉，你们的服务太差了") < -0.3

    def test_neutral(self):
        sa = SentimentAnalyzer()
        score = sa.analyze("今天天气怎么样")
        assert -0.3 <= score <= 0.3

    def test_empty(self):
        sa = SentimentAnalyzer()
        assert sa.analyze("") == 0.0

    def test_negation_reversal(self):
        sa = SentimentAnalyzer()
        s1 = sa.analyze("不错")
        s2 = sa.analyze("不是不错")
        assert s1 > 0 and s2 < s1


# ============================================================
class TestTicketManager:
    @pytest.fixture
    def tm(self):
        return TicketManager(ttl_minutes=60)

    def test_create_ticket(self, tm):
        t = tm.create(session_id="s1", user_id="u1", trigger_reason="关键词:人工")
        assert t.ticket_id.startswith("TKT-")
        assert t.status == TicketStatus.PENDING

    def test_list_pending(self, tm):
        tm.create(session_id="s1", user_id="u1", trigger_reason="人工")
        tm.create(session_id="s2", user_id="u2", trigger_reason="人工")
        pending = tm.list_pending()
        assert len(pending) == 2

    def test_claim_and_resolve(self, tm):
        t = tm.create(session_id="s1", user_id="u1", trigger_reason="人工")
        claimed = tm.claim(t.ticket_id, "agent_1")
        assert claimed is not None
        assert claimed.status == TicketStatus.IN_PROGRESS

        resolved = tm.resolve(t.ticket_id, "已解决", ["hr", "policy"])
        assert resolved is not None
        assert resolved.status == TicketStatus.RESOLVED
        assert resolved.resolution == "已解决"

    def test_queue_filtering(self, tm):
        tm.create(session_id="s1", user_id="u1", queue="hr", trigger_reason="人工")
        tm.create(session_id="s2", user_id="u2", queue="it", trigger_reason="人工")
        hr_pending = tm.list_pending(queue="hr")
        assert len(hr_pending) == 1
        assert hr_pending[0]["queue"] == "hr"

    def test_stats(self, tm):
        tm.create(session_id="s1", user_id="u1", trigger_reason="人工")
        stats = tm.stats()
        assert stats["total"] == 1
        assert stats["pending"] == 1

    def test_ticket_not_found(self, tm):
        assert tm.get("NONEXIST") is None
        assert tm.claim("NONEXIST", "a1") is None
        assert tm.resolve("NONEXIST", "x") is None


# ============================================================
class TestSessionManager:
    @pytest.fixture
    def sm(self):
        return SessionManager()

    def test_create_and_get(self, sm):
        sm.create("s1", "u1", "张三", "Engineering")
        sess = sm.get("s1")
        assert sess.user_name == "张三"
        assert sess.status == SessionStatus.ACTIVE

    def test_escalate_flow(self, sm):
        sm.create("s1", "u1")
        sm.add_turn("s1", "user", "帮我转人工")
        sm.escalate("s1", "TKT-ABC")
        sess = sm.get("s1")
        assert sess.status == SessionStatus.WAITING_HUMAN
        assert sess.current_ticket_id == "TKT-ABC"

    def test_inject_human_reply(self, sm):
        sm.create("s1", "u1")
        sm.escalate("s1", "TKT-ABC")
        sm.inject_human_reply("s1", "agent_1", "您好，我是HR专员")
        sess = sm.get("s1")
        assert sess.status == SessionStatus.HUMAN_HANDLING
        assert sess.human_agent_id == "agent_1"
        assert len(sess.conversation_history) == 1

    def test_resume(self, sm):
        sm.create("s1", "u1")
        sm.escalate("s1", "TKT-ABC")
        sm.inject_human_reply("s1", "agent_1", "已处理")
        sm.resume("s1")
        sess = sm.get("s1")
        assert sess.status == SessionStatus.RESUMED

    def test_unknown_session(self, sm):
        assert sm.get("nonexist") is None

    def test_get_or_create(self, sm):
        s1 = sm.get_or_create("s1", "u1", user_name="李四")
        s2 = sm.get_or_create("s1", "u1")
        assert s1 is s2  # 同一个对象


# ============================================================
class TestEscalationHandlerE2E:
    def test_keyword_trigger_creates_ticket(self):
        from escalation_handler.escalation_handler import EscalationHandler
        handler = EscalationHandler()
        result = handler.escalate(
            session_id="s1", user_id="u1",
            user_message="我要转人工", user_name="张三",
        )
        assert result.escalated
        assert result.ticket_id.startswith("TKT-")
        assert "人工" in result.user_message

    def test_negative_sentiment_trigger(self):
        from escalation_handler.escalation_handler import EscalationHandler
        handler = EscalationHandler()
        for i in range(3):
            result = handler.escalate(
                session_id="s1", user_id="u1",
                user_message=f"太垃圾了{i}次了都解决不了",
            )
        assert result.escalated

    def test_full_escalation_lifecycle(self):
        """完整转接生命周期: 触发 → 创建工单 → 坐席处理 → 恢复"""
        from escalation_handler.escalation_handler import EscalationHandler
        handler = EscalationHandler()

        # 1. 触发转接
        result = handler.escalate(
            session_id="s1", user_id="u1",
            user_message="转人工", user_name="张三",
        )
        assert result.escalated
        ticket_id = result.ticket_id

        # 2. 查看待处理工单
        pending = handler.list_pending()
        assert len(pending) >= 1
        assert any(t["ticket_id"] == ticket_id for t in pending)

        # 3. 查看会话状态
        sess = handler.get_session_state("s1")
        assert sess["status"] == "waiting"
        assert sess["ticket_id"] == ticket_id

        # 4. 坐席注入回复
        handler.inject_human_reply("s1", "agent_1", "已处理您的问题")
        sess = handler.get_session_state("s1")
        assert sess["status"] == "handling"

        # 5. 解决工单
        handler.resolve_ticket(ticket_id, "已向HR确认年假政策")
        ticket = handler.get_ticket(ticket_id)
        assert ticket["status"] == "resolved"

        # 6. 恢复
        sess = handler.get_session_state("s1")
        assert sess["status"] == "resumed"

    def test_user_disatisfaction_simulation(self):
        """模拟用户连续表达不满的场景。"""
        from escalation_handler.escalation_handler import EscalationHandler
        handler = EscalationHandler()

        turns = [
            "年假有几天？",
            "你给的信息不对，太差了",
            "还是没解决我的问题，失望",
            "算了，叫你们经理来",
        ]
        escalated = False
        for i, msg in enumerate(turns):
            result = handler.escalate(
                session_id="s1", user_id="u1",
                user_message=msg, user_name="张三",
            )
            if result.escalated:
                escalated = True
                if "经理" in msg:
                    assert result.trigger_reason  # 关键词触发
                    break

        assert escalated, "连续不满应触发转接"

    def test_queue_stats(self):
        from escalation_handler.escalation_handler import EscalationHandler
        handler = EscalationHandler()
        handler.escalate(session_id="s1", user_id="u1",
                        user_message="转人工")
        stats = handler.get_queue_stats()
        assert stats["total"] >= 1
        assert stats["pending"] >= 1
