"""QueryRewriter 模块的完整测试。Lazy import 所有重量模块避免 torch 等依赖冲突。"""

import sys
from unittest.mock import MagicMock, patch
import pytest

from langchain_core.messages import HumanMessage, AIMessage

from query_rewriter.conversation_state import (
    ConversationState, UserProfile, EntityInfo
)
from query_rewriter.coreference_rules import CoreferenceResolver
from query_rewriter.config import RewriterConfig


# ---- Lazy imports ----
def _qr():
    from query_rewriter.query_rewriter import QueryRewriter, RewriteOutput
    return QueryRewriter, RewriteOutput

def _cc():
    from query_rewriter.consistency_checker import ConsistencyChecker
    return ConsistencyChecker


# ============================================================
class TestConversationState:
    def test_empty_state(self, empty_state):
        assert empty_state.round_count == 0
        assert empty_state.last_topic == ""
        assert empty_state.current_entity is None

    def test_add_messages(self, empty_state):
        empty_state.add_user_message("年假有几天？")
        assert empty_state.round_count == 1
        assert isinstance(empty_state.messages[0], HumanMessage)
        empty_state.add_assistant_message("入职满1年有5天年假。")
        assert len(empty_state.messages) == 2
        assert isinstance(empty_state.messages[1], AIMessage)

    def test_track_entity(self, empty_state):
        entity = empty_state.track_entity("年假", "topic", "带薪年假")
        assert entity.name == "年假"
        assert entity.type == "topic"
        assert "年假" in empty_state.entity_map

    def test_find_latest_entity_by_type(self, empty_state):
        empty_state.track_entity("张三", "person")
        empty_state.track_entity("年假", "topic")
        empty_state.track_entity("李四", "person")
        person = empty_state.find_latest_entity_by_type("person")
        assert person is not None
        # "李四" 比 "张三" 后添加，应该被找到
        assert person.name == "李四"

    def test_get_recent_messages(self, empty_state):
        for i in range(5):
            empty_state.add_user_message(f"问题{i}")
            empty_state.add_assistant_message(f"回答{i}")
        recent = empty_state.get_recent_messages(3)
        assert len(recent) == 6
        assert recent[-1].content == "回答4"

    def test_summarize(self, empty_state):
        empty_state.track_entity("年假", "topic")
        empty_state.update_topic("休假政策")
        summary = empty_state.summarize()
        assert "休假政策" in summary
        assert "年假" in summary


# ============================================================
class TestCoreferenceResolver:
    def test_pronoun_it_to_topic(self, empty_state):
        empty_state.track_entity("年假", "topic")
        empty_state.track_entity("张三", "person")
        empty_state.update_topic("休假政策")
        resolver = CoreferenceResolver()
        result = resolver.resolve("它需要什么材料？", empty_state)
        assert result is not None
        assert result.resolved
        assert "年假" in result.rewritten_query

    def test_pronoun_this(self, empty_state):
        empty_state.track_entity("绩效评定制度", "policy")
        empty_state.update_topic("绩效考核")
        resolver = CoreferenceResolver()
        result = resolver.resolve("这个的具体标准是什么？", empty_state)
        assert result is not None
        assert "绩效评定制度" in result.rewritten_query

    def test_ellipsis_na_ne(self, multi_turn_state):
        resolver = CoreferenceResolver()
        result = resolver.resolve("那事假呢？", multi_turn_state)
        assert result is not None
        assert result.resolved
        assert "事假" in result.rewritten_query
        assert result.matched_pattern == "topic_ellipsis"

    def test_supplement_request(self, multi_turn_state):
        resolver = CoreferenceResolver()
        result = resolver.resolve("还有吗？", multi_turn_state)
        assert result is not None
        assert result.resolved
        assert result.matched_pattern == "supplement_request"

    def test_detail_inquiry(self, multi_turn_state):
        resolver = CoreferenceResolver()
        result = resolver.resolve("具体怎么申请？", multi_turn_state)
        assert result is not None
        assert result.resolved
        assert result.matched_pattern == "detail_inquiry"
        assert "申请" in result.rewritten_query

    def test_no_history_no_resolve(self, empty_state):
        resolver = CoreferenceResolver()
        result = resolver.resolve("它是什么？", empty_state)
        assert result is None or not result.resolved

    def test_new_topic_passthrough(self, empty_state):
        empty_state.track_entity("年假", "topic")
        empty_state.update_topic("休假政策")
        resolver = CoreferenceResolver()
        result = resolver.resolve("公司的WiFi密码是什么？", empty_state)
        assert result is None

    def test_person_pronoun_he(self, empty_state):
        empty_state.track_entity("张三", "person")
        empty_state.track_entity("绩效评定", "topic")
        resolver = CoreferenceResolver()
        result = resolver.resolve("他的绩效评定结果呢？", empty_state)
        assert result is not None
        assert "张三" in result.rewritten_query


# ============================================================
class TestConsistencyChecker:
    def test_identical_passes(self, config, mock_sentence_transformer):
        CC = _cc()
        with patch.object(CC, 'model', new_callable=MagicMock) as m:
            m.encode = mock_sentence_transformer
            checker = CC(config)
            result = checker.check("年假有几天？", "年假有几天？")
            assert result.passed
            assert result.score_original == 1.0

    def test_high_similarity_passes(self, config, mock_sentence_transformer):
        import numpy as np
        def high_sim_encode(sentences, normalize_embeddings=False, **kw):
            if isinstance(sentences, str):
                sentences = [sentences]
            base = np.ones(384, dtype=np.float32) / np.sqrt(384)
            if len(sentences) == 1:
                return np.array([base])
            vecs = [base]
            for _ in range(len(sentences) - 1):
                noise = np.random.RandomState(42).randn(384).astype(np.float32) * 0.02
                v = base + noise
                vecs.append(v / np.linalg.norm(v))
            return np.array(vecs)

        CC = _cc()
        with patch.object(CC, 'model', new_callable=MagicMock) as m:
            m.encode = high_sim_encode
            checker = CC(config)
            result = checker.check("年假有几天？", "带薪年假的天数是多少？")
            assert result.passed
            assert result.score_original >= config.similarity_threshold

    def test_low_similarity_fails(self, config, mock_sentence_transformer):
        import numpy as np
        def low_sim_encode(sentences, normalize_embeddings=False, **kw):
            vecs = []
            for i, _ in enumerate(sentences):
                rng = np.random.RandomState(i * 999)
                v = rng.randn(384).astype(np.float32)
                vecs.append(v / np.linalg.norm(v))
            return np.array(vecs)

        CC = _cc()
        with patch.object(CC, 'model', new_callable=MagicMock) as m:
            m.encode = low_sim_encode
            checker = CC(config)
            result = checker.check("年假有几天？", "公司的盈利状况如何？")
            assert not result.passed

    def test_with_context(self, config, mock_sentence_transformer):
        import numpy as np
        def fake_encode(sentences, normalize_embeddings=False, **kw):
            vecs = []
            for s in sentences:
                rng = np.random.RandomState(hash(s) % (2**31))
                v = rng.randn(384).astype(np.float32)
                vecs.append(v / np.linalg.norm(v))
            return np.array(vecs)

        CC = _cc()
        with patch.object(CC, 'model', new_callable=MagicMock) as m:
            m.encode = fake_encode
            checker = CC(config)
            result = checker.check("年假有几天？", "带薪年假天数多少？",
                                   "用户之前问了休假政策")
            assert isinstance(result.score_context, float); assert -1.0 <= result.score_context <= 1.0


# ============================================================
class TestQueryRewriterE2E:
    @pytest.mark.asyncio
    async def test_passthrough_new_topic(self, config, mock_llm, empty_state):
        QR, _ = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=True, score_original=1.0, score_context=1.0,
            threshold=0.85, verdict="pass")):
            rewriter = QR(llm=mock_llm, config=config)
            result = await rewriter.rewrite("公司的WiFi密码是什么？", empty_state)
            assert result.used_method == "passthrough"
            assert result.clarification_needed is False
            assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_rule_coreference_it(self, config, mock_llm, multi_turn_state):
        QR, _ = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=True, score_original=0.95, score_context=0.9,
            threshold=0.85, verdict="pass")):
            rewriter = QR(llm=mock_llm, config=config)
            state = ConversationState()
            state.add_user_message("病假需要什么条件？")
            state.track_entity("病假", "topic")
            state.update_topic("病假")
            result = await rewriter.rewrite("它需要什么材料？", state)
            assert result.used_method.startswith("rule_")
            assert "病假" in result.rewritten_query

    @pytest.mark.asyncio
    async def test_rule_ellipsis_na_ne(self, config, mock_llm, multi_turn_state):
        QR, _ = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=True, score_original=0.9, score_context=0.88,
            threshold=0.85, verdict="pass")):
            rewriter = QR(llm=mock_llm, config=config)
            state = ConversationState()
            state.add_user_message("年假有几天？")
            state.track_entity("年假", "topic")
            state.update_topic("休假政策")
            result = await rewriter.rewrite("那病假呢？", state)
            assert result.used_method == "rule_topic_ellipsis"
            assert "病假" in result.rewritten_query

    @pytest.mark.asyncio
    async def test_clarification_when_low_similarity(self, config, mock_llm,
                                                      multi_turn_state):
        QR, _ = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=False, score_original=0.55, score_context=0.5,
            threshold=0.85, verdict="semantic_drift")):
            rewriter = QR(llm=mock_llm, config=config)
            state = ConversationState()
            state.add_user_message("年假有几天？")
            state.track_entity("年假", "topic")
            state.update_topic("休假政策")
            result = await rewriter.rewrite("那病假呢？", state)
            assert result.clarification_needed is True
            assert len(result.clarification_question) > 0
            assert result.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_multi_turn_simulation(self, config, mock_llm):
        QR, RewriteOutput = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=True, score_original=0.92, score_context=0.9,
            threshold=0.85, verdict="pass")):
            rewriter = QR(llm=mock_llm, config=config)
            state = ConversationState(user_profile=UserProfile(
                user_id="u1", department="dev", level="P6"))
            r1 = await rewriter.rewrite("年假有几天？", state)
            state.track_entity("年假", "topic")
            state.update_topic("休假政策")
            state.add_assistant_message("入职满1年有5天年假。")
            assert r1.used_method == "passthrough"
            r2 = await rewriter.rewrite("那病假呢？", state)
            state.track_entity("病假", "topic")
            state.add_assistant_message("病假需要提供医院证明。")
            assert r2.used_method.startswith("rule_")
            assert "病假" in r2.rewritten_query
            r3 = await rewriter.rewrite("它需要什么材料？", state)
            state.add_assistant_message("需要医院证明和请假申请表。")
            assert r3.used_method.startswith("rule_")
            r4 = await rewriter.rewrite("还有吗？", state)
            assert r4.used_method == "rule_supplement_request"

    @pytest.mark.asyncio
    async def test_output_structure(self, config, mock_llm, empty_state):
        QR, RewriteOutput = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=True, score_original=0.95, score_context=0.9,
            threshold=0.85, verdict="pass")):
            rewriter = QR(llm=mock_llm, config=config)
            result = await rewriter.rewrite("你好", empty_state)
            assert isinstance(result, RewriteOutput)
            for attr in ['original_query', 'rewritten_query', 'used_method',
                         'confidence', 'clarification_needed',
                         'clarification_question', 'consistency', 'metadata']:
                assert hasattr(result, attr)
            assert 0.0 <= result.confidence <= 1.0


# ============================================================
class TestEdgeCases:
    def test_empty_query(self, config, mock_llm, empty_state):
        assert CoreferenceResolver().resolve("", empty_state) is None

    def test_very_short_query(self, config, mock_llm, empty_state):
        assert CoreferenceResolver().resolve("？", empty_state) is None

    @pytest.mark.asyncio
    async def test_llm_unavailable_fallback(self, config, empty_state):
        config.enable_llm_rewrite = False
        QR, _ = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=True, score_original=1.0, score_context=1.0,
            threshold=0.85, verdict="pass")):
            rewriter = QR(llm=None, config=config)
            result = await rewriter.rewrite("年假有几天？", empty_state)
            assert result.used_method == "passthrough"
            assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_llm_returns_passthrough(self, config, mock_llm, empty_state):
        QR, _ = _qr(); CC = _cc()
        with patch.object(CC, 'check', return_value=MagicMock(
            passed=True, score_original=1.0, score_context=1.0,
            threshold=0.85, verdict="pass")):
            rewriter = QR(llm=mock_llm, config=config)
            result = await rewriter.rewrite("公司的地址在哪里？", empty_state)
            assert result.used_method == "passthrough"

    def test_track_entity_from_query(self, config, mock_llm, empty_state):
        QR, _ = _qr()
        rewriter = QR(llm=None, config=config)
        rewriter.track_entity_from_query("《员工手册》中的年假规定", empty_state)
        assert "《员工手册》" in empty_state.entity_map or \
               "员工手册" in empty_state.entity_map
        assert "年假" in empty_state.entity_map
