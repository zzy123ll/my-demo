"""Pytest fixtures shared across test files."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from query_rewriter.conversation_state import (
    ConversationState, UserProfile, EntityInfo
)
from query_rewriter.config import RewriterConfig


@pytest.fixture
def config():
    """Return a RewriterConfig for testing (no real LLM calls)."""
    return RewriterConfig(
        llm_api_key="",
        llm_base_url="",
        llm_model="test-model",
        similarity_threshold=0.85,
        enable_llm_rewrite=True,
    )


@pytest.fixture
def empty_state():
    """Return an empty ConversationState."""
    return ConversationState(user_profile=UserProfile(
        user_id="u1", department="dev", level="P6", name="Tester"
    ))


@pytest.fixture
def multi_turn_state():
    """Return a ConversationState with 3 turns of history.

    Turn 1: 年假有几天？ -> 入职满1年5天，满5年10天
    Turn 2: 那病假呢？    -> 病假需提供医院证明...
    Turn 3: 它需要什么材料？ (当前轮)
    """
    state = ConversationState(user_profile=UserProfile(
        user_id="u1", department="dev", level="P6", name="Tester"
    ))

    # Turn 1
    state.add_user_message("年假有几天？")
    state.track_entity("年假", "topic", "带薪年假政策")
    state.track_entity("Q1休假政策", "policy", "员工手册第3章")
    state.update_topic("休假政策")
    state.add_assistant_message(
        "根据公司政策，入职满1年享有带薪年假5天，满5年10天，满10年15天。"
    )

    # Turn 2
    state.add_user_message("那病假呢？")
    state.track_entity("病假", "topic", "病假申请流程")
    state.update_topic("休假政策")
    state.add_assistant_message(
        "病假需要提供二级以上医院开具的病假证明，3天以内由部门主管审批，"
        "3天以上需HR审批。"
    )

    # Turn 3 (current: will be added in test)
    state.add_user_message("它需要什么材料？")
    state.update_topic("病假材料")
    state.add_assistant_message(
        "病假需要的材料包括：病假证明原件、请假申请表。"
    )

    return state


@pytest.fixture
def mock_llm():
    """Return a mocked LangChain BaseChatModel that returns predictable results."""
    mock = AsyncMock()

    async def mock_ainvoke(messages, **kwargs):
        content = messages[-1].content if messages else ""

        # Simulate LLM rewrite based on input patterns
        if "它" in content:
            return AIMessage(content='{"rewritten_query": "病假需要什么材料？", "strategy": "coreference", "confidence": 0.92}')
        elif "那" in content and "呢" in content:
            return AIMessage(content='{"rewritten_query": "病假有几天？", "strategy": "ellipsis", "confidence": 0.95}')
        elif "这个" in content or "那个" in content:
            return AIMessage(content='{"rewritten_query": "报销流程是什么？", "strategy": "coreference", "confidence": 0.88}')
        else:
            return AIMessage(content='{"rewritten_query": "' + (content[-80:] if len(content) > 80 else content) + '", "strategy": "passthrough", "confidence": 0.5}')

    mock.ainvoke = mock_ainvoke

    # Also support sync invoke for compatibility
    mock.invoke = MagicMock(return_value=AIMessage(content="mocked"))

    return mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence-transformers to avoid downloading models during tests.

    Returns a tuple of (encode_fn, similarity_matrix).
    encode_fn simulates encoding text and returns fake embeddings.
    """
    import numpy as np

    def fake_encode(sentences, normalize_embeddings=False, **kwargs):
        # Return deterministic fake embeddings based on text length
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for i, text in enumerate(sentences):
            # Create a simple deterministic vector based on text hash
            seed = sum(ord(c) * (j + 1) for j, c in enumerate(text[:50]))
            rng = np.random.RandomState(seed % (2**31))
            vec = rng.randn(384).astype(np.float32)
            if normalize_embeddings:
                vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec)

        return np.array(embeddings)

    return fake_encode
