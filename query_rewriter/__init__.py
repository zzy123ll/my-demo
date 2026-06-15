# 轻量模块：无 LLM 依赖，可安全导入
from .conversation_state import ConversationState, UserProfile, EntityInfo
from .config import load_config, RewriterConfig
from .coreference_rules import CoreferenceResolver

# 重量模块：需要 langchain / sentence-transformers，按需导入
# from .query_rewriter import QueryRewriter
# from .llm_rewriter import LLMRewriter
# from .consistency_checker import ConsistencyChecker

__all__ = [
    "ConversationState",
    "UserProfile",
    "EntityInfo",
    "load_config",
    "RewriterConfig",
    "CoreferenceResolver",
]
