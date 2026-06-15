"""QueryRewriter 主编排器：集成双阶段改写 + 一致性校验 + 反问生成。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from .config import RewriterConfig, load_config
from .conversation_state import ConversationState, EntityInfo
from .coreference_rules import CoreferenceResolver, RuleResult
from .llm_rewriter import LLMRewriter, LLMRewriteResult
from .consistency_checker import ConsistencyChecker, ConsistencyResult

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)


CLARIFICATION_PROMPT = """你需要为用户的模糊查询生成一个简短的反问，帮助澄清意图。

对话历史中涉及的主要话题和实体：
{context}

用户当前所说：{query}

这个查询为什么不清楚：{reason}

请生成一个简洁的反问句（不超过30字），让用户选择或澄清。例如：
- "您指的是刚才提到的请假流程，还是报销流程？"
- "您想了解年假的具体天数，还是申请方式？"

只输出反问句，不要加引号或其他内容。"""


@dataclass
class RewriteOutput:
    """QueryRewriter 的最终输出结构。"""
    original_query: str
    rewritten_query: str
    used_method: str
    confidence: float
    clarification_needed: bool
    clarification_question: str = ""
    consistency: Optional[ConsistencyResult] = None
    metadata: dict = field(default_factory=dict)


class QueryRewriter:
    """双阶段查询改写器。

    工作流程:
    1. 阶段一: 规则引擎尝试指代消解和省略补全
    2. 阶段二: 规则失败时调用 LLM 改写
    3. 一致性校验: 用 sentence-transformers 验证改写后查询未改变原意
    4. 反问生成: 如果无法确定，生成反问让用户澄清

    使用 LangChain 的 BaseMessage 管理对话历史，
    所有模型配置从 wolkplace/.env 隐式读取。
    """

    def __init__(self, llm=None, config: RewriterConfig = None):
        self.config = config or load_config()
        self.rule_resolver = CoreferenceResolver()
        self.llm_rewriter = LLMRewriter(llm, self.config) if llm else None
        self._consistency_checker: Optional[ConsistencyChecker] = None

    @property
    def consistency_checker(self) -> ConsistencyChecker:
        """延迟初始化一致性校验器（sentence-transformers 加载较慢）。"""
        if self._consistency_checker is None:
            self._consistency_checker = ConsistencyChecker(self.config)
        return self._consistency_checker

    async def rewrite(self, query: str,
                      state: ConversationState = None) -> RewriteOutput:
        """主入口：改写用户查询。"""
        if state is None:
            state = ConversationState()

        state.add_user_message(query)
        original = query.strip()

        # 阶段一: 规则引擎
        rule_result = self.rule_resolver.resolve(original, state)

        if rule_result and rule_result.resolved:
            consistency = self._validate(
                original, rule_result.rewritten_query, state
            )

            if consistency.passed:
                return RewriteOutput(
                    original_query=original,
                    rewritten_query=rule_result.rewritten_query,
                    used_method=f"rule_{rule_result.matched_pattern}",
                    confidence=rule_result.confidence,
                    clarification_needed=False,
                    consistency=consistency,
                    metadata={
                        "stage": "rule",
                        "matched_pronoun": rule_result.matched_pronoun,
                    },
                )
            else:
                return RewriteOutput(
                    original_query=original,
                    rewritten_query=original,
                    used_method="rule_uncertain",
                    confidence=0.5,
                    clarification_needed=True,
                    clarification_question=await self._generate_clarification(
                        original, state, "规则改写后语义相似度偏低"
                    ),
                    consistency=consistency,
                    metadata={"stage": "rule_fallback"},
                )

        # 阶段二: LLM 改写
        if self.llm_rewriter:
            llm_result = await self.llm_rewriter.rewrite(original, state)

            if llm_result and llm_result.strategy != "passthrough":
                consistency = self._validate(
                    original, llm_result.rewritten_query, state
                )

                if consistency.passed:
                    return RewriteOutput(
                        original_query=original,
                        rewritten_query=llm_result.rewritten_query,
                        used_method=f"llm_{llm_result.strategy}",
                        confidence=llm_result.confidence,
                        clarification_needed=False,
                        consistency=consistency,
                        metadata={
                            "stage": "llm",
                            "strategy": llm_result.strategy,
                        },
                    )
                else:
                    return RewriteOutput(
                        original_query=original,
                        rewritten_query=original,
                        used_method="llm_uncertain",
                        confidence=0.4,
                        clarification_needed=True,
                        clarification_question=await self._generate_clarification(
                            original, state,
                            f"LLM改写后语义相似度仅{consistency.score_original:.2f}"
                        ),
                        consistency=consistency,
                        metadata={"stage": "llm_fallback"},
                    )

        # 兜底: 透传
        return RewriteOutput(
            original_query=original,
            rewritten_query=original,
            used_method="passthrough",
            confidence=1.0,
            clarification_needed=False,
            metadata={"stage": "passthrough"},
        )

    def _validate(self, original: str, rewritten: str,
                  state: ConversationState) -> ConsistencyResult:
        context_parts = []
        if state.last_topic:
            context_parts.append(f"话题: {state.last_topic}")
        if state.current_entity:
            context_parts.append(f"实体: {state.current_entity.name}")
        for msg in state.get_recent_messages(3):
            context_parts.append(msg.content[:200])
        context_text = " ".join(context_parts)
        return self.consistency_checker.check(original, rewritten, context_text)

    async def _generate_clarification(self, query: str,
                                      state: ConversationState,
                                      reason: str) -> str:
        state.add_clarification(query, reason)

        if self.llm_rewriter and self.config.is_llm_configured():
            try:
                entities = ", ".join(state.entity_map.keys())
                context = f"话题: {state.last_topic}, 已知实体: {entities}"
                messages = [
                    HumanMessage(content=CLARIFICATION_PROMPT.format(
                        context=context, query=query, reason=reason,
                    )),
                ]
                llm = self.llm_rewriter.llm
                response = await llm.ainvoke(messages)
                return response.content.strip()[:100]
            except Exception:
                pass

        return self._rule_clarification(query, state)

    def _rule_clarification(self, query: str,
                            state: ConversationState) -> str:
        entities = list(state.entity_map.keys())
        if len(entities) >= 2:
            return f"您指的是{entities[-2]}，还是{entities[-1]}？"
        if state.last_topic:
            return f"您想了解{state.last_topic}的哪个方面？请再详细说明一下。"
        return "抱歉我没太理解，您能再详细说明一下吗？"

    def track_entity_from_query(self, query: str,
                                state: ConversationState) -> None:
        import re
        policy_pattern = r'《([^》]+)》'
        entity_pattern = r'(?:年假|病假|事假|调休|绩效|薪资|报销|加班|培训|入职|离职)'

        for m in re.finditer(policy_pattern, query):
            state.track_entity(m.group(1), "policy", m.group())
        for m in re.finditer(entity_pattern, query):
            state.track_entity(m.group(), "topic", m.group())
