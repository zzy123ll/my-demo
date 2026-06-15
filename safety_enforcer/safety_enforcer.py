"""Safety Enforcer 主编排器 — L1 → L2 → L3 三级流水线。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import EnforcerConfig, load_enforcer_config
from .l1_keyword_filter import L1KeywordFilter, L1Result
from .l2_text_classifier import L2TextClassifier, L2Result
from .l3_llm_arbiter import L3LLMArbiter, L3Result
from .audit_logger import AuditLogger
from .access_control import AccessController

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    """安全检测最终输出。"""
    decision: str              # "PASS" | "BLOCK" | "WARN"
    triggered_level: str       # "L1" | "L2" | "L3"
    matched_category: str
    score: float
    action: str                # block / block_and_escalate / warn
    escalate: bool
    user_message: str          # 返回给用户的提示
    details: dict = field(default_factory=dict)


class SafetyEnforcer:
    """安全执行器。

    流水线:
    1. L1 (关键词+正则): <1ms, 命中→直接拦截
    2. L2 (文本分类): ~10-20ms, score>0.7→拦截, 0.4-0.7→L3
    3. L3 (LLM裁决): 仅模糊区间, 2s超时→按L2处理

    附: AccessController 做部门级文档访问控制
    """

    def __init__(self, config: EnforcerConfig = None, llm=None):
        self.config = config or load_enforcer_config()
        self.l1 = L1KeywordFilter(self.config.categories)
        self.l2 = L2TextClassifier(
            self.config.categories,
            threshold=self.config.l2_similarity_threshold,
        )
        self.l3 = L3LLMArbiter(llm=llm, config=self.config)
        self.audit = AuditLogger(
            enabled=self.config.audit_enabled,
            webhook_config=self.config.notifications,
        )
        self.access = AccessController(self.config.access_control)

    async def enforce(self, query: str,
                      user_id: str = "anonymous",
                      department: str = "unknown",
                      session_id: str = "") -> SafetyResult:
        """执行安全检测。

        Args:
            query: 用户查询文本
            user_id: 用户 ID
            department: 用户部门
            session_id: 会话 ID

        Returns:
            SafetyResult: 检测结果
        """
        t0 = time.perf_counter()

        # ==== L1: 快速过滤 ====
        l1_result = self.l1.check(query)

        if l1_result.blocked:
            total_ms = (time.perf_counter() - t0) * 1000
            result = SafetyResult(
                decision="BLOCK",
                triggered_level="L1",
                matched_category=l1_result.matched_category,
                score=1.0,
                action=l1_result.action,
                escalate=l1_result.action == "block_and_escalate",
                user_message=self._block_message(l1_result.matched_category,
                                                  l1_result.action),
                details={
                    "matched_keywords": l1_result.matched_keywords,
                    "matched_regex": l1_result.matched_regex,
                    "l1_latency_us": l1_result.latency_us,
                    "total_latency_ms": round(total_ms, 2),
                },
            )
            self.audit.log(
                user_id=user_id, department=department,
                original_query=query, decision="BLOCK",
                triggered_by="L1",
                matched_category=l1_result.matched_category,
                score=1.0, action=l1_result.action,
                session_id=session_id,
            )
            return result

        # ==== L2: 文本分类 ====
        l2_result = self.l2.classify(query)

        if l2_result.score > self.config.l2_similarity_threshold:
            total_ms = (time.perf_counter() - t0) * 1000
            cat = self.config.categories.get(l2_result.top_category)
            action = cat.action if cat else "block"
            result = SafetyResult(
                decision="BLOCK",
                triggered_level="L2",
                matched_category=l2_result.top_category,
                score=l2_result.score,
                action=action,
                escalate=action == "block_and_escalate",
                user_message=self._block_message(l2_result.top_category, action),
                details={
                    "l2_scores": l2_result.all_scores,
                    "l1_latency_us": l1_result.latency_us,
                    "l2_latency_us": l2_result.latency_us,
                    "total_latency_ms": round(total_ms, 2),
                },
            )
            self.audit.log(
                user_id=user_id, department=department,
                original_query=query, decision="BLOCK",
                triggered_by="L2",
                matched_category=l2_result.top_category,
                score=l2_result.score, action=action,
                session_id=session_id,
            )
            return result

        # ==== L3: LLM 裁决（仅在模糊区间） ====
        if l2_result.needs_l3:
            l3_result = await self.l3.arbitrate(
                query, l2_result.top_category, l2_result.score
            )

            total_ms = (time.perf_counter() - t0) * 1000

            if l3_result.verdict == "unsafe":
                cat_name = l3_result.category or l2_result.top_category
                cat = self.config.categories.get(cat_name)
                action = cat.action if cat else "block"
                result = SafetyResult(
                    decision="BLOCK",
                    triggered_level="L3",
                    matched_category=cat_name,
                    score=l3_result.confidence,
                    action=action,
                    escalate=action == "block_and_escalate",
                    user_message=self._block_message(cat_name, action),
                    details={
                        "l3_verdict": l3_result.verdict,
                        "l3_confidence": l3_result.confidence,
                        "l1_latency_us": l1_result.latency_us,
                        "l2_latency_us": l2_result.latency_us,
                        "l3_latency_ms": l3_result.latency_ms,
                        "total_latency_ms": round(total_ms, 2),
                    },
                )
                self.audit.log(
                    user_id=user_id, department=department,
                    original_query=query, decision="BLOCK",
                    triggered_by="L3",
                    matched_category=cat_name,
                    score=l3_result.confidence, action=action,
                    session_id=session_id,
                )
                return result
        else:
            total_ms = (time.perf_counter() - t0) * 1000

        # ==== PASS ====
        total_ms = (time.perf_counter() - t0) * 1000
        result = SafetyResult(
            decision="PASS",
            triggered_level="L2",
            matched_category="",
            score=l2_result.score,
            action="",
            escalate=False,
            user_message="",
            details={
                "l2_scores": l2_result.all_scores,
                "total_latency_ms": round(total_ms, 2),
            },
        )
        return result

    def _block_message(self, category: str, action: str) -> str:
        if action == "block_and_escalate":
            return "该问题涉及敏感信息，已为您转接人工客服处理。"
        return "抱歉，该问题超出我能回答的范围。"

    def filter_documents(self, docs: list[dict],
                         department: str) -> list[dict]:
        """部门级文档访问过滤。"""
        return self.access.filter_documents(docs, department)

    def get_chroma_filter(self, department: str) -> dict | None:
        """获取 ChromaDB 的 where 条件。"""
        return self.access.get_chroma_where_filter(department)
