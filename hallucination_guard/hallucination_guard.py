"""Hallucination Guard 主编排器。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import GuardConfig, load_guard_config
from .nli_checker import NLIChecker
from .claim_splitter import ClaimSplitter, AtomicClaim

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    statement_index: int
    source_doc_id: str
    highlight_text: str


@dataclass
class UnsupportedSpan:
    text: str
    reason: str
    statement_index: int


@dataclass
class GuardOutput:
    """幻觉检测的最终输出。"""
    safe_answer: str
    unsupported_spans: list[UnsupportedSpan]
    citations: list[Citation]
    all_hallucination: bool
    total_statements: int
    supported_count: int
    hallucination_count: int
    detection_latency_ms: float
    metadata: dict = field(default_factory=dict)


class HallucinationGuard:
    """幻觉检测器。

    流程:
    1. ClaimSplitter 将 LLM 回答拆分为原子声明
    2. NLIChecker 对每条声明检查是否被检索文档蕴含
    3. 按文档顺序输出:
       - safe_answer: 仅保留有支撑的声明
       - unsupported_spans: 无支撑的声明
       - citations: 每条声明的来源引用
    4. 全幻觉时返回预设话术
    """

    DEFAULT_FALLBACK = "抱歉，我无法根据现有知识确认这一点。"

    def __init__(self, config: GuardConfig = None):
        self.config = config or load_guard_config()
        self.splitter = ClaimSplitter()
        self.nli = NLIChecker(
            model_name=self.config.nli_model,
            entailment_threshold=self.config.entailment_threshold,
            contradiction_threshold=self.config.contradiction_threshold,
            batch_size=self.config.nli_batch_size,
            max_length=self.config.nli_max_length,
        )

    def guard(self, answer: str,
              retrieved_docs: list[dict]) -> GuardOutput:
        """检测并过滤幻觉。

        Args:
            answer: LLM 生成的回答
            retrieved_docs: 检索到的文档列表
                [{chunk_id, doc_id, content, metadata}, ...]

        Returns:
            GuardOutput: 幻觉检测结果
        """
        t0 = time.perf_counter()

        if not answer or not answer.strip():
            return GuardOutput(
                safe_answer="", unsupported_spans=[], citations=[],
                all_hallucination=False, total_statements=0,
                supported_count=0, hallucination_count=0,
                detection_latency_ms=0,
            )

        # 1. 拆分回答为原子声明
        claims = self.splitter.split(answer)
        if not claims:
            return GuardOutput(
                safe_answer=answer, unsupported_spans=[], citations=[],
                all_hallucination=False, total_statements=0,
                supported_count=0, hallucination_count=0,
                detection_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # 2. 提取文档前提文本
        premise_texts = [doc.get("content", "") for doc in retrieved_docs]
        premise_texts = [t for t in premise_texts if t.strip()]

        # 3. 对每条声明做 NLI 检测
        supported = []
        unsupported = []
        citations = []

        for claim in claims:
            if self.nli.check_entailment(premise_texts, claim.text):
                supported.append(claim)
                # 找到最佳支撑文档
                best_doc = self._find_best_doc(claim.text, retrieved_docs)
                if best_doc:
                    citations.append(Citation(
                        statement_index=claim.index,
                        source_doc_id=best_doc.get("doc_id",
                                                    best_doc.get("chunk_id", "")),
                        highlight_text=claim.text[:100],
                    ))
            else:
                unsupported.append(UnsupportedSpan(
                    text=claim.text,
                    reason="无文档支持",
                    statement_index=claim.index,
                ))

        # 4. 构建 safe_answer
        all_hallucination = len(supported) == 0 and len(unsupported) > 0

        if all_hallucination:
            safe_answer = self.config.fallback_message
        else:
            safe_answer = "".join(c.text for c in sorted(supported,
                                         key=lambda x: x.index))

        latency = (time.perf_counter() - t0) * 1000

        return GuardOutput(
            safe_answer=safe_answer,
            unsupported_spans=unsupported,
            citations=citations,
            all_hallucination=all_hallucination,
            total_statements=len(claims),
            supported_count=len(supported),
            hallucination_count=len(unsupported),
            detection_latency_ms=round(latency, 1),
        )

    def _find_best_doc(self, claim: str,
                       docs: list[dict]) -> Optional[dict]:
        """找到最佳支撑文档（简单关键词匹配）。"""
        if not docs:
            return None
        best = None
        best_score = -1
        for doc in docs:
            content = doc.get("content", "")
            score = sum(1 for word in claim if word in content)
            if score > best_score:
                best_score = score
                best = doc
        return best

    def to_json(self, output: GuardOutput) -> dict:
        """序列化为 JSON 格式。"""
        return {
            "safe_answer": output.safe_answer,
            "unsupported_spans": [
                {"text": s.text, "reason": s.reason,
                 "statement_index": s.statement_index}
                for s in output.unsupported_spans
            ],
            "citations": [
                {"statement_index": c.statement_index,
                 "source_doc_id": c.source_doc_id,
                 "highlight_text": c.highlight_text}
                for c in output.citations
            ],
            "all_hallucination": output.all_hallucination,
            "stats": {
                "total": output.total_statements,
                "supported": output.supported_count,
                "hallucination": output.hallucination_count,
                "latency_ms": output.detection_latency_ms,
            },
        }
