"""检索性能指标收集。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RetrievalMetrics:
    """单次检索的完整指标。"""

    # 时间指标 (ms)
    total_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0

    # 数量指标
    bm25_results_count: int = 0
    vector_results_count: int = 0
    fused_results_count: int = 0
    final_results_count: int = 0

    # 分数分布
    fusion_score_min: float = 0.0
    fusion_score_max: float = 0.0
    fusion_score_mean: float = 0.0
    rerank_score_min: float = 0.0
    rerank_score_max: float = 0.0
    rerank_score_mean: float = 0.0

    # 降级信息
    bm25_fallback: bool = False
    vector_fallback: bool = False
    rerank_fallback: bool = False
    error: str = ""

    # 版本信息
    bm25_version: int = 0
    vector_version: int = 0
    fusion_strategy: str = ""
    query: str = ""

    def to_dict(self) -> dict:
        return {
            "total_latency_ms": round(self.total_latency_ms, 2),
            "bm25_latency_ms": round(self.bm25_latency_ms, 2),
            "vector_latency_ms": round(self.vector_latency_ms, 2),
            "fusion_latency_ms": round(self.fusion_latency_ms, 2),
            "rerank_latency_ms": round(self.rerank_latency_ms, 2),
            "bm25_count": self.bm25_results_count,
            "vector_count": self.vector_results_count,
            "fused_count": self.fused_results_count,
            "final_count": self.final_results_count,
            "fusion_score_range": [
                round(self.fusion_score_min, 4),
                round(self.fusion_score_max, 4),
            ],
            "fusion_score_mean": round(self.fusion_score_mean, 4),
            "rerank_score_range": [
                round(self.rerank_score_min, 4),
                round(self.rerank_score_max, 4),
            ],
            "rerank_score_mean": round(self.rerank_score_mean, 4),
            "fallback": {
                "bm25": self.bm25_fallback,
                "vector": self.vector_fallback,
                "rerank": self.rerank_fallback,
            },
            "error": self.error,
        }


class MetricsCollector:
    """累积统计指标收集器（用于监控聚合）。"""

    def __init__(self):
        self._total_calls: int = 0
        self._total_latency: float = 0.0
        self._total_bm25_count: int = 0
        self._total_vector_count: int = 0
        self._bm25_fallback_count: int = 0
        self._vector_fallback_count: int = 0
        self._errors: list[str] = []

    def record(self, metrics: RetrievalMetrics) -> None:
        self._total_calls += 1
        self._total_latency += metrics.total_latency_ms
        self._total_bm25_count += metrics.bm25_results_count
        self._total_vector_count += metrics.vector_results_count
        if metrics.bm25_fallback:
            self._bm25_fallback_count += 1
        if metrics.vector_fallback:
            self._vector_fallback_count += 1
        if metrics.error:
            self._errors.append(metrics.error)

    def snapshot(self) -> dict:
        if self._total_calls == 0:
            return {"calls": 0}
        return {
            "calls": self._total_calls,
            "avg_latency_ms": round(self._total_latency / self._total_calls, 2),
            "avg_bm25_count": round(self._total_bm25_count / self._total_calls, 2),
            "avg_vector_count": round(self._total_vector_count / self._total_calls, 2),
            "bm25_fallback_rate": round(self._bm25_fallback_count / self._total_calls, 4),
            "vector_fallback_rate": round(self._vector_fallback_count / self._total_calls, 4),
            "error_rate": round(len(self._errors) / self._total_calls, 4),
        }

    def reset(self) -> None:
        self.__init__()
