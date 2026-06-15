"""关键指标存储 — 内存环形缓冲。"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MetricPoint:
    timestamp: float
    metric_name: str
    value: float
    tags: dict = field(default_factory=dict)


class MetricsStore:
    """指标收集器。

    记录:
    - 端到端延迟
    - 每个模块的延迟和成功率
    - 幻觉检测触发率
    - 安全拦截率
    """

    def __init__(self, window_minutes: int = 15, max_points: int = 10000):
        self._points: deque[MetricPoint] = deque(maxlen=max_points)
        self._window_seconds = window_minutes * 60
        self._counters: dict[str, int] = {}

    def record(self, metric_name: str, value: float,
               tags: dict = None) -> None:
        self._points.append(MetricPoint(
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            tags=tags or {},
        ))

    def record_latency(self, module: str, latency_ms: float) -> None:
        self.record("latency_ms", latency_ms, {"module": module})

    def record_success(self, module: str) -> None:
        key = f"success_{module}"
        self._counters[key] = self._counters.get(key, 0) + 1
        total_key = f"total_{module}"
        self._counters[total_key] = self._counters.get(total_key, 0) + 1

    def record_error(self, module: str) -> None:
        key = f"total_{module}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_hallucination(self, verdict: str) -> None:
        self.record("hallucination", 1.0 if verdict == "reject" else 0.0,
                    {"verdict": verdict})

    def record_safety(self, blocked: bool) -> None:
        self.record("safety_block", 1.0 if blocked else 0.0)

    def get_recent(self, minutes: int = None) -> list[MetricPoint]:
        window = (minutes or 15) * 60
        cutoff = time.time() - window
        return [p for p in self._points if p.timestamp >= cutoff]

    def get_success_rate(self, module: str) -> float:
        total = self._counters.get(f"total_{module}", 0)
        if total == 0:
            return 1.0
        success = self._counters.get(f"success_{module}", 0)
        return success / total

    def get_avg_latency(self, module: str = None,
                        minutes: int = 15) -> float:
        recent = self.get_recent(minutes)
        points = [p for p in recent if p.metric_name == "latency_ms"]
        if module:
            points = [p for p in points
                     if p.tags.get("module") == module]
        if not points:
            return 0.0
        return sum(p.value for p in points) / len(points)

    def get_hallucination_rate(self, minutes: int = 15) -> float:
        recent = self.get_recent(minutes)
        h_points = [p for p in recent
                    if p.metric_name == "hallucination"]
        if not h_points:
            return 0.0
        return sum(p.value for p in h_points) / len(h_points)

    def get_safety_block_rate(self, minutes: int = 15) -> float:
        recent = self.get_recent(minutes)
        s_points = [p for p in recent
                    if p.metric_name == "safety_block"]
        if not s_points:
            return 0.0
        return sum(p.value for p in s_points) / len(s_points)

    def snapshot(self, minutes: int = 15) -> dict:
        return {
            "period_minutes": minutes,
            "total_requests": len([p for p in self.get_recent(minutes)
                                   if p.metric_name == "latency_ms"
                                   and p.tags.get("module") == "e2e"]),
            "avg_latency_ms": {
                "e2e": round(self.get_avg_latency("e2e", minutes), 2),
                "retriever": round(self.get_avg_latency("retriever", minutes), 2),
                "generator": round(self.get_avg_latency("generator", minutes), 2),
                "guard": round(self.get_avg_latency("guard", minutes), 2),
                "safety": round(self.get_avg_latency("safety", minutes), 2),
            },
            "success_rate": {
                "retriever": round(self.get_success_rate("retriever"), 4),
                "generator": round(self.get_success_rate("generator"), 4),
            },
            "hallucination_rate": round(self.get_hallucination_rate(minutes), 4),
            "safety_block_rate": round(self.get_safety_block_rate(minutes), 4),
        }
