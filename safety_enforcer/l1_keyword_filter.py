"""L1: 基于正则和关键词的快速过滤 (<1ms)。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import SensitiveCategory


@dataclass
class L1Result:
    """L1 检测结果。"""
    blocked: bool
    matched_category: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    matched_regex: list[str] = field(default_factory=list)
    latency_us: float = 0.0
    action: str = ""


class L1KeywordFilter:
    """L1 快速过滤器。

    先检查 critical 级类别（正则+关键词），命中即拦截。
    再检查 high 级别。
    使用 AC 自动机加速多关键词匹配。
    """

    def __init__(self, categories: dict[str, SensitiveCategory]):
        self.categories = categories
        # 按风险等级排序：critical > high > medium > low
        self._priority = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def check(self, query: str) -> L1Result:
        """执行 L1 检查。"""
        t0 = time.perf_counter()

        # 按优先级排序类别
        sorted_cats = sorted(
            self.categories.values(),
            key=lambda c: self._priority.get(c.risk_level, 99),
        )

        for cat in sorted_cats:
            # 1. 关键词匹配
            kw_hits = cat.match_keywords(query)
            if kw_hits:
                latency = (time.perf_counter() - t0) * 1_000_000  # us
                return L1Result(
                    blocked=True,
                    matched_category=cat.name,
                    matched_keywords=kw_hits,
                    latency_us=round(latency, 1),
                    action=cat.action,
                )

            # 2. 正则匹配
            re_hits = cat.match_regex(query)
            if re_hits:
                latency = (time.perf_counter() - t0) * 1_000_000
                return L1Result(
                    blocked=True,
                    matched_category=cat.name,
                    matched_regex=re_hits,
                    latency_us=round(latency, 1),
                    action=cat.action,
                )

        latency = (time.perf_counter() - t0) * 1_000_000
        return L1Result(blocked=False, latency_us=round(latency, 1))
