"""融合算法：Reciprocal Rank Fusion (RRF) 和加权分数融合。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


class FusionStrategy(ABC):
    """融合策略抽象基类。"""

    @abstractmethod
    def fuse(self, bm25_results: list[dict],
             vector_results: list[dict]) -> list[dict]:
        """融合两路检索结果。返回按融合分数降序排列的列表。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---- RRF 融合 ----

class RRFusion(FusionStrategy):
    """Reciprocal Rank Fusion: RRF_score(d) = sum(1/(k + rank_i(d)))"""

    def __init__(self, k: int = 60):
        self.k = k

    @property
    def name(self) -> str:
        return f"rrf_k{self.k}"

    def fuse(self, bm25_results: list[dict],
             vector_results: list[dict]) -> list[dict]:
        """RRF 融合。"""
        scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # BM25 路：排名从 1 开始
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            rrf_score = 1.0 / (self.k + rank)
            scores[cid] = rrf_score
            chunk_data[cid] = {
                **item,
                "fusion_score": rrf_score,
                "sources": ["bm25"],
                "rrf_rank_bm25": rank,
            }

        # Vector 路：排名从 1 开始
        for rank, item in enumerate(vector_results, start=1):
            cid = item["chunk_id"]
            rrf_score = 1.0 / (self.k + rank)
            if cid in scores:
                scores[cid] += rrf_score
                chunk_data[cid]["fusion_score"] = scores[cid]
                chunk_data[cid]["sources"].append("vector")
                chunk_data[cid]["rrf_rank_vector"] = rank
            else:
                scores[cid] = rrf_score
                chunk_data[cid] = {
                    **item,
                    "fusion_score": rrf_score,
                    "sources": ["vector"],
                    "rrf_rank_vector": rank,
                }

        # 按融合分数降序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_data[cid] for cid, _ in ranked]


# ---- 加权分数融合 ----

class WeightedFusion(FusionStrategy):
    """加权分数融合：先 min-max 归一化，再加权求和。

    fused_score = alpha * norm_vector + (1-alpha) * norm_bm25
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: 向量检索权重 (0-1)。alpha=1 全向量，alpha=0 全 BM25。
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"weighted_a{self.alpha}"

    def fuse(self, bm25_results: list[dict],
             vector_results: list[dict]) -> list[dict]:
        """加权分数融合。"""

        # 1. 提取原始分数
        bm25_scores = [r["score"] for r in bm25_results]
        vector_scores = [r["score"] for r in vector_results]

        # 2. Min-Max 归一化
        bm25_norm = self._min_max_normalize(bm25_scores)
        vector_norm = self._min_max_normalize(vector_scores)

        # 3. 加权融合
        fused: dict[str, dict] = {}

        for i, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            ns = vector_norm[i] if i < len(vector_norm) else 0
            fused[cid] = {
                **item,
                "fusion_score": (1 - self.alpha) * bm25_norm[i],
                "sources": ["bm25"],
            }

        for i, item in enumerate(vector_results):
            cid = item["chunk_id"]
            if cid in fused:
                fused[cid]["fusion_score"] += self.alpha * vector_norm[i]
                fused[cid]["sources"].append("vector")
            else:
                fused[cid] = {
                    **item,
                    "fusion_score": self.alpha * vector_norm[i],
                    "sources": ["vector"],
                }

        # 4. 排序
        ranked = sorted(fused.values(),
                       key=lambda x: x["fusion_score"], reverse=True)
        return ranked

    @staticmethod
    def _min_max_normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
