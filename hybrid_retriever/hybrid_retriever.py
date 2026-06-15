"""Hybrid Retriever 主编排器。集成 BM25 + 向量检索 + 融合 + 重排序。"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .config import RetrieverConfig, load_retriever_config
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever
from .fusion import RRFusion, WeightedFusion, FusionStrategy
from .reranker import CrossEncoderReranker
from .metrics import RetrievalMetrics, MetricsCollector
from .index_manager import IndexManager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索引擎。

    流程: BM25 + Vector 并行检索 -> 融合(RRF/Weighted) -> Cross-encoder 重排序。

    工厂方法 create_hybrid_retriever() 支持动态切换融合策略和重排序模型。
    """

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever,
                 fusion: FusionStrategy, reranker: CrossEncoderReranker = None,
                 config: RetrieverConfig = None,
                 metrics_collector: MetricsCollector = None):
        self._bm25 = bm25
        self._vector = vector
        self._fusion = fusion
        self._reranker = reranker
        self._config = config or load_retriever_config()
        self._metrics_collector = metrics_collector or MetricsCollector()

    # ---- 检索 ----

    async def search(self, query: str,
                     top_k: int = None,
                     metadata_filter: dict = None) -> list[dict]:
        """执行混合检索。

        Args:
            query: 查询文本
            top_k: 最终返回数量（默认从 config 读取）
            metadata_filter: 元数据过滤条件（传给向量检索）

        Returns:
            检索结果列表 [{chunk_id, content, score, rerank_score, sources, ...}]
        """
        top_k = top_k or self._config.final_top_k
        metrics = RetrievalMetrics(query=query[:200])
        t_start = time.perf_counter()

        # 1. 并行 BM25 + Vector
        t0 = time.perf_counter()
        bm25_task = asyncio.to_thread(self._bm25.search, query,
                                       self._config.top_k_per_path)
        vector_task = asyncio.to_thread(self._vector.search, query,
                                         self._config.top_k_per_path,
                                         metadata_filter)

        bm25_results, vector_results = await asyncio.gather(
            bm25_task, vector_task, return_exceptions=True
        )

        # 降级处理
        if isinstance(bm25_results, Exception):
            logger.warning(f"BM25 search failed: {bm25_results}")
            bm25_results = []
            metrics.bm25_fallback = True
            metrics.error += f"BM25: {bm25_results}; "

        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search failed: {vector_results}")
            vector_results = []
            metrics.vector_fallback = True
            metrics.error += f"Vector: {vector_results}; "

        metrics.bm25_results_count = len(bm25_results)
        metrics.vector_results_count = len(vector_results)
        metrics.bm25_latency_ms = (time.perf_counter() - t0) * 1000
        metrics.vector_latency_ms = metrics.bm25_latency_ms  # 并行，时间相似

        # 2. 结果融合
        t0 = time.perf_counter()

        if not bm25_results and not vector_results:
            metrics.total_latency_ms = (time.perf_counter() - t_start) * 1000
            self._record(metrics)
            return []

        if not bm25_results:
            # 纯向量降级
            fused = self._label_as_fused(vector_results, ["vector"])
            metrics.bm25_fallback = True
        elif not vector_results:
            # 纯 BM25 降级
            fused = self._label_as_fused(bm25_results, ["bm25"])
            metrics.vector_fallback = True
        else:
            fused = self._fusion.fuse(bm25_results, vector_results)

        metrics.fused_results_count = len(fused)
        metrics.fusion_latency_ms = (time.perf_counter() - t0) * 1000

        # 计算融合分数分布
        if fused:
            scores = [c.get("fusion_score", 0) for c in fused]
            metrics.fusion_score_min = min(scores)
            metrics.fusion_score_max = max(scores)
            metrics.fusion_score_mean = sum(scores) / len(scores)

        # 3. 重排序 (裁剪到 fusion_top_k)
        fused_for_rerank = fused[:self._config.fusion_top_k]

        if self._reranker and len(fused_for_rerank) > 1:
            t0 = time.perf_counter()

            try:
                reranked = await asyncio.to_thread(
                    self._reranker.rerank, query, fused_for_rerank
                )
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")
                reranked = fused_for_rerank
                metrics.rerank_fallback = True

            metrics.rerank_latency_ms = (time.perf_counter() - t0) * 1000
        else:
            reranked = fused_for_rerank
            # 没有 reranker 时，用 fusion_score 代替
            for c in reranked:
                c["rerank_score"] = c.get("fusion_score", 0)

        # 计算重排序分数分布
        if reranked:
            scores = [c.get("rerank_score", 0) for c in reranked]
            metrics.rerank_score_min = min(scores)
            metrics.rerank_score_max = max(scores)
            metrics.rerank_score_mean = sum(scores) / len(scores)

        # 4. 最终 Top-K
        final = reranked[:top_k]
        metrics.final_results_count = len(final)

        # 移除中间分数（减小输出体积）
        for c in final:
            c.pop("fusion_score", None)

        metrics.total_latency_ms = (time.perf_counter() - t_start) * 1000
        metrics.fusion_strategy = self._fusion.name
        metrics.bm25_version = self._bm25.get_version()
        metrics.vector_version = self._vector.get_version()

        self._record(metrics)

        return final

    def _label_as_fused(self, results: list[dict],
                        sources: list[str]) -> list[dict]:
        """单路结果标记为融合格式。"""
        for c in results:
            c["fusion_score"] = c.get("score", 0)
            c["sources"] = sources
        results.sort(key=lambda x: x["fusion_score"], reverse=True)
        return results

    def _record(self, metrics: RetrievalMetrics) -> None:
        if self._config.metrics_enabled:
            self._metrics_collector.record(metrics)

    # ---- 信息 ----

    def get_metrics(self) -> dict:
        return self._metrics_collector.snapshot()

    def search_sync(self, query: str, top_k: int = None,
                    metadata_filter: dict = None) -> list[dict]:
        """同步检索接口（方便非 async 调用）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        return loop.run_until_complete(
            self.search(query, top_k, metadata_filter)
        )


# ---- 工厂方法 ----

def create_hybrid_retriever(
    fusion_strategy: str = "rrf",
    reranker_model: str = None,
    embedding_model: str = None,
    bm25_k1: float = None,
    bm25_b: float = None,
    rrf_k: int = None,
    weighted_alpha: float = None,
    config: RetrieverConfig = None,
) -> HybridRetriever:
    """工厂方法：动态切换融合策略和重排序模型，方便 A/B 测试。

    Args:
        fusion_strategy: "rrf" 或 "weighted"
        reranker_model: 重排序模型名（None 则用 config 默认值）
        embedding_model: 嵌入模型名（None 则用 config 默认值）
        bm25_k1: BM25 k1 参数
        bm25_b: BM25 b 参数
        rrf_k: RRF k 值
        weighted_alpha: 加权融合 alpha
        config: 配置对象

    Returns:
        配置好的 HybridRetriever 实例
    """
    cfg = config or load_retriever_config()

    # 融合策略
    if fusion_strategy == "weighted":
        alpha = weighted_alpha if weighted_alpha is not None else cfg.weighted_alpha
        fusion = WeightedFusion(alpha=alpha)
    else:
        k = rrf_k if rrf_k is not None else cfg.rrf_k
        fusion = RRFusion(k=k)

    # BM25
    bm25 = BM25Retriever(
        k1=bm25_k1 if bm25_k1 is not None else cfg.bm25_k1,
        b=bm25_b if bm25_b is not None else cfg.bm25_b,
    )

    # Vector
    vector = VectorRetriever(
        persist_dir=cfg.chroma_persist_dir,
    )
    # 注意：embedding model 的选择需要根据实际情况配置

    # Reranker
    reranker = None
    if reranker_model:
        # 用自定义模型创建临时 config
        temp_cfg = RetrieverConfig(reranker_model=reranker_model)
        reranker = CrossEncoderReranker(temp_cfg)
    else:
        reranker = CrossEncoderReranker(cfg)

    return HybridRetriever(
        bm25=bm25,
        vector=vector,
        fusion=fusion,
        reranker=reranker,
        config=cfg,
    )
