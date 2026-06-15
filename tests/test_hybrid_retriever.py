"""Hybrid Retriever 模块的 mock 测试。

覆盖:
1. BM25 索引和检索
2. RRF 和 Weighted 融合算法
3. 向量检索降级为纯 BM25 的场景
4. Cross-encoder 重排序
5. 工厂方法动态切换
6. 索引管理器异步同步
7. 性能指标收集
"""

import asyncio
import math
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from hybrid_retriever.bm25_retriever import BM25Retriever, BM25Document
from hybrid_retriever.fusion import RRFusion, WeightedFusion
from hybrid_retriever.metrics import RetrievalMetrics, MetricsCollector
from hybrid_retriever.config import RetrieverConfig


# ============================================================
class TestBM25Retriever:
    """BM25 检索器测试。"""

    def _build_docs(self) -> list[BM25Document]:
        return [
            BM25Document("d1", "c1", "年假政策规定入职满一年的员工享有五天带薪年假", {}),
            BM25Document("d2", "c2", "病假需要提供医院证明并经主管审批", {}),
            BM25Document("d3", "c3", "加班费按照基本工资的1.5倍计算", {}),
            BM25Document("d4", "c4", "绩效评定采用360度评估和OKR相结合", {}),
            BM25Document("d5", "c5", "员工入职满五年享有十天带薪年假", {}),
        ]

    def test_basic_index_and_search(self):
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        bm25.add_batch(self._build_docs())

        results = bm25.search("年假有几天？", top_k=3)
        assert len(results) >= 1
        # 年假相关文档应排在前面
        assert any("年假" in r["content"] for r in results)
        assert all(r["source"] == "bm25" for r in results)

    def test_empty_index(self):
        bm25 = BM25Retriever()
        results = bm25.search("测试", top_k=10)
        assert results == []

    def test_remove_document(self):
        bm25 = BM25Retriever()
        docs = self._build_docs()
        bm25.add_batch(docs)
        assert len(bm25) == 5

        bm25.remove_document("d1")
        assert len(bm25) == 4
        results = bm25.search("年假", top_k=5)
        # d5 还在，但 d1 已删除
        assert all(r["chunk_id"] != "c1" for r in results)

    def test_clear_and_rebuild(self):
        bm25 = BM25Retriever()
        bm25.add_batch(self._build_docs())
        assert len(bm25) == 5

        bm25.clear()
        assert len(bm25) == 0
        assert bm25.search("年假") == []

        # 重建
        bm25.add_document(BM25Document("d6", "c6", "新的文档内容", {}))
        results = bm25.search("新的文档")
        assert len(results) == 1

    def test_version_tracking(self):
        bm25 = BM25Retriever()
        v1 = bm25.get_version()
        bm25.add_document(BM25Document("d1", "c1", "test", {}))
        bm25.search("test")  # 触发 rebuild
        v2 = bm25.get_version()
        assert v2 > v1

        bm25.add_document(BM25Document("d2", "c2", "test2", {}))
        bm25.search("test2")
        v3 = bm25.get_version()
        assert v3 > v2


# ============================================================
class TestFusionAlgorithms:
    """融合算法测试。"""

    _bm25_results = [
        {"chunk_id": "c1", "content": "doc A", "score": 3.5, "source": "bm25"},
        {"chunk_id": "c2", "content": "doc B", "score": 2.1, "source": "bm25"},
        {"chunk_id": "c5", "content": "doc E", "score": 0.8, "source": "bm25"},
    ]
    _vec_results = [
        {"chunk_id": "c3", "content": "doc C", "score": 0.95, "source": "vector"},
        {"chunk_id": "c2", "content": "doc B", "score": 0.88, "source": "vector"},
        {"chunk_id": "c4", "content": "doc D", "score": 0.72, "source": "vector"},
    ]

    def test_rrf_fusion(self):
        fusion = RRFusion(k=60)
        results = fusion.fuse(self._bm25_results, self._vec_results)

        assert len(results) == 5  # c1, c2(合并), c3, c4, c5
        # 检查 c2 被两路命中
        c2 = [r for r in results if r["chunk_id"] == "c2"]
        assert len(c2) == 1
        assert sorted(c2[0]["sources"]) == ["bm25", "vector"]

        # 检查融合分数降序
        scores = [r["fusion_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_single_source(self):
        fusion = RRFusion(k=60)
        results = fusion.fuse(self._bm25_results, [])
        assert len(results) == 3
        assert all(r["source"] == "bm25" for r in results)

    def test_weighted_fusion(self):
        fusion = WeightedFusion(alpha=0.5)
        results = fusion.fuse(self._bm25_results, self._vec_results)

        assert len(results) == 5
        scores = [r["fusion_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        # 分数应在 [0, 1] 区间
        assert 0 <= max(scores) <= 1

    def test_weighted_alpha_extreme(self):
        # alpha=1: 纯向量
        fusion = WeightedFusion(alpha=1.0)
        results = fusion.fuse(self._bm25_results, self._vec_results)
        # 向量结果应排在前面
        top = results[0]
        assert top["chunk_id"] in ["c3", "c2", "c4"]

    def test_weighted_alpha_zero(self):
        # alpha=0: 纯 BM25
        fusion = WeightedFusion(alpha=0.0)
        results = fusion.fuse(self._bm25_results, self._vec_results)
        # BM25 结果应排在前面
        top = results[0]
        assert top["chunk_id"] in ["c1", "c2", "c5"]


# ============================================================
class TestMetrics:
    """指标收集器测试。"""

    def test_single_metrics(self):
        m = RetrievalMetrics(
            total_latency_ms=150.0,
            bm25_results_count=20,
            vector_results_count=18,
            fused_results_count=25,
            fusion_score_mean=0.045,
            rerank_score_mean=0.72,
        )
        d = m.to_dict()
        assert d["total_latency_ms"] == 150.0
        assert d["bm25_count"] == 20
        assert d["fusion_score_mean"] == 0.045

    def test_collector_snapshot(self):
        collector = MetricsCollector()
        for _ in range(10):
            collector.record(RetrievalMetrics(
                total_latency_ms=100,
                bm25_results_count=15,
                vector_results_count=15,
            ))
        collector.record(RetrievalMetrics(
            total_latency_ms=100, bm25_results_count=0,
            vector_results_count=0, bm25_fallback=True, error="test",
        ))

        snap = collector.snapshot()
        assert snap["calls"] == 11
        assert snap["avg_latency_ms"] == 100.0
        assert snap["bm25_fallback_rate"] == pytest.approx(1/11, abs=0.01)
        assert snap["error_rate"] == pytest.approx(1/11, abs=0.01)

    def test_collector_empty(self):
        collector = MetricsCollector()
        assert collector.snapshot() == {"calls": 0}


# ============================================================
class TestVectorFallback:
    """向量检索降级为纯 BM25 + 重排序的场景。"""

    def _build_bm25_index(self, bm25: BM25Retriever):
        docs = [
            ("d1", "c1", "员工手册第三章规定，入职满1年享有5天带薪年假"),
            ("d2", "c2", "年假应于当年12月31日前使用完毕"),
            ("d3", "c3", "病假需要提供医院开具的病假证明"),
            ("d4", "c4", "加班申请需经部门主管审批"),
            ("d5", "c5", "绩效评定标准分为S/A/B/C四个等级"),
            ("d6", "c6", "年假未使用的最多可结转5天至次年3月"),
            ("d7", "c7", "员工培训由HR部门统一安排"),
            ("d8", "c8", "差旅报销需要提供发票和行程单"),
            ("d9", "c9", "2024年年假政策有所调整"),
            ("d10", "c10", "带薪年假的天数按工龄计算"),
        ]
        for args in docs:
            bm25.add_document(BM25Document(*args, {}))

    @pytest.mark.asyncio
    async def test_vector_empty_fallback_to_bm25(self):
        """向量检索返回空 -> 纯 BM25 + 重排序（mock），结果不为空。"""
        from hybrid_retriever.hybrid_retriever import HybridRetriever
        from hybrid_retriever.fusion import RRFusion

        bm25 = BM25Retriever()
        self._build_bm25_index(bm25)

        # Mock 向量检索器：始终返回空
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        mock_vector.get_version.return_value = 1

        # Mock reranker: 透传原始结果并加上 rerank_score
        mock_reranker = MagicMock()
        def fake_rerank(query, candidates):
            for c in candidates:
                c["rerank_score"] = c.get("fusion_score", 0)
            return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        mock_reranker.rerank = fake_rerank

        config = RetrieverConfig(fusion_strategy="rrf", final_top_k=5)
        retriever = HybridRetriever(
            bm25=bm25,
            vector=mock_vector,
            fusion=RRFusion(k=60),
            reranker=mock_reranker,
            config=config,
        )

        results = await retriever.search("年假天数", top_k=5)

        # 验证: 结果不为空
        assert len(results) > 0
        assert len(results) <= 5

        # 验证: 所有结果都有 rerank_score
        assert all("rerank_score" in r for r in results)

        # 验证: 结果与查询相关（含"年假"）
        assert any("年假" in r["content"] for r in results)

        # 验证指标记录了降级
        snap = retriever.get_metrics()
        assert snap["calls"] == 1
        assert snap["vector_fallback_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty(self):
        """BM25 和向量都返回空 -> 结果为空。"""
        from hybrid_retriever.hybrid_retriever import HybridRetriever
        from hybrid_retriever.fusion import RRFusion

        bm25 = BM25Retriever()  # 空索引

        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        mock_vector.get_version.return_value = 1

        config = RetrieverConfig()
        retriever = HybridRetriever(
            bm25=bm25, vector=mock_vector,
            fusion=RRFusion(), config=config,
        )

        results = await retriever.search("不存在的内容")
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_exception_fallback(self):
        """向量检索抛异常 -> 纯 BM25 降级。"""
        from hybrid_retriever.hybrid_retriever import HybridRetriever
        from hybrid_retriever.fusion import RRFusion

        bm25 = BM25Retriever()
        self._build_bm25_index(bm25)

        mock_vector = MagicMock()
        mock_vector.search.side_effect = RuntimeError("ChromaDB unavailable")
        mock_vector.get_version.return_value = 1

        config = RetrieverConfig()
        retriever = HybridRetriever(
            bm25=bm25, vector=mock_vector,
            fusion=RRFusion(), config=config,
        )

        results = await retriever.search("年假天数")

        # 验证: 降级后仍有结果
        assert len(results) > 0
        assert any("年假" in r["content"] for r in results)

        # 验证: sources 不包含 vector
        for r in results:
            assert "vector" not in r.get("sources", [])

        # 验证: fallback 指标
        snap = retriever.get_metrics()
        assert snap["vector_fallback_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_metrics_latency_tracking(self):
        """验证所有耗时指标都被记录。"""
        from hybrid_retriever.hybrid_retriever import HybridRetriever
        from hybrid_retriever.fusion import RRFusion

        bm25 = BM25Retriever()
        self._build_bm25_index(bm25)

        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        mock_vector.get_version.return_value = 1

        # 添加可测延迟
        import time

        config = RetrieverConfig()
        retriever = HybridRetriever(
            bm25=bm25, vector=mock_vector, fusion=RRFusion(), config=config,
        )

        # 重置 collector 以获取最新一条
        collector = MetricsCollector()
        retriever._metrics_collector = collector

        await retriever.search("年假")

        snap = collector.snapshot()
        assert snap["calls"] == 1
        assert snap["avg_latency_ms"] > 0


# ============================================================
class TestFactoryMethod:
    """工厂方法测试。"""

    def test_create_with_rrf(self):
        from hybrid_retriever.hybrid_retriever import create_hybrid_retriever
        retriever = create_hybrid_retriever(
            fusion_strategy="rrf", rrf_k=120
        )
        assert retriever._fusion.name == "rrf_k120"

    def test_create_with_weighted(self):
        from hybrid_retriever.hybrid_retriever import create_hybrid_retriever
        retriever = create_hybrid_retriever(
            fusion_strategy="weighted", weighted_alpha=0.7
        )
        assert retriever._fusion.name == "weighted_a0.7"
        assert isinstance(retriever._fusion, WeightedFusion)

    def test_create_with_custom_bm25_params(self):
        from hybrid_retriever.hybrid_retriever import create_hybrid_retriever
        retriever = create_hybrid_retriever(
            fusion_strategy="rrf", bm25_k1=2.0, bm25_b=0.5
        )
        # BM25 参数已应用
        assert retriever._bm25.k1 == 2.0
        assert retriever._bm25.b == 0.5


# ============================================================
class TestIndexManager:
    """索引管理器异步同步测试。"""

    @pytest.mark.asyncio
    async def test_enqueue_and_sync(self):
        from hybrid_retriever.index_manager import IndexManager

        bm25 = BM25Retriever()
        mock_vector = MagicMock()
        mock_vector.add_batch = MagicMock()
        mock_vector.remove_document = MagicMock()
        mock_vector.get_version.return_value = 0

        manager = IndexManager(bm25, mock_vector, sync_interval=0.1)

        # 入队文档
        manager.enqueue_add("d1", "c1", "测试文档内容", version=1)
        manager.enqueue_add("d2", "c2", "另一篇文档", version=2)

        assert manager.pending_count() == 2

        # 同步
        count = await manager.sync_now()
        assert count == 2
        assert manager.pending_count() == 0
        assert len(bm25) == 2
        mock_vector.add_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_idempotent_version(self):
        from hybrid_retriever.index_manager import IndexManager

        bm25 = BM25Retriever()
        mock_vector = MagicMock()
        mock_vector.add_batch = MagicMock()
        mock_vector.get_version.return_value = 0

        manager = IndexManager(bm25, mock_vector)

        # 同版本重复添加
        manager.enqueue_add("d1", "c1", "内容A", version=1)
        manager.enqueue_add("d1", "c1", "内容A", version=1)  # 重复，应被忽略

        assert manager.pending_count() == 1  # 只有 1 个

        await manager.sync_now()
        assert len(bm25) == 1

    @pytest.mark.asyncio
    async def test_status_tracking(self):
        from hybrid_retriever.index_manager import IndexManager, IndexStatus

        bm25 = BM25Retriever()
        mock_vector = MagicMock()
        mock_vector.add_batch = MagicMock()
        mock_vector.get_version.return_value = 0

        manager = IndexManager(bm25, mock_vector)
        assert manager.get_status() == IndexStatus.IDLE

        manager.enqueue_add("d1", "c1", "test", version=1)
        await manager.sync_now()
        assert manager.get_status() == IndexStatus.IDLE

        # 版本信息
        versions = manager.get_versions()
        assert versions["status"] == "idle"
        assert versions["pending"] == 0

    @pytest.mark.asyncio
    async def test_delete_document(self):
        from hybrid_retriever.index_manager import IndexManager

        bm25 = BM25Retriever()
        bm25.add_document(BM25Document("d1", "c1", "待删除", {}))
        assert len(bm25) == 1

        mock_vector = MagicMock()
        mock_vector.add_batch = MagicMock()
        mock_vector.remove_document = MagicMock()
        mock_vector.get_version.return_value = 0

        manager = IndexManager(bm25, mock_vector)
        manager._chunk_to_doc["c1"] = "d1"
        manager.enqueue_delete("c1", doc_id="d1", version=1)
        await manager.sync_now()

        assert len(bm25) == 0
        mock_vector.remove_document.assert_called_once_with("c1")
