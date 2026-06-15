"""Observability 模块的完整测试。"""

import json
import tempfile
import os
import time
import pytest
from unittest.mock import patch

from observability.trace_context import TraceContext, get_current_trace, set_current_trace
from observability.config import ObsConfig
from observability.metrics_store import MetricsStore
from observability.tracer import Tracer, init_tracer, span
from observability.alerting import AlertEngine, AlertRule
from observability.dashboard import Dashboard


# ============================================================
class TestTraceContext:
    def test_basic_trace(self):
        ctx = TraceContext(service_name="test")
        span1 = ctx.start_span("retriever.search", "retriever",
                               {"top_k": 20})
        ctx.end_span("OK")
        ctx.finish()

        d = ctx.to_dict()
        assert d["span_count"] == 1
        assert d["service"] == "test"
        assert len(ctx.trace_id) == 16
        assert ctx.duration_ms >= 0

    def test_nested_spans(self):
        ctx = TraceContext(service_name="test")
        ctx.start_span("e2e", "e2e")
        ctx.start_span("retriever.search", "retriever")
        ctx.end_span("OK")
        ctx.start_span("generator.generate", "generator")
        ctx.end_span("OK")
        ctx.end_span("OK")
        ctx.finish()

        d = ctx.to_dict()
        assert d["span_count"] == 3
        spans = d["spans"]
        assert spans[0]["name"] == "e2e"
        assert spans[1]["parent"] == spans[0]["span_id"]

    def test_contextvars_propagation(self):
        ctx = TraceContext(service_name="test")
        set_current_trace(ctx)

        retrieved = get_current_trace()
        assert retrieved is ctx
        assert retrieved.trace_id == ctx.trace_id

    def test_attributes(self):
        ctx = TraceContext()
        ctx.start_span("search", "retriever",
                       {"top_k": 20, "model": "bge-m3"})
        ctx.end_span("OK")
        ctx.finish()

        attrs = ctx.spans[0].attributes
        assert attrs["top_k"] == 20
        assert attrs["model"] == "bge-m3"


# ============================================================
class TestTracer:
    def test_start_and_finish(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        config = ObsConfig(json_export_path=path, exporter_type="json")
        tracer = Tracer(config)
        ctx = tracer.start_trace()
        tracer.start_span("retriever.search", "retriever")
        tracer.end_span("OK")
        result = tracer.finish_trace()

        assert result is not None
        assert result["span_count"] == 1

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        exported = json.loads(lines[0])
        assert exported["trace_id"] == ctx.trace_id

        os.unlink(path)

    def test_span_context_manager(self):
        config = ObsConfig(exporter_type="console")
        init_tracer(config)
        t = get_current_trace()

        if t:
            t.start_span("test", "test")
            t.end_span("OK")
            assert any(s.name == "test" for s in t.spans)

    def test_module_span(self):
        from observability.tracer import module_span
        with module_span("retriever", {"top_k": 10}):
            pass
        # Should not raise


# ============================================================
class TestMetricsStore:
    @pytest.fixture
    def store(self):
        return MetricsStore(window_minutes=15)

    def test_record_latency(self, store):
        store.record_latency("retriever", 150.0)
        store.record_latency("retriever", 200.0)
        store.record_latency("generator", 1000.0)

        avg_ret = store.get_avg_latency("retriever", minutes=30)
        assert avg_ret == pytest.approx(175.0, abs=1)
        avg_gen = store.get_avg_latency("generator", minutes=30)
        assert avg_gen == 1000.0

    def test_success_rate(self, store):
        store.record_success("retriever")
        store.record_success("retriever")
        store.record_error("retriever")

        rate = store.get_success_rate("retriever")
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_hallucination_rate(self, store):
        store.record_hallucination("reject")
        store.record_hallucination("pass")
        store.record_hallucination("reject")

        rate = store.get_hallucination_rate(minutes=30)
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_snapshot(self, store):
        store.record_latency("e2e", 1000)
        store.record_latency("retriever", 150)
        store.record_latency("generator", 800)
        store.record_hallucination("reject")
        store.record_safety(True)

        snap = store.snapshot(minutes=30)
        assert "hallucination_rate" in snap
        assert "safety_block_rate" in snap
        assert "avg_latency_ms" in snap
        assert snap["avg_latency_ms"]["e2e"] == 1000.0


# ============================================================
class TestAlertEngine:
    def test_rule_evaluation(self):
        store = MetricsStore()
        engine = AlertEngine(store)
        engine.add_rule(AlertRule(
            "high_hallucination", "hallucination_rate",
            "> 0.2", 60, "WARNING",
            "Hallucination: {value:.2%}"
        ))

        # 添加数据: 幻觉率 > 0.2
        for _ in range(5):
            store.record_hallucination("reject")
        for _ in range(15):
            store.record_hallucination("pass")

        # 首次评估: 进入 breach 但不告警 (duration < 60s)
        alerts = engine.evaluate()
        assert len(alerts) == 0  # just breached, not sustained

        # 模拟持续时间
        engine._breach_times["high_hallucination"] = time.time() - 61
        alerts = engine.evaluate()
        assert len(alerts) == 1
        assert alerts[0].severity == "WARNING"

    def test_no_breach(self):
        store = MetricsStore()
        engine = AlertEngine(store)
        engine.add_rule(AlertRule(
            "low_hallucination", "hallucination_rate",
            "> 0.9", 10, "WARNING"
        ))
        for _ in range(20):
            store.record_hallucination("pass")

        # 模拟持续
        engine._breach_times["low_hallucination"] = time.time() - 15
        alerts = engine.evaluate()
        assert len(alerts) == 0

    def test_alert_reset(self):
        store = MetricsStore()
        engine = AlertEngine(store)
        engine.add_rule(AlertRule(
            "test_rule", "hallucination_rate", "> 0.5", 10, "INFO"
        ))
        store.record_hallucination("reject")
        store.record_hallucination("reject")
        store.record_hallucination("pass")

        engine._breach_times["test_rule"] = time.time() - 15
        alerts = engine.evaluate()
        assert len(alerts) == 1

        engine.reset()
        assert len(engine._breach_times) == 0


# ============================================================
class TestDashboard:
    def test_generate_report(self):
        store = MetricsStore()
        store.record_latency("e2e", 1200)
        store.record_latency("retriever", 150)
        store.record_latency("generator", 900)
        store.record_hallucination("pass")

        dash = Dashboard(store, default_minutes=15)
        report = dash.generate(minutes=30)
        text = report.to_text()

        assert "Health Dashboard" in text
        assert "1200.0" in text or "1200" in text
        assert "Hallucination Rate" in text

    def test_empty_dashboard(self):
        store = MetricsStore()
        dash = Dashboard(store)
        report = dash.generate(minutes=5)
        text = report.to_text()
        assert "Requests: 0" in text


# ============================================================
class TestFullRequestSimulation:
    """模拟一次完整请求: 验证所有 Span 正确生成。"""

    def test_complete_request_trace(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        config = ObsConfig(json_export_path=path, exporter_type="json")
        tracer = Tracer(config)
        store = MetricsStore()

        # 模拟完整的 RAG 请求
        ctx = tracer.start_trace()

        # Safety Enforcer
        tracer.start_span("safety.check", "safety", {"query": "年假有几天？"})
        store.record_latency("safety", 2.5)
        store.record_safety(False)
        tracer.end_span("OK")

        # Query Rewriter
        tracer.start_span("query.rewrite", "rewriter", {"strategy": "passthrough"})
        tracer.end_span("OK")

        # Hybrid Retriever
        tracer.start_span("retrieve.hybrid", "retriever",
                         {"top_k": 20, "fusion": "rrf"})
        store.record_latency("retriever", 120.0)
        store.record_success("retriever")
        tracer.end_span("OK")

        # Context Compressor
        tracer.start_span("context.compress", "compressor", {"method": "extractive"})
        tracer.end_span("OK")

        # Generator
        tracer.start_span("generator.generate", "generator",
                         {"model": "deepseek-v4"})
        store.record_latency("generator", 850.0)
        store.record_success("generator")
        tracer.end_span("OK")

        # Hallucination Guard
        tracer.start_span("guard.verify", "guard")
        store.record_latency("guard", 200.0)
        store.record_hallucination("pass")
        tracer.end_span("OK")

        # 完成
        store.record_latency("e2e", 1500.0)
        result = tracer.finish_trace()

        # 验证
        assert result is not None
        assert result["span_count"] == 6
        assert result["duration_ms"] >= 0

        # 验证 Span 层级
        spans = result["spans"]
        module_names = [s["module"] for s in spans]
        assert "safety" in module_names
        assert "rewriter" in module_names
        assert "retriever" in module_names
        assert "generator" in module_names
        assert "guard" in module_names
        assert "compressor" in module_names

        # 验证属性
        retriever_span = [s for s in spans if s["module"] == "retriever"][0]
        assert retriever_span["attributes"]["top_k"] == 20
        assert retriever_span["attributes"]["fusion"] == "rrf"

        # 验证 JSON 导出
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        exported = json.loads(lines[0])
        assert exported["span_count"] == 6

        # 验证指标
        snap = store.snapshot(minutes=30)
        assert snap["avg_latency_ms"]["retriever"] == 120.0
        assert snap["avg_latency_ms"]["generator"] == 850.0
        assert snap["hallucination_rate"] == 0.0

        os.unlink(path)
