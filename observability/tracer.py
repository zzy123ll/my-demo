"""手动埋点 Tracer — 兼容 OpenTelemetry API 的轻量实现。"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

from .config import ObsConfig
from .trace_context import TraceContext, set_current_trace, get_current_trace


logger = logging.getLogger(__name__)


class Tracer:
    """手动埋点 Tracer。

    在每个模块入口/出口创建 Span，记录属性和耗时。
    导出到控制台或 JSON 文件。
    """

    def __init__(self, config: ObsConfig = None):
        self.config = config or ObsConfig()
        self._export_path = Path(self.config.json_export_path)
        self._export_path.parent.mkdir(parents=True, exist_ok=True)

    def start_trace(self, trace_id: str = "") -> TraceContext:
        ctx = TraceContext(
            trace_id=trace_id,
            service_name=self.config.service_name,
        )
        set_current_trace(ctx)
        return ctx

    def start_span(self, name: str, module: str = "",
                   attributes: dict = None) -> TraceContext:
        ctx = get_current_trace()
        if not ctx:
            ctx = self.start_trace()
        ctx.start_span(name, module, attributes)
        return ctx

    def end_span(self, status: str = "OK") -> None:
        ctx = get_current_trace()
        if ctx:
            ctx.end_span(status=status)

    def finish_trace(self) -> dict | None:
        ctx = get_current_trace()
        if ctx:
            ctx.finish()
            self._export(ctx)
            return ctx.to_dict()
        return None

    def _export(self, ctx: TraceContext) -> None:
        data = ctx.to_dict()

        if self.config.exporter_type == "console":
            logger.info(f"TRACE {ctx.trace_id}: {ctx.duration_ms:.1f}ms, {len(ctx.spans)} spans")
            for s in ctx.spans:
                logger.debug(f"  [{s.module}] {s.name}: {s.duration_ms:.1f}ms {s.status}")

        if self.config.exporter_type in ("json", "console"):
            with open(self._export_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")


# 全局实例
_tracer: Tracer | None = None


def init_tracer(config: ObsConfig = None) -> Tracer:
    global _tracer
    _tracer = Tracer(config)
    return _tracer


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


@contextmanager
def span(name: str, module: str = "", attributes: dict = None):
    """上下文管理器: 自动管理 Span 生命周期。"""
    tracer = get_tracer()
    ctx = tracer.start_span(name, module, attributes)
    try:
        yield ctx
        tracer.end_span("OK")
    except Exception:
        tracer.end_span("ERROR")
        raise


@contextmanager
def module_span(module_name: str, attributes: dict = None):
    """模块级 Span 包装器。"""
    attrs = {**(attributes or {}), "module": module_name}
    with span(module_name, module_name, attrs):
        yield
