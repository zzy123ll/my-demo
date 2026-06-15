"""全局 Trace Context — 使用 contextvars 在协程/线程间安全传递 trace_id 和 span_id。"""

from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field


_current_trace: contextvars.ContextVar = contextvars.ContextVar(
    "current_trace_ctx", default=None
)


@dataclass
class TraceSpan:
    """单个 Span 记录。"""
    span_id: str
    parent_span_id: str = ""
    name: str = ""
    module: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "OK"      # OK | ERROR
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class TraceContext:
    """一次完整请求的 Trace 上下文。

    使用 contextvars 确保在 async 任务中正确传递。
    """

    def __init__(self, trace_id: str = "", service_name: str = ""):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.service_name = service_name
        self.spans: list[TraceSpan] = []
        self._span_stack: list[str] = []   # 当前 Span ID 栈
        self.start_time = time.perf_counter()
        self.end_time: float = 0.0
        self.metadata: dict = {}

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000

    def start_span(self, name: str, module: str = "",
                   attributes: dict = None) -> TraceSpan:
        parent_id = self._span_stack[-1] if self._span_stack else ""
        sid = uuid.uuid4().hex[:8]

        span = TraceSpan(
            span_id=sid,
            parent_span_id=parent_id,
            name=name,
            module=module,
            start_time=time.perf_counter(),
            attributes=attributes or {},
        )
        self.spans.append(span)
        self._span_stack.append(sid)
        return span

    def end_span(self, span_id: str = None, status: str = "OK") -> None:
        if not self._span_stack:
            return
        sid = span_id or self._span_stack[-1]
        for span in self.spans:
            if span.span_id == sid:
                span.end_time = time.perf_counter()
                span.status = status
                break
        if self._span_stack and self._span_stack[-1] == sid:
            self._span_stack.pop()

    def finish(self) -> None:
        """完成 trace，结束所有未关闭的 span。"""
        self.end_time = time.perf_counter()
        while self._span_stack:
            self.end_span(status="OK")

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "service": self.service_name,
            "duration_ms": round(self.duration_ms, 2),
            "span_count": len(self.spans),
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent": s.parent_span_id,
                    "name": s.name,
                    "module": s.module,
                    "duration_ms": round(s.duration_ms, 2),
                    "status": s.status,
                    "attributes": s.attributes,
                }
                for s in self.spans
            ],
        }


def get_current_trace() -> TraceContext | None:
    return _current_trace.get()


def set_current_trace(ctx: TraceContext) -> None:
    _current_trace.set(ctx)
