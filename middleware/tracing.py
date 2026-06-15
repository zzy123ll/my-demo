"""OpenTelemetry 追踪中间件。"""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from observability.trace_context import TraceContext, set_current_trace, get_current_trace
from observability.metrics_store import MetricsStore


_global_metrics = MetricsStore()


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:16])
        ctx = TraceContext(trace_id=trace_id, service_name="rag-cs-api")
        set_current_trace(ctx)

        ctx.start_span("http_request", "gateway", {
            "method": request.method,
            "path": request.url.path,
        })

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            ctx.end_span("OK")
            latency_ms = (time.perf_counter() - t0) * 1000
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Request-Time-Ms"] = str(round(latency_ms, 1))
        except Exception:
            ctx.end_span("ERROR")
            raise
        finally:
            ctx.finish()

        return response


def get_metrics() -> MetricsStore:
    return _global_metrics
