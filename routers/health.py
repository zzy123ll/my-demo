"""健康检查端点。"""

import time
from fastapi import APIRouter
from middleware.tracing import get_metrics

router = APIRouter(tags=["health"])
_startup_time = time.time()


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "rag-cs-api",
        "version": "2.1.0",
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@router.get("/health/metrics")
async def metrics():
    store = get_metrics()
    return store.snapshot(minutes=15)


@router.get("/health/dashboard")
async def dashboard():
    from observability.dashboard import Dashboard
    store = get_metrics()
    dash = Dashboard(store, default_minutes=15)
    report = dash.generate(minutes=15)
    return {
        "title": report.title,
        "period_minutes": report.period_minutes,
        "snapshot": report.snapshot,
        "alert_count": report.alert_count,
        "text": report.to_text(),
    }
