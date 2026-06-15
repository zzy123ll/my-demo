from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from middleware.tracing import TracingMiddleware
from routers import rag_router, escalate_router, feedback_router, health_router
from routers.auth import router as auth_router

app = FastAPI(title="Enterprise RAG CS", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TracingMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(rag_router)
app.include_router(escalate_router)
app.include_router(feedback_router)

@app.get("/")
async def root():
    return {"service": "Enterprise RAG CS", "version": "2.1.0", "dashboard": "/static/index.html"}

@app.on_event("startup")
async def startup():
    import routers.rag as rag_mod
    rag_mod._ensure_bm25()
    import routers.escalate as esc_mod
    from database.models import init_db; init_db()
    from escalation_handler import EscalationHandler
    esc_mod._handler = EscalationHandler()
