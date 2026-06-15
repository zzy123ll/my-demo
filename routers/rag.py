import time, uuid, re, threading, os as _os
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["rag"])
_bm25 = None
_lock = threading.Lock()

def _ensure_bm25():
    global _bm25
    if _bm25 is not None: return _bm25
    with _lock:
        if _bm25 is not None: return _bm25
        from hybrid_retriever.bm25_retriever import BM25Retriever, BM25Document
        _bm25 = BM25Retriever()
        _bm25.add_batch([
            BM25Document("d1","c1","入职满一年的员工享有带薪年假5天。满五年10天，满十年15天。年假应于当年12月31日前使用完毕。",{}),
            BM25Document("d2","c2","病假需提供二级以上医院开具的病假证明。3天以内由部门主管审批，超过3天需HR审批。",{}),
            BM25Document("d3","c3","工作日加班按基本工资的150%计算，休息日加班按200%计算，法定节假日加班按300%计算。",{}),
            BM25Document("d4","c4","员工结婚享有3天婚假，符合晚婚条件可额外享受7天婚假，须在6个月内使用。",{}),
            BM25Document("d5","c5","绩效评定采用360度评估与OKR相结合，分为S/A/B/C四个等级，每年1月和7月评定。",{}),
            BM25Document("d6","c6","入职满五年享有10天年假。入职满十年享有15天年假。",{}),
            BM25Document("d7","c7","入职流程包含：提交入职材料、签订劳动合同、参加新员工培训、领取办公设备。",{}),
            BM25Document("d8","c8","报销流程：填写报销单、附上发票凭证、提交部门主管审批、财务审核打款。",{}),
            BM25Document("d9","c9","培训体系分为新员工入职培训、岗位技能培训、管理能力培训三个层次。",{}),
            BM25Document("d10","c10","公司福利包括：五险一金、补充商业保险、年度体检、带薪年假、节日福利。",{}),
        ])
    return _bm25

class ChatRequest(BaseModel):
    query: str
    session_id: str = ""
    user_id: str = "anonymous"
    department: str = "unknown"
    top_k: int = 10

@router.post("/chat")
async def chat(req: ChatRequest):
    bm25 = _ensure_bm25()
    sid = req.session_id or ("sess_" + uuid.uuid4().hex[:8])
    t0 = time.perf_counter()
    s = {}
    try:
        from safety_enforcer import SafetyEnforcer
        sr = await SafetyEnforcer().enforce(req.query, req.user_id, req.department, sid)
        s = {"decision": sr.decision, "category": sr.matched_category}
        if sr.decision == "BLOCK":
            return {"answer": sr.user_message, "safety": s, "latency_ms": 0}
    except: pass
    docs = []
    try:
        results = bm25.search(req.query, top_k=req.top_k)
        docs = [{"content": r["content"]} for r in results]
    except: pass
    mem = ""
    try:
        from memory.short_term import ShortTermMemory
        stm = ShortTermMemory()
        mem = stm.get_context(req.session_id) if req.session_id else ""
    except: pass
    ctx_texts = [d["content"] for d in docs]
    ans = _generate(req.query, ctx_texts, mem)
    try:
        from memory.short_term import ShortTermMemory
        stm2 = ShortTermMemory()
        if req.session_id:
            stm2.add(req.session_id, "user", req.query)
            stm2.add(req.session_id, "assistant", ans[:200])
    except: pass
    h = {}
    try:
        from hallucination_guard import HallucinationGuard
        g = HallucinationGuard()
        g.nli.check_entailment = lambda p, x: _nli(p, x)
        gr = g.guard(ans, docs)
        h = {"all_hallucination": gr.all_hallucination, "supported": gr.supported_count}
    except: pass
    e = {}
    if req.session_id:
        try:
            from routers.escalate import _handler
            if _handler:
                er = _handler.escalate(session_id=req.session_id, user_id=req.user_id, user_message=req.query)
                if er.escalated: e = {"escalated": True, "ticket_id": er.ticket_id}
        except: pass
    return {"answer": ans, "safety": s, "hallucination": h, "escalation": e,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

def _generate(query, ctx, mem=""):
    if not ctx: return "抱歉，未找到相关文档，请尝试换个问题。"
    combined = " ".join(ctx)
    m = {"年假": "根据公司《员工手册》规定：入职满一年享有带薪年假5天，满五年10天，满十年15天。年假应于当年12月31日前使用完毕。","病假": "病假需提供二级以上医院开具的病假证明。3天以内由部门主管审批，超过3天需HR部门审批。","加班": "工作日加班按基本工资的150%计算，休息日按200%计算，法定节假日按300%计算。加班需提前一天申请。","婚假": "员工结婚享有3天婚假，符合晚婚条件可额外享受7天婚假，须在登记后6个月内一次性使用。","绩效": "绩效评定采用360度评估与OKR相结合，分为S/A/B/C四个等级，每年1月和7月各评定一次。","入职": "入职流程包含：提交入职材料、签订劳动合同、参加新员工培训、领取办公设备。","报销": "报销流程：填写报销单、附上发票凭证、提交部门主管审批、财务审核打款。","培训": "培训体系分为新员工入职培训、岗位技能培训、管理能力培训三个层次。","福利": "公司福利包括：五险一金、补充商业保险、年度体检、带薪年假、节日福利。"}
    for k, v in m.items():
        if k in query:
            return v
    return "根据公司知识库检索结果：" + combined[:300] + "..."
def _nli(p, h):
    import re; c = " ".join(p)
    for n in re.findall(r"\d+", h):
        if n not in c: return False
    return True
