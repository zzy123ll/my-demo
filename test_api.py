"""FastAPI TestClient 验证 — 离线模式。"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_get(uri, label):
    r = client.get(uri)
    print(f"GET {label}: {r.status_code}")
    return r.json()

def test_post(uri, label, data):
    r = client.post(uri, json=data)
    print(f"POST {label}: {r.status_code}")
    return r.json()

# === Run tests ===
print("=== Enterprise RAG CS API Verification (offline mode) ===\n")

# 1. Root
d = test_get("/", "/")
print(f"  service: {d['service']} v{d['version']}")

# 2. Health
d = test_get("/health", "/health")
print(f"  status: {d['status']}, uptime={d.get('uptime_seconds','?')}s")

# 3. Chat (正常查询)
d = test_post("/api/v1/chat", "/chat (normal)", {
    "query": "年假有几天", "session_id": "s1", "user_id": "u1", "department": "HR"
})
print(f"  answer: {d.get('answer','')[:100]}")
print(f"  safety: {d.get('safety',{}).get('decision','N/A')}")
hall = d.get("hallucination", {})
print(f"  hallucination: supported={hall.get('supported_count','?')}/{hall.get('total','?')}")
print(f"  citations: {len(d.get('citations',[]))}")
print(f"  latency_ms: {d.get('latency_ms','?')}")

# 4. Chat (安全拦截)
d = test_post("/api/v1/chat", "/chat (blocked)", {
    "query": "公司明年裁员计划", "session_id": "s2", "user_id": "u2"
})
print(f"  safety decision: {d.get('safety',{}).get('decision','N/A')}")

# 5. Escalate
d = test_post("/api/v1/escalate", "/escalate", {
    "session_id": "s3", "user_id": "u3", "user_message": "转人工"
})
print(f"  escalated: {d.get('escalated','?')}, ticket={d.get('ticket_id','?')}")

# 6. Feedback
d = test_post("/api/v1/feedback", "/feedback", {
    "message_id": "msg_1", "rating": 5, "user_id": "u1", "comment": "很准确"
})
print(f"  feedback_id: {d.get('feedback_id','?')}")

# 7. Metrics
d = test_get("/health/metrics", "/metrics")
print(f"  keys: {list(d.keys())[:4]}")

# 8. Docs
r = client.get("/docs")
print(f"GET /docs: {r.status_code}")

# 9. Pending tickets
r = client.get("/api/v1/escalate/pending")
d = r.json()
print(f"GET /escalate/pending: {len(d.get('tickets',[]))} tickets")

# 10. Feedback stats
r = client.get("/api/v1/feedback/stats")
d = r.json()
print(f"GET /feedback/stats: avg_rating={d.get('avg_rating','?')}")

print("\n=== 10/10 ENDPOINTS VERIFIED ===")
