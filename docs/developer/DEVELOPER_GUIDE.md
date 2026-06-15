# Enterprise RAG CS — 开发者完全指南

> 企业级 RAG 客服系统 | 版本 2.1.0 | 8 个核心模块 | 210+ 测试用例

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [Query Rewriter 模块](#3-query-rewriter-模块)
4. [Hybrid Retriever 模块](#4-hybrid-retriever-模块)
5. [Context Compressor 模块](#5-context-compressor-模块)
6. [Hallucination Guard 模块](#6-hallucination-guard-模块)
7. [Safety Enforcer 模块](#7-safety-enforcer-模块)
8. [Escalation Handler 模块](#8-escalation-handler-模块)
9. [API 接口文档](#9-api-接口文档)
10. [前端页面](#10-前端页面)
11. [部署指南](#11-部署指南)
12. [数据库设计](#12-数据库设计)
13. [测试体系](#13-测试体系)

---

## 1. 项目概述

Enterprise RAG CS 是一套面向企业内部的智能客服系统，基于 **RAG（Retrieval-Augmented Generation）** 架构，提供生产级质量标准。

### 1.1 核心能力

| 能力 | 描述 |
|---|---|
| 智能问答 | 基于公司知识库的多轮对话问答 |
| 安全拦截 | 三级流水线检测敏感问题（裁员、薪资、注入攻击） |
| 幻觉检测 | NLI 模型验证回答真实性，溯源高亮 |
| 人工转接 | 自动触发 + 工单队列 + 坐席工作台 |
| 多用户管理 | JWT 认证、角色权限（管理员/坐席/员工） |
| 全链路监控 | OpenTelemetry 埋点 + 指标看板 + 告警引擎 |

### 1.2 项目结构

```
enterprise-rag-cs/
  app.py                        FastAPI 主入口
  routers/
    auth.py                     认证接口 (POST /login, POST /register)
    rag.py                      核心问答接口 (POST /chat)
    escalate.py                 转接管理接口 (POST /escalate, GET /pending)
    feedback.py                 反馈收集接口 (POST /feedback)
    health.py                   健康检查接口 (GET /health, /metrics, /dashboard)
  query_rewriter/               查询改写模块 (双阶段 + 一致性校验)
  hybrid_retriever/             混合检索模块 (BM25 + RRF + 重排序)
  context_compressor/           上下文压缩模块 (提取式 + Token预算)
  hallucination_guard/          幻觉检测模块 (NLI + 答案拆分 + 流式协议)
  safety_enforcer/              安全拦截模块 (L1关键词 + L2分类 + L3 LLM)
  escalation_handler/           转接处理模块 (规则引擎 + 工单队列 + 会话管理)
  evaluation/                   评估模块 (指标计算 + A/B测试 + 反馈收集)
  observability/                可观测性模块 (OTel埋点 + 指标 + 看板 + 告警)
  database/                     数据库层 (SQLAlchemy + SQLite)
  memory/                       记忆系统 (短时滑动窗口 + 长时摘要)
  middleware/                    HTTP中间件 (追踪 + 认证)
  static/                       前端页面 (login.html / user_chat.html / admin.html)
  tests/                        测试用例 (210+ 个)
  docs/                         文档 (架构设计 / 面试题 / 开发指南)
```

---

## 2. 系统架构

### 2.1 请求处理流程

```
用户请求 (POST /api/v1/chat)
  |
  +--[Auth Middleware]--> JWT 令牌验证, 解析 user_id/role
  |
  +--[Tracing Middleware]--> 创建 trace_id, 记录入口 Span
  |
  +--[Safety Enforcer]      L1 关键词检测
  |     +-- 命中 sensitive 关键词 --> 拦截并返回拒绝消息
  |     +-- 未命中 --> 继续
  |
  +--[Query Rewriter]       多轮对话上下文补全
  |     +-- 规则引擎: 消解"它/这个/那个/那...呢?"
  |     +-- LLM 改写: 规则失败时调用轻量模型
  |     +-- 一致性校验: sentence-transformers 相似度 >= 0.85
  |
  +--[Hybrid Retriever]     BM25 + 向量双路检索
  |     +-- BM25 (内存): 纯 Python 实现, 关键词+双字词分词
  |     +-- 向量 (ChromaDB): embedding 相似度搜索
  |     +-- RRF 融合: 1/(k+rank) 分数
  |     +-- Cross-encoder 重排序: bge-reranker-v2-m3
  |
  +--[Context Compressor]   检索结果压缩
  |     +-- TokenBudgetManager: 计算可用 token 数
  |     +-- 提取式: 相似度选 top-k 句
  |     +-- 完整性检查: 实体保留率 >= 85%
  |
  +--[LLM / Rule Generator] 生成回答
  |     +-- LLM: ChatOpenAI + prompt (需 .env 配置 API key)
  |     +-- 规则兜底: 关键词匹配 9 类常见问题
  |
  +--[Hallucination Guard]  幻觉检测
  |     +-- ClaimSplitter: 拆分为原子声明
  |     +-- NLI 检查: 每个声明是否被文档蕴含
  |     +-- 全幻觉 --> 返回预设话术
  |
  +--[Escalation Handler]   转接检测
  |     +-- 关键词触发 (转人工/投诉)
  |     +-- 负面情绪触发 (连续 3 轮)
  |     +-- Hallucination 拒答触发
  |
  +--[Memory System]        记忆更新
  |     +-- ShortTermMemory: 滑动窗口追加本轮对话
  |     +-- LongTermMemory: 超量时 LLM 压缩为摘要
  |
  +--[Metrics / Tracing]    性能埋点
        +-- 记录各模块延迟
        +-- 记录安全/幻觉标记
        +-- 导出到 JSON 文件
```

### 2.2 数据流图

```
[用户] --query--> [Safety L1] --pass--> [QueryRewriter] --rewritten--> [BM25+Vector]
                                                                          |
                                                                     [RRF Fusion]
                                                                          |
                                                                     [CrossEncoder]
                                                                          |
                                                                     [Compressor]
                                                                          |
                                                                     [LLM/Generator]
                                                                          |
                                                                     [HallucinationGuard]
                                                                          |
                                                                     [Answer + Citations] --> [用户]
```

---

## 3. Query Rewriter 模块

### 3.1 模块位置

`query_rewriter/`

### 3.2 核心类

| 类 | 职责 |
|---|---|
| `ConversationState` | 维护多轮对话状态 (entity_map, last_topic, user_profile) |
| `CoreferenceResolver` | 阶段一: 规则引擎, 处理中文代词和省略句式 |
| `LLMRewriter` | 阶段二: 调用 LLM_MODEL1 (DeepSeek-V4-Flash) 改写 |
| `ConsistencyChecker` | 一致性校验: sentence-transformers 相似度 >= 0.85 |
| `QueryRewriter` | 主编排器: 双阶段 + 校验 + 反问生成 |

### 3.3 中文指代消解示例

```
第1轮: "年假有几天?"          --> entity_map = {"年假": EntityInfo(...)}
第2轮: "那病假呢?"            --> 规则: topic_ellipsis → "病假有几天?"
第3轮: "它需要什么材料?"      --> 规则: pronoun_it → "病假需要什么材料?"
第4轮: "还有吗?"              --> 规则: supplement_request → "关于病假还有哪些信息?"
```

### 3.4 使用示例

```python
from query_rewriter import ConversationState, CoreferenceResolver

state = ConversationState()
state.track_entity("年假", "topic")
state.update_topic("休假政策")

resolver = CoreferenceResolver()
result = resolver.resolve("它需要什么材料?", state)
# result.rewritten_query = "年假需要什么材料?"
```

---

## 4. Hybrid Retriever 模块

### 4.1 模块位置

`hybrid_retriever/`

### 4.2 核心类

| 类 | 职责 |
|---|---|
| `BM25Retriever` | 纯 Python BM25 (无外部依赖), 中文双字词分词 |
| `VectorRetriever` | ChromaDB 向量检索 (支持 OpenAI 兼容 embedding API) |
| `RRFusion` | Reciprocal Rank Fusion: RRF_score = sum(1/(60 + rank)) |
| `WeightedFusion` | Min-max 归一化 + alpha 加权融合 |
| `CrossEncoderReranker` | sentence-transformers CrossEncoder 重排序 |
| `HybridRetriever` | 主编排器: 并行检索 → 融合 → 重排 → Top-K |
| `IndexManager` | 异步索引更新 (队列 + 批量同步) |

### 4.3 融合算法

**RRF (k=60, 默认)**:
```python
fusion = RRFusion(k=60)
results = fusion.fuse(bm25_results, vector_results)
```

**加权融合 (alpha=0.5, 可配)**:
```python
fusion = WeightedFusion(alpha=0.5)
results = fusion.fuse(bm25_results, vector_results)
```

### 4.4 工厂方法 (用于 A/B 测试)

```python
from hybrid_retriever import create_hybrid_retriever

# RRF 模式
retriever = create_hybrid_retriever(fusion_strategy="rrf", rrf_k=120)

# 加权模式
retriever = create_hybrid_retriever(fusion_strategy="weighted", weighted_alpha=0.7)
```

### 4.5 BM25 预填充知识库

```python
from hybrid_retriever.bm25_retriever import BM25Retriever, BM25Document

bm25 = BM25Retriever()
bm25.add_batch([
    BM25Document("doc_1", "chunk_1", "入职满一年的员工享有带薪年假5天。", {}),
    BM25Document("doc_2", "chunk_2", "病假需提供医院证明。", {}),
])
results = bm25.search("年假有几天", top_k=5)
```

---

## 5. Context Compressor 模块

### 5.1 模块位置

`context_compressor/`

### 5.2 核心类

| 类 | 职责 |
|---|---|
| `TokenBudgetManager` | 基于 context_window 和 reserved_tokens 计算可用 token |
| `ExtractiveCompressor` | sentence-transformers 相似度选 top-k 句 |
| `GenerativeCompressor` | LLM 提取式压缩 (面向问题的摘要) |
| `IntegrityChecker` | 正则提取关键实体, 验证保留率 >= 0.85 |
| `ContextCompressor` | 主编排器: 压缩 + 完整性检查 + 截断降级 |

### 5.3 使用示例

```python
from context_compressor import ContextCompressor, CompressorConfig

config = CompressorConfig(mode="extractive", context_window=4096)
compressor = ContextCompressor(config)

result = compressor.compress_sync(
    document="入职满一年5天年假。满五年10天。病假需医院证明。...",
    query="年假有几天",
)
# result.compressed_text = 选中的 top-k 个最相关句子
# result.compression_ratio = 压缩比
# result.latency_ms = 耗时
```

### 5.4 Token 预算计算

```
available = context_window - reserved_tokens(500) - system_prompt - query
如果 document_tokens > available:
    触发压缩 (提取式或生成式)
否则:
    透传原文档
```

---

## 6. Hallucination Guard 模块

### 6.1 模块位置

`hallucination_guard/`

### 6.2 核心类

| 类 | 职责 |
|---|---|
| `ClaimSplitter` | 将 LLM 回答拆分为原子声明 (spaCy 优先, 正则降级) |
| `NLIChecker` | mDeBERTa-v3 做 NLI 蕴含检测 |
| `HallucinationGuard` | 主编排器: 拆分 → NLI 检测 → 输出 safe_answer + citations |
| `StreamEvent` | 流式协议: retract / replace / citation 事件 |

### 6.3 使用示例

```python
from hallucination_guard import HallucinationGuard

guard = HallucinationGuard()
output = guard.guard(
    answer="年假有15天。病假需要证明。",
    retrieved_docs=[{"content": "入职满一年5天年假。"}, {"content": "病假需医院证明。"}]
)
# output.safe_answer = "病假需要证明。"  (只有这句话有文档支撑)
# output.unsupported_spans = [{"text": "年假有15天。", "reason": "无文档支持"}]
```

### 6.4 输出格式

```json
{
  "safe_answer": "病假需要提供医院证明。",
  "unsupported_spans": [
    {"text": "年假有15天。", "reason": "无文档支持"}
  ],
  "citations": [
    {"statement_index": 0, "source_doc_id": "doc_2", "highlight_text": "病假需医院证明"}
  ],
  "all_hallucination": false,
  "stats": {"total": 2, "supported": 1, "hallucination": 1}
}
```

---

## 7. Safety Enforcer 模块

### 7.1 模块位置

`safety_enforcer/`

### 7.2 三级流水线

| 级别 | 方法 | 延迟 | 触发条件 | 动作 |
|---|---|---|---|---|
| L1 | AC 自动机 + 正则 | <1ms | 关键词/正则命中 | 直接拦截 |
| L2 | sentence-transformers 分类 | ~20ms | score > 0.7 | 拦截 |
| L2→L3 | LLM 裁决 | ~2s | 0.4 < score < 0.7 | LLM 最终判定 |
| L3 超时 | 超时降级 | 2s | LLM 超时 | 按 L2 结果处理 |

### 7.3 敏感类别配置

`sensitive_categories.yaml`:
```yaml
categories:
  layoff:     {risk_level: critical, keywords: [裁员, 优化人员, ...], action: block_and_escalate}
  salary:     {risk_level: high, keywords: [工资, 薪资, ...], action: block_and_escalate}
  jailbreak:  {risk_level: critical, keywords: [忽略, 管理员密码, ...], action: block}
  pii:        {risk_level: high, regex: ['\d{17}[\dXx]', ...], action: block}
  politics:   {risk_level: critical, keywords: [...], action: block}
  harassment: {risk_level: medium, keywords: [...], action: warn}
```

### 7.4 使用示例

```python
from safety_enforcer import SafetyEnforcer

enforcer = SafetyEnforcer()
result = await enforcer.enforce("公司明年裁员计划", "user_1", "Engineering")
# result.decision = "BLOCK"
# result.triggered_level = "L1"
# result.matched_category = "layoff"
# result.user_message = "该问题涉及敏感信息，已为您转接人工客服处理。"
```

---

## 8. Escalation Handler 模块

### 8.1 模块位置

`escalation_handler/`

### 8.2 触发规则

| 触发条件 | 优先级 |
|---|---|
| 用户消息包含转接关键词 (转人工/投诉/找经理) | 3 |
| 同一对话中 Hallucination Guard 拒答 2 次 | 2 |
| Safety Enforcer 拦截 + 用户申诉 | 1 |
| 连续 3 轮负面情绪 | 2 |

### 8.3 工单生命周期

```
PENDING → IN_PROGRESS (坐席接单) → RESOLVED (坐席回复)
                                  → EXPIRED (超时 60 分钟)
```

### 8.4 使用示例

```python
from escalation_handler import EscalationHandler

handler = EscalationHandler()

# 触发转接
result = handler.escalate(
    session_id="sess_1", user_id="zhangsan",
    user_message="帮我转人工"
)
# result.escalated = True
# result.ticket_id = "TKT-A1B2C3D4"

# 坐席查看待处理
pending = handler.list_pending()  # 返回工单列表

# 坐席解决
handler.resolve_ticket("TKT-A1B2C3D4", "已向用户解释年假政策")
```



# Enterprise RAG CS — 开发者完全指南 (Part 2)

---

## 9. API 接口文档

### 9.1 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`
- **认证**: JWT Bearer Token (`Authorization: Bearer <token>`)
- **Swagger UI**: `http://localhost:8000/docs`

### 9.2 认证接口

#### POST /api/v1/auth/login — 登入

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id":"zhangsan","password":"staff123"}'
```

**Response (200)**:
```json
{
  "token": "eyJhbGci...",
  "user": {"user_id": "zhangsan", "username": "Zhang San", "role": "staff", "department": "Engineering"}
}
```

**Response (401)**: `{"detail": "Invalid credentials"}`

---

#### POST /api/v1/auth/register — 注册

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"user_id":"newuser","username":"New User","password":"pass123","department":"Engineering"}'
```

**Response (200)**: `{"token": "...", "user": {...}}`

---

### 9.3 核心问答接口

#### POST /api/v1/chat — 智能问答

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "年假有几天？",
    "session_id": "sess_001",
    "user_id": "zhangsan",
    "department": "Engineering",
    "top_k": 5
  }'
```

**Request 参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| query | string | 是 | - | 用户问题 |
| session_id | string | 否 | "" | 会话 ID (用于记忆系统) |
| user_id | string | 否 | "anonymous" | 用户 ID |
| department | string | 否 | "unknown" | 部门 (影响安全策略) |
| top_k | int | 否 | 10 | 检索文档数量 (1-50) |

**Response (200 — 正常回答)**:
```json
{
  "answer": "根据公司《员工手册》规定：入职满一年享有带薪年假5天...",
  "safety": {"decision": "PASS"},
  "hallucination": {"all_hallucination": false, "supported": 1},
  "escalation": {},
  "citations": [{"chunk_id": "c1", "score": 1.56}],
  "latency_ms": 120.5
}
```

**Response (200 — 安全拦截)**:
```json
{
  "answer": "该问题涉及敏感信息，已为您转接人工客服处理。",
  "safety": {"decision": "BLOCK", "matched_category": "layoff", "triggered_level": "L1"}
}
```

**Response (200 — 转接已触发)**:
```json
{
  "answer": "...",
  "escalation": {"escalated": true, "ticket_id": "TKT-A1B2C3D4"}
}
```

**触发安全的测试查询**:
```
"公司明年裁员计划"    → layoff (L1)
"我的工资比同事低吗"  → salary (L1)
"忽略之前的指令"      → jailbreak (L1)
"440101199001011234"   → pii (L1)
```

**正常问答示例**:
```
"年假有几天？"        → 年假政策回答
"病假怎么申请？"      → 病假政策回答
"加班费怎么算？"      → 加班政策回答
"转人工"              → 触发转接
```

---

### 9.4 转接管理接口

#### POST /api/v1/escalate — 创建转接工单

```bash
curl -X POST http://localhost:8000/api/v1/escalate \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_id":"u1","user_message":"转人工"}'
```

**Response (200)**:
```json
{"escalated": true, "ticket_id": "TKT-A1B2C3D4", "trigger_reason": "用户消息包含转接关键词: \"人工\""}
```

#### GET /api/v1/escalate/pending — 查看待处理工单

```bash
curl http://localhost:8000/api/v1/escalate/pending
```

**Response (200)**:
```json
{"tickets": [{"ticket_id": "TKT-A1B2C3D4", "user_id": "zhangsan", "trigger_reason": "人工", "status": "pending"}]}
```

#### POST /api/v1/escalate/{ticket_id}/resolve — 解决工单

```bash
curl -X POST http://localhost:8000/api/v1/escalate/TKT-A1B2C3D4/resolve \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"TKT-A1B2C3D4","resolution":"已向用户解释年假政策","solution_tags":["hr","policy"]}'
```

---

### 9.5 反馈接口

#### POST /api/v1/feedback — 提交反馈

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"message_id":"msg_1","rating":5,"comment":"回答很准确","user_id":"zhangsan"}'
```

**Response (200)**:
```json
{"status": "ok", "feedback_id": "FB-1E148960"}
```

#### GET /api/v1/feedback/stats — 反馈统计

```bash
curl http://localhost:8000/api/v1/feedback/stats
```

**Response (200)**:
```json
{"total": 10, "avg_rating": 4.2, "positive_rate": 0.8, "negative_rate": 0.1}
```

---

### 9.6 健康检查接口

#### GET /health — 基础健康检查

```bash
curl http://localhost:8000/health
```

**Response (200)**:
```json
{"status": "healthy", "service": "rag-cs-api", "version": "2.1.0", "uptime_seconds": 3600.5}
```

#### GET /health/metrics — 性能指标

```bash
curl http://localhost:8000/health/metrics
```

**Response (200)**:
```json
{
  "period_minutes": 15,
  "total_requests": 42,
  "avg_latency_ms": {"e2e": 1250.5, "retriever": 120.3, "generator": 850.2},
  "hallucination_rate": 0.05,
  "safety_block_rate": 0.12
}
```

#### GET /health/dashboard — 文本看板

```bash
curl http://localhost:8000/health/dashboard
```

返回完整的文本格式健康看板。

---

## 10. 前端页面

### 10.1 页面列表

| 页面 | 路径 | 用途 |
|---|---|---|
| 登入页 | `/static/login.html` | 用户登入, 按角色跳转 |
| 员工问答 | `/static/user_chat.html` | 普通员工提问, 多轮对话 |
| 坐席工作台 | `/static/admin.html` | 人工坐席处理工单, 回复用户 |

### 10.2 预置用户

| 角色 | 用户名 | 密码 | 跳转目标 |
|---|---|---|---|
| 管理员 | `admin` | `admin123` | admin.html |
| 坐席 | `agent01` | `agent123` | admin.html |
| 员工 | `zhangsan` | `staff123` | user_chat.html |

### 10.3 前端调用流程

```
1. 用户访问 /static/login.html
2. 输入用户名密码 → POST /api/v1/auth/login → 获得 JWT token
3. token 存入 localStorage
4. 根据 role 跳转到 user_chat.html (员工) 或 admin.html (坐席/管理员)
5. 后续所有请求自动携带 Authorization: Bearer <token>
```

### 10.4 坐席工作流

```
1. 坐席登入 → admin.html
2. 页面每 10 秒轮询 GET /api/v1/escalate/pending
3. 点击工单查看详情
4. 输入回复内容 → 点击"标记已解决" → POST /api/v1/escalate/{id}/resolve
5. 工单状态变为 RESOLVED
```

---

## 11. 部署指南

### 11.1 本地开发 (uvicorn)

```powershell
set HF_HUB_OFFLINE=1
uvicorn app:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://localhost:8000/static/login.html`

### 11.2 Docker Compose

```powershell
docker compose build
docker compose up -d
```

服务:
- API: `localhost:8000`
- Redis: `localhost:6379`
- ChromaDB (可选): `localhost:8001` (`--profile full`)
- PostgreSQL (可选): `localhost:5432` (`--profile full`)

### 11.3 环境变量

| 变量 | 说明 | 示例 |
|---|---|---|
| `LLM_API_KEY` | LLM API 密钥 | `sk-bff68d...` |
| `LLM_BASE_URL` | LLM API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL` | 强模型 | `deepseek-v4-pro` |
| `LLM_MODEL1` | 轻模型 | `deepseek-v4-flash` |
| `HF_HUB_OFFLINE` | 离线模式 (设 1 跳过 HuggingFace 下载) | `1` |
| `CHROMA_PERSIST_DIR` | ChromaDB 存储路径 | `./chroma_data` |

---

## 12. 数据库设计

### 12.1 SQLite 表结构

**users** — 用户表:
| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| user_id | VARCHAR(64) | 用户 ID (唯一) |
| username | VARCHAR(128) | 用户名 |
| password_hash | VARCHAR(256) | 密码哈希 (PBKDF2 + salt) |
| salt | VARCHAR(64) | 密码 salt |
| role | VARCHAR(16) | 角色: admin / agent / staff |
| department | VARCHAR(64) | 部门 |

**conversations** — 会话表:
| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| session_id | VARCHAR(64) | 会话 ID (唯一) |
| user_id | VARCHAR(64) | 用户 ID |
| status | VARCHAR(16) | 状态: active / archived |
| summary | TEXT | 会话摘要 |

**messages** — 消息表:
| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| session_id | VARCHAR(64) | 会话 ID |
| user_id | VARCHAR(64) | 用户 ID |
| role | VARCHAR(16) | user / assistant / agent |
| content | TEXT | 消息内容 |

**tickets** — 工单表:
| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| ticket_id | VARCHAR(64) | 工单 ID (唯一) |
| session_id | VARCHAR(64) | 关联会话 |
| user_id | VARCHAR(64) | 用户 ID |
| status | VARCHAR(16) | pending / in_progress / resolved |
| resolution | TEXT | 解决方案 |

### 12.2 数据库初始化

数据库文件 `data/rag_cs.db` 在服务器首次启动时自动创建，同时插入 3 个预置用户。

---

## 13. 测试体系

### 13.1 测试覆盖

| 模块 | 测试文件 | 测试数 |
|---|---|---|
| Query Rewriter | `tests/test_query_rewriter.py` | 29 |
| Hybrid Retriever | `tests/test_hybrid_retriever.py` | 24 |
| Context Compressor | `tests/test_context_compressor.py` | 30 |
| Hallucination Guard | `tests/test_hallucination_guard.py` | 26 |
| Safety Enforcer | `tests/test_safety_enforcer.py` | 30 |
| Escalation Handler | `tests/test_escalation_handler.py` | 30 |
| Evaluation | `tests/test_evaluation.py` | 24 |
| Observability | `tests/test_observability.py` | 17 |
| **总计** | | **210** |

### 13.2 运行测试

```powershell
# 全部测试
python -m pytest tests/ -v

# 单模块
python -m pytest tests/test_hallucination_guard.py -v

# 按名称过滤
python -m pytest tests/ -v -k "safety"
```

### 13.3 Mock 策略

由于沙箱环境网络受限，所有依赖 HuggingFace 模型的测试均使用 mock:

- `sentence-transformers`: 确定性假嵌入向量
- `NLIChecker`: 直接设置 `_model` 和 `_tokenizer` 属性
- `ChatOpenAI`: AsyncMock 返回预定义回答

---

> 文档版本: v1.0 | 生成日期: 2026-06-15 | 项目: Enterprise RAG CS
