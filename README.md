# Enterprise RAG Customer Service System

> 企业级 RAG 客服系统 - 基于公司知识库的内部员工问答平台

---

## 项目概述

本项目设计并实现一套面向企业内部员工的知识库问答系统，基于 RAG（Retrieval-Augmented Generation）架构，提供生产级质量标准，包括：

- 多轮对话上下文理解与查询改写
- 混合检索（BM25 + 向量检索）+ Cross-Encoder 重排序
- 检索结果智能压缩
- 幻觉检测与溯源高亮
- 安全内容分级拦截
- 人工转接与上下文打包

---

## 文档结构

```
docs/
  architecture/
    01-system-architecture.md    # 整体架构设计（本文档）
    02-query-rewriter.md         # Query Rewriter 详细设计
    03-hybrid-retriever.md       # Hybrid Retriever 详细设计
    04-context-compressor.md     # Context Compressor 详细设计
    05-hallucination-guard.md    # Hallucination Guard 详细设计
    06-safety-enforcer.md        # Safety Enforcer 详细设计
    07-escalation-handler.md     # Escalation Handler 详细设计
    08-evaluation.md             # 质量评估体系
    09-observability.md          # 部署与可观测性
```

---

## 技术栈（计划）

| 层级 | 选型 |
|---|---|
| 向量数据库 | Qdrant |
| 全文检索 | Elasticsearch |
| 嵌入模型 | BGE-M3 |
| 重排序 | bge-reranker-v2-m3 |
| 生成模型 | DeepSeek-V3 / Qwen2.5-72B |
| 编排框架 | LangGraph |
| 文档解析 | LlamaParse + Unstructured.io |
| 缓存 | Redis |
| 消息队列 | RabbitMQ |

---

## 设计原则

1. **安全是架构问题，不是 feature**
2. **分层而非大单体**
3. **多信号而非单点判断**
4. **可审计是生产系统的基本要求**

---

> 最后更新: 2026-06-13
