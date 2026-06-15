"""流式输出中的幻觉检测协议设计。

当后端流式输出 LLM token 时，异步进行 NLI 检测。
检测到幻觉时，向前端发送"撤回并替换"信号。

协议消息格式（SSE / WebSocket JSON）:
"""

from dataclasses import dataclass, field
from enum import Enum


class StreamEventType(Enum):
    TOKEN = "token"                 # 正常流式 token
    SENTENCE_COMPLETE = "sentence"  # 一句话完成（NLI 检测触发点）
    RETRACT = "retract"             # 撤回之前的输出
    REPLACE = "replace"             # 替换为修正后的文本
    CITATION = "citation"           # 引用标注
    DONE = "done"                   # 流式结束


@dataclass
class StreamEvent:
    """流式事件。"""
    event: str  # token | sentence | retract | replace | citation | done
    data: dict = field(default_factory=dict)


# ---- 协议设计 ----

STREAMING_PROTOCOL = """
## 流式幻觉检测协议 v1.0

### 消息格式
所有消息以 JSON Lines 格式通过 SSE 或 WebSocket 发送。

```json
{"event": "token", "data": {"text": "年", "index": 0}}
{"event": "token", "data": {"text": "假", "index": 1}}
{"event": "token", "data": {"text": "有", "index": 2}}
...
{"event": "sentence", "data": {"sentence": "年假有5天。", "status": "pending_nli"}}
```

### 检测到幻觉时的撤回-替换流程

1. LLM 流式输出累积到一个完整句子
2. 后端异步发起 NLI 检测
3. 在 NLI 结果返回前，前端正常显示 token
4. NLI 检测到幻觉 → 后端发送 retract + replace:

```json
{"event": "retract", "data": {
    "span": [{"start": 0, "end": 5}, {"start": 10, "end": 15}],
    "reason": "hallucination_detected"
}}
{"event": "replace", "data": {
    "original": "年假有10天，病假无需证明。",
    "replacement": "年假有5天。",
    "highlight": [{"span": [8, 15], "type": "hallucination"}]
}}
{"event": "citation", "data": {
    "sentence_index": 0,
    "source_doc_id": "doc_42",
    "highlight_text": "入职满一年的员工享有带薪年假5天。"
}}
```

### 前端处理逻辑

1. 收到 `token`: 追加到显示缓冲
2. 收到 `sentence`: 标记该句等待验证（前端可显示淡色/灰色表示"验证中"）
3. 收到 `retract`: 从显示缓冲中移除指定 span 的文本
4. 收到 `replace`: 用 replacement 替换 original 内容
5. 收到 `citation`: 在对应句子末尾添加引用标注 [1]
6. 收到 `done`: 流式结束，恢复交互

### 后端异步架构

```
LLM Token Stream
      |
      v
[Sentence Accumulator] --> 累积到完整句子
      |                        |
      v                        v
[前端显示 token]      [NLI Detector (异步)]
                           |
                      entailment?
                      /         \\
                    YES           NO
                    |              |
               [标注引用]    [撤回 + 替换]
                    |              |
                    v              v
                [前端更新]    [前端撤回并替换]
```

### 降级策略

- NLI 超时（>2s）: 不撤回，但前端标记"待验证"
- NLI 模型不可用: 所有句子标记"待验证"，不撤回
- 用户主动提问: 终止当前流的 NLI 检测
"""


def create_retract_event(spans: list[dict], reason: str) -> StreamEvent:
    """创建撤回事件。"""
    return StreamEvent(
        event="retract",
        data={"span": spans, "reason": reason},
    )


def create_replace_event(original: str, replacement: str,
                         highlights: list[dict] = None) -> StreamEvent:
    """创建替换事件。"""
    return StreamEvent(
        event="replace",
        data={
            "original": original,
            "replacement": replacement,
            "highlight": highlights or [],
        },
    )


def create_citation_event(statement_index: int, source_doc_id: str,
                          highlight_text: str) -> StreamEvent:
    """创建引用事件。"""
    return StreamEvent(
        event="citation",
        data={
            "statement_index": statement_index,
            "source_doc_id": source_doc_id,
            "highlight_text": highlight_text,
        },
    )
