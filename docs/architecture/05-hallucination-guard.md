# Hallucination Guard - 详细工程设计

> 三路并行检测（NLI 语义蕴含 + 归因溯源 + 置信度分析），综合判定是否存在幻觉，决定回答是放行、标注警告还是拒答。

---

## 1. 设计哲学

单一检测方法不可靠：

| 检测方法 | 优点 | 盲区 |
|---|---|---|
| NLI 语义蕴含 | 能检测"回答是否基于上下文" | 对长文本误判率高 |
| 归因溯源 | 能定位每个 claim 的来源 | 无法检测措辞偏差 |
| 置信度分析 | 反映模型自身的不确定 | 过度自信是 LLM 通病 |

**三路并行 + 多数投票 + 权重融合**，是当前工程中最务实的方案。

## 2. 整体架构

```mermaid
flowchart TB
    ANSWER[LLM 生成的回答] --> SPLIT{句子拆分}

    SPLIT --> |逐句| NLI[NLI 语义蕴含检测]

    subgraph "路径1: NLI 语义蕴含"
        NLI --> NLI_MODEL[gte-Qwen2-7B-instruct<br/>判断: 回答是否蕴含于上下文]
        NLI_MODEL --> NLI_SCORE[每句蕴含分数 0-1]
    end

    SPLIT --> |逐句| ATTR[归因溯源]

    subgraph "路径2: 归因溯源"
        ATTR --> ATTR_MODEL["LLM 辅助判断<br/>这个 claim 是否能在<br/>检索结果中找到支撑"]
        ATTR_MODEL --> ATTR_SCORE[来源覆盖率]
    end

    ANSWER --> |整体| UNCERT[置信度分析]

    subgraph "路径3: 置信度分析"
        UNCERT --> LOGPROB["logprob 分析<br/>低概率 token 比例"]
        UNCERT --> PERPLEX[困惑度]
        LOGPROB --> UNCERT_SCORE[置信度分数]
    end

    NLI_SCORE --> FUSION[决策融合层]
    ATTR_SCORE --> FUSION
    UNCERT_SCORE --> FUSION

    FUSION --> DECISION{综合判定}

    DECISION -->|"绿灯 (分数 > 0.85)"| PASS[正常回答<br/>+ 溯源脚注 [1][2]]
    DECISION -->|"黄灯 (0.6 - 0.85)"| WARN["警告模式<br/>末尾加'以下内容可能存在偏差'<br/>+ 溯源高亮"]
    DECISION -->|"红灯 (< 0.6)"| REJECT["拒答<br/>'抱歉，我无法确认...'<br/>+ 触发人工转接"]
```

## 3. 路径1: NLI 语义蕴含检测

### 3.1 为什么用 NLI 而非简单的相似度？

- 简单的余弦相似度无法区分"回答是上下文的同义表达"和"回答是模型编造的听起来合理的内容"
- NLI 判断的是逻辑蕴含关系（premise => hypothesis），更接近人的理解方式

### 3.2 实现

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class NLIVerifier:
    NLI_LABELS = ["entailment", "neutral", "contradiction"]
    # gte-Qwen2 的 NLI 输出：entailment 表示假设蕴含于前提

    def __init__(self, model_path: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def verify_sentence(self, premise: str, hypothesis: str) -> dict:
        """
        premise: 检索到的上下文（ground truth）
        hypothesis: LLM 生成的单个句子（待验证）

        Returns:
          { "label": "entailment|neutral|contradiction",
            "entailment_score": 0.87, ... }
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        return {
            "entailment_score": float(probs[0]),
            "neutral_score": float(probs[1]),
            "contradiction_score": float(probs[2]),
            "label": self.NLI_LABELS[torch.argmax(probs).item()]
        }

    def verify_answer(self, context: str, answer: str) -> dict:
        """对整个回答做逐句 NLI 验证"""
        sentences = self._split_sentences(answer)
        results = []

        for sent in sentences:
            result = self.verify_sentence(context, sent)
            results.append({
                "sentence": sent,
                **result
            })

        # 汇总：平均蕴含分数
        avg_entailment = sum(r["entailment_score"] for r in results) / len(results)

        # 标记矛盾句
        contradictions = [r for r in results
                          if r["contradiction_score"] > 0.5]

        return {
            "per_sentence": results,
            "avg_entailment_score": avg_entailment,
            "contradiction_count": len(contradictions),
            "total_sentences": len(sentences),
            "flagged": len(contradictions) > 0 or avg_entailment < 0.6
        }

    def _split_sentences(self, text: str) -> list[str]:
        import re
        return [s.strip() for s in re.split(r'(?<=[。！？])', text) if s.strip()]
```

## 4. 路径2: 归因溯源

### 4.1 设计思路

NLI 只能判断"回答是否与上下文一致"，但无法回答"回答中的每个 claim 是否都能在原文中找到具体支撑"。

归因溯源要做的是：将回答拆解为原子 claim，逐一验证是否能在检索结果中找到对应。

### 4.2 实现

```python
class AttributionChecker:
    ATTRIBUTION_PROMPT = """
你是一个事实核查员。对于下面列出的每个声明，判断它是否能在提供的文档中找到支撑。

【文档内容】
{context}

【待核查的声明】
{claims}

【输出格式】
对于每个声明，输出：
{{
  "claim_index": 声明编号,
  "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",
  "source_chunk_id": "支撑文档的 chunk_id（如果 SUPPORTED）",
  "evidence": "原文中的支撑证据（如果 SUPPORTED）"
}}
"""

    def __init__(self, llm):
        self.llm = llm

    async def check(self, answer: str, chunks: list[dict]) -> dict:
        # 1. 将回答分解为原子声明
        claims = await self._extract_claims(answer)

        # 2. 组装上下文（用压缩后的 chunk 文本）
        context = self._format_context(chunks)

        # 3. LLM 逐一核查
        prompt = self.ATTRIBUTION_PROMPT.format(
            context=context,
            claims=self._format_claims(claims)
        )

        response = await self.llm.generate(prompt, max_tokens=1000)
        verdicts = self._parse_verdicts(response)

        # 4. 计算覆盖率
        supported = sum(1 for v in verdicts if v["verdict"] == "SUPPORTED")
        partial = sum(1 for v in verdicts if v["verdict"] == "PARTIALLY_SUPPORTED")

        coverage = (supported + 0.5 * partial) / len(claims) if claims else 0

        return {
            "claims": claims,
            "verdicts": verdicts,
            "coverage": coverage,
            "supported_count": supported,
            "partial_count": partial,
            "unsupported_count": len(claims) - supported - partial,
            "flagged": coverage < 0.7  # 低于 70% 的归因覆盖率视为可疑
        }

    async def _extract_claims(self, answer: str) -> list[str]:
        """将回答分解为原子声明"""
        prompt = f"""
将以下回答分解为原子声明列表。每条声明应是一个不可再分的独立断言。
只分解事实性陈述，忽略连接词和引导语。

回答：{answer}

输出 JSON 格式：{{"claims": ["声明1", "声明2", ...]}}
"""
        response = await self.llm.generate(prompt, max_tokens=500)
        import json
        return json.loads(response)["claims"]

    def _parse_verdicts(self, response: str) -> list[dict]:
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 解析失败返回全部 UNSUPPORTED（保守策略）
            return []
```

## 5. 路径3: 置信度分析

### 5.1 方法

两种信号来源：

**信号 A：Token-level logprob**

```python
class LogprobAnalyzer:
    def analyze(self, generation_output) -> dict:
        """
        分析 LLM 生成过程中的 token logprobs。

        generation_output 需要包含 token-level logprobs
        （OpenAI API 设置 logprobs=True, top_logprobs=5）
        """
        if not hasattr(generation_output, "logprobs"):
            return {"available": False}

        logprobs = generation_output.logprobs

        # 提取每个位置的 top-1 logprob
        token_probs = []
        for lp in logprobs:
            if lp.top_logprobs:
                token_probs.append(lp.top_logprobs[0].logprob)
            else:
                token_probs.append(float("-inf"))

        if not token_probs:
            return {"available": False}

        # 指标1: 平均 logprob（越低越不确定）
        avg_logprob = sum(token_probs) / len(token_probs)

        # 指标2: 低置信度 token 比例（logprob < -5 的 token 占比）
        low_conf_ratio = sum(1 for p in token_probs if p < -5) / len(token_probs)

        # 指标3: logprob 的方差（方差大表示模型在摇摆）
        variance = sum((p - avg_logprob) ** 2 for p in token_probs) / len(token_probs)

        # 置信度分数（归一化到 0-1）
        # avg_logprob 通常在 -10 到 0 之间，映射方式可以校准
        confidence = 1.0 / (1.0 + np.exp(-(avg_logprob + 2)))  # sigmoid 校准

        return {
            "available": True,
            "avg_logprob": avg_logprob,
            "low_conf_ratio": low_conf_ratio,
            "logprob_variance": variance,
            "confidence": confidence,
            "flagged": confidence < 0.4 or low_conf_ratio > 0.3
        }
```

**信号 B：语义一致性（自检）**

```python
class SelfConsistencyCheck:
    """让 LLM 对同一问题多次采样，检查回答一致性"""

    def __init__(self, llm, num_samples: int = 3):
        self.llm = llm
        self.num_samples = num_samples

    async def check(self, query: str, context: str) -> dict:
        # 多次采样
        samples = []
        for i in range(self.num_samples):
            response = await self.llm.generate(
                prompt=self._build_prompt(query, context),
                temperature=0.7,  # 非零温度以获得多样性
                max_tokens=500
            )
            samples.append(response)

        # 两两计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings = [self._embed(s) for s in samples]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)

        avg_consistency = sum(similarities) / len(similarities) if similarities else 0

        return {
            "samples": samples,
            "avg_consistency": avg_consistency,
            "flagged": avg_consistency < 0.7
        }
```

## 6. 决策融合层

```python
class DecisionFusion:
    """三路信号综合决策"""

    # 各路权重（可通过 A/B 调整）
    WEIGHTS = {
        "nli": 0.45,         # NLI 权重最高
        "attribution": 0.35,  # 归因次之
        "uncertainty": 0.20   # 置信度最低（LLM 自评不可靠）
    }

    # 各等级的阈值
    THRESHOLDS = {
        "green": 0.85,   # >= 绿灯放行
        "yellow": 0.60,  # >= 黄灯警告
        # < 0.60 红灯拒答
    }

    def decide(self, nli_result, attr_result, uncert_result) -> dict:
        # 各路信号归一化到 0-1
        signals = {
            "nli": self._normalize_nli(nli_result),
            "attribution": self._normalize_attr(attr_result),
            "uncertainty": self._normalize_uncert(uncert_result)
        }

        # 加权融合
        fused_score = sum(
            signals[k] * self.WEIGHTS[k] for k in signals
        )

        # 判级
        if fused_score >= self.THRESHOLDS["green"]:
            verdict = "pass"
        elif fused_score >= self.THRESHOLDS["yellow"]:
            verdict = "warn"
        else:
            verdict = "reject"

        return {
            "verdict": verdict,
            "fused_score": fused_score,
            "signals": signals,
            "details": {
                "nli": nli_result,
                "attribution": attr_result,
                "uncertainty": uncert_result
            }
        }

    def _normalize_nli(self, nli_result):
        return nli_result.get("avg_entailment_score", 0)

    def _normalize_attr(self, attr_result):
        return attr_result.get("coverage", 0)

    def _normalize_uncert(self, uncert_result):
        if not uncert_result.get("available"):
            return 0.5  # 不可用时中性
        return uncert_result.get("confidence", 0.5)
```

## 7. 溯源高亮实现

```python
class SourceHighlighter:
    """为回答的每个句子生成引用标注"""

    def annotate(self, answer: str,
                 attribution_result: dict,
                 chunks: list[dict]) -> dict:
        sentences = self._split_sentences(answer)
        verdicts = attribution_result.get("verdicts", [])

        annotated_parts = []
        source_map = {}

        for i, sent in enumerate(sentences):
            # 找到对应的归因判决
            v = verdicts[i] if i < len(verdicts) else None

            if v and v["verdict"] == "SUPPORTED" and v.get("source_chunk_id"):
                chunk_id = v["source_chunk_id"]
                ref_num = len(source_map) + 1
                source_map[ref_num] = self._get_chunk_info(chunk_id, chunks)
                annotated_parts.append(f"{sent} [{ref_num}]")
            else:
                annotated_parts.append(sent)

        annotated_answer = "".join(annotated_parts)

        return {
            "answer_text": annotated_answer,
            "sources": source_map  # {1: {chunk_id, doc_title, content, page}, ...}
        }

    def _get_chunk_info(self, chunk_id, chunks):
        for c in chunks:
            if c["chunk_id"] == chunk_id:
                return {
                    "chunk_id": c["chunk_id"],
                    "doc_title": c["doc_title"],
                    "content_preview": c["content"][:200],
                    "page_number": c.get("page_number")
                }
        return {"chunk_id": chunk_id, "error": "not found"}
```

## 8. 边界情况与降级

| 情况 | 策略 |
|---|---|
| NLI 模型不可用（OOM/超时） | 降级为仅用归因+置信度，权重重新分配 |
| 归因 LLM 返回格式错误 | 保守处理——假设所有声明 UNSUPPORTED |
| logprobs 不可用（如 DeepSeek 不支持） | 置信度分数设为中性 0.5 |
| 回答太短（<2句） | 跳过句子级检查，只做整体 NLI |
| 融合分数在边界（0.84-0.86） | 加一次额外的 LLM 自查判断 |

## 9. 监控指标

| 指标 | 目标 | 报警 |
|---|---|---|
| 放行率（Pass / Total） | > 85% | < 70% |
| 误拒率（应为 Pass 但被 Reject） | < 1% | > 3% |
| NLI 推理延迟 P99 | < 500ms | > 1s |
| 归因检测延迟 P99 | < 1s | > 2s |
| 幻觉漏检率（幻觉但未标记） | < 2% | > 5% |

## 10. API 契约

```
POST /api/v1/verify

Request:
{
  "answer": "2024年绩效评定采用360度评估与OKR达成率相结合的方式...",
  "context_chunks": [{...}],
  "query": "2024年绩效评定的标准是什么？",
  "generation_metadata": {
    "logprobs": [...],
    "model": "qwen2.5-72b"
  }
}

Response:
{
  "verdict": "pass",
  "fused_score": 0.91,
  "signals": {
    "nli": 0.93,
    "attribution": 0.88,
    "uncertainty": 0.92
  },
  "annotated_answer": "2024年绩效评定采用360度评估与OKR达成率相结合的方式 [1]...",
  "sources": {
    "1": {
      "chunk_id": "doc_42_chunk_3",
      "doc_title": "2024年绩效管理制度",
      "content_preview": "绩效评定采用360度评估与OKR达成率相结合...",
      "page_number": 3
    }
  }
}
```

---

> 继续阅读: [06-safety-enforcer.md](06-safety-enforcer.md)
