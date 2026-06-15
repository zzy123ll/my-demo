"""评估指标: Hit@k, MRR, Faithfulness, Relevance, ROUGE-L。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---- Retrieval metrics ----

def compute_hit_at_k(retrieved_ids: list[str],
                     relevant_ids: list[str],
                     k_values: list[int] = (1, 5, 10)) -> dict[int, float]:
    """Hit@k: 相关文档是否在 Top-K 中。"""
    result = {}
    for k in k_values:
        top_k = retrieved_ids[:k]
        hit = any(rid in top_k for rid in relevant_ids)
        result[k] = 1.0 if hit else 0.0
    return result


def compute_mrr(retrieved_ids: list[str],
                relevant_ids: list[str],
                k: int = 10) -> float:
    """MRR@k: 第一个相关文档的排名倒数。"""
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


# ---- Generation metrics ----

def compute_faithfulness(answer: str,
                         retrieved_docs: list[str],
                         nli_checker=None) -> float:
    """Faithfulness: 答案中原子声明被文档支持的比率。

    复用 Hallucination Guard 的 NLI 逻辑（如果可用），
    否则用关键词匹配模拟。
    """
    claims = _split_claims(answer)
    if not claims:
        return 1.0

    combined_docs = " ".join(retrieved_docs)
    if not combined_docs.strip():
        return 0.0

    supported = 0
    for claim in claims:
        if nli_checker:
            entailed = nli_checker.check_entailment(retrieved_docs, claim)
        else:
            entailed = _simple_entailment(combined_docs, claim)
        if entailed:
            supported += 1

    return supported / len(claims)


def compute_relevance(answer: str, question: str,
                      encoder=None) -> float:
    """Answer Relevance: 答案与问题的语义相似度。"""
    if encoder:
        import numpy as np
        embs = encoder.encode([question, answer], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    return _simple_relevance(answer, question)


def compute_rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L: 最长公共子序列 F1。

    局限性说明: ROUGE-L 基于词重叠，不能衡量语义等价。
    仅作为参考答案与生成答案的 token 级相似度参考。
    """
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    lcs_len = _lcs_length(cand_tokens, ref_tokens)

    if not cand_tokens or not ref_tokens:
        return 0.0

    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ---- Helpers ----

def _tokenize(text: str) -> list[str]:
    """中文简单分词。"""
    tokens = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            tokens.append(ch)
        elif ch.isalnum():
            tokens.append(ch)
    return tokens


def _lcs_length(a: list, b: list) -> int:
    """最长公共子序列长度。"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _split_claims(text: str) -> list[str]:
    parts = re.split(r'(?<=[。！？\n])', text)
    return [p.strip() for p in parts if p.strip()]


def _simple_entailment(docs: str, claim: str) -> bool:
    numbers = re.findall(r'\d+', claim)
    for n in numbers:
        if n not in docs:
            return False
    key_terms = re.findall(r'[\u4e00-\u9fff]{2,}', claim)
    if key_terms:
        if not any(t in docs for t in key_terms):
            return False
    return bool(numbers) or bool(key_terms)


def _simple_relevance(answer: str, question: str) -> float:
    q_chars = set(question)
    a_chars = set(answer)
    if not q_chars:
        return 1.0
    return len(q_chars & a_chars) / len(q_chars)


# ---- Aggregate ----

@dataclass
class EvalMetrics:
    """单条用例的评估指标。"""
    question: str = ""
    hit_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    rouge_l: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question[:200],
            "hit_at_k": self.hit_at_k,
            "mrr": round(self.mrr, 4),
            "faithfulness": round(self.faithfulness, 4),
            "relevance": round(self.relevance, 4),
            "rouge_l": round(self.rouge_l, 4),
            **self.metadata,
        }
