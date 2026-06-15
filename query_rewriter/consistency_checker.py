"""一致性校验：用 sentence-transformers 计算语义相似度，验证改写后查询没有改变原意。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    """一致性校验的输出。"""
    passed: bool                    # 是否通过校验
    score_original: float           # 改写句与原句的相似度
    score_context: float            # 改写句与上下文的相似度（可选）
    threshold: float               # 使用的阈值
    verdict: str                   # "pass" | "low_similarity" | "semantic_drift"
    detail: str = ""               # 人类可读的说明


class ConsistencyChecker:
    """基于 sentence-transformers 的语义一致性校验。

    计算两个维度的相似度：
    1. rewritten vs original: 确保改写没有改变原意
    2. rewritten vs context: 确保改写与对话上下文相关（可选）

    默认模型为 paraphrase-multilingual-MiniLM-L12-v2，支持中文。
    模型名称可从 config.sentence_model 读取。
    """

    def __init__(self, config):
        self.config = config
        self._model = None
        self._threshold = config.similarity_threshold

    @property
    def model(self):
        """延迟加载 sentence-transformers 模型。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.sentence_model
            logger.info(f"Loading sentence-transformers model: {model_name}")
            self._model = SentenceTransformer(model_name)
        return self._model

    def check(self, original_query: str, rewritten_query: str,
              context_text: str = "") -> ConsistencyResult:
        """执行一致性校验。

        Args:
            original_query: 用户原始查询
            rewritten_query: 改写后的查询
            context_text: 对话上下文文本（可选）

        Returns:
            ConsistencyResult: 校验结果
        """
        # 如果改写结果与原句完全相同，直接通过
        if rewritten_query.strip() == original_query.strip():
            return ConsistencyResult(
                passed=True,
                score_original=1.0,
                score_context=1.0,
                threshold=self._threshold,
                verdict="pass",
                detail="改写结果与原句相同，无需校验",
            )

        # 编码句子
        sentences = [original_query, rewritten_query]
        if context_text:
            sentences.append(context_text)

        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        emb_original = embeddings[0]
        emb_rewritten = embeddings[1]

        # 计算余弦相似度（embedding 已归一化，点积 = 余弦相似度）
        score_original = float(np.dot(emb_original, emb_rewritten))

        # 如果有上下文，也计算改写句与上下文的相似度
        score_context = 1.0
        if context_text and len(embeddings) > 2:
            emb_context = embeddings[2]
            score_context = float(np.dot(emb_rewritten, emb_context))

        # 综合判定
        # 主要看与原句的相似度，上下文相似度作为辅助
        if score_original >= self._threshold:
            return ConsistencyResult(
                passed=True,
                score_original=round(score_original, 4),
                score_context=round(score_context, 4),
                threshold=self._threshold,
                verdict="pass",
                detail=f"语义相似度 {score_original:.3f} >= 阈值 {self._threshold}",
            )
        elif score_original >= 0.6:
            # 在 0.6 ~ 0.85 之间，标记为低相似度但不失败
            return ConsistencyResult(
                passed=False,
                score_original=round(score_original, 4),
                score_context=round(score_context, 4),
                threshold=self._threshold,
                verdict="low_similarity",
                detail=f"语义相似度 {score_original:.3f} < 阈值 {self._threshold}，存在一定偏差",
            )
        else:
            return ConsistencyResult(
                passed=False,
                score_original=round(score_original, 4),
                score_context=round(score_context, 4),
                threshold=self._threshold,
                verdict="semantic_drift",
                detail=f"语义相似度 {score_original:.3f} 严重偏低，疑似改写改变了原意",
            )

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """计算两个文本的语义相似度（0-1）。"""
        embeddings = self.model.encode(
            [text_a, text_b], normalize_embeddings=True
        )
        return float(np.dot(embeddings[0], embeddings[1]))
