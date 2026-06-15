"""提取式压缩：基于句子相似度选择最相关的句子。"""

from __future__ import annotations

import re
import logging
import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractiveResult:
    """提取式压缩结果。"""
    compressed_text: str
    selected_sentence_indices: list[int]
    similarity_scores: list[float]
    original_sentence_count: int
    compression_ratio: float


class ExtractiveCompressor:
    """基于句子相似度的提取式压缩。

    对每个文档块：
    1. 分句
    2. 计算每句与问题的语义相似度
    3. 选择 top-k 个最相关的句子
    """

    def __init__(self, top_k: int = 5,
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.top_k = top_k
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """延迟加载 sentence-transformers 模型。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading similarity model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compress(self, document: str, query: str) -> ExtractiveResult:
        """提取式压缩。

        Args:
            document: 原文文档
            query: 用户问题

        Returns:
            ExtractiveResult: 压缩结果
        """
        sentences = self._split_sentences(document)
        if not sentences:
            return ExtractiveResult(
                compressed_text="", selected_sentence_indices=[],
                similarity_scores=[], original_sentence_count=0,
                compression_ratio=1.0,
            )

        if len(sentences) <= self.top_k:
            return ExtractiveResult(
                compressed_text=document,
                selected_sentence_indices=list(range(len(sentences))),
                similarity_scores=[1.0] * len(sentences),
                original_sentence_count=len(sentences),
                compression_ratio=1.0,
            )

        # 计算相似度
        texts_to_encode = [query] + sentences
        embeddings = self.model.encode(texts_to_encode, normalize_embeddings=True)
        query_emb = embeddings[0]
        sentence_embs = embeddings[1:]

        scores = [
            float(np.dot(query_emb, sent_emb))
            for sent_emb in sentence_embs
        ]

        # 选 top-k
        indices = list(range(len(sentences)))
        pairs = list(zip(indices, scores))
        selected = heapq.nlargest(self.top_k, pairs, key=lambda x: x[1])
        selected.sort()  # 按原文顺序排列

        selected_idx = [i for i, _ in selected]
        selected_scores = [s for _, s in selected]

        compressed = "".join(sentences[i] for i in selected_idx)
        original_len = len(document)
        compressed_len = len(compressed)

        return ExtractiveResult(
            compressed_text=compressed,
            selected_sentence_indices=selected_idx,
            similarity_scores=selected_scores,
            original_sentence_count=len(sentences),
            compression_ratio=(compressed_len / original_len
                              if original_len > 0 else 1.0),
        )

    def compress_with_budget(self, document: str, query: str,
                             token_budget: 'TokenBudgetManager',
                             available_tokens: int) -> ExtractiveResult:
        """带 token 预算的压缩。先取 top-k 句子，超出预算时减少 k。"""
        original_k = self.top_k

        for k in range(original_k, 0, -1):
            self.top_k = k
            result = self.compress(document, query)
            tokens = token_budget.count_tokens(result.compressed_text)
            if tokens <= available_tokens:
                return result

        # 最后兜底：只取最相关的 1 句
        self.top_k = 1
        result = self.compress(document, query)
        self.top_k = original_k
        return result

    def _split_sentences(self, text: str) -> list[str]:
        """中文分句。在 。！？\n 处切分，保留分隔符。"""
        # 按标点或换行分句
        parts = re.split(r'(?<=[。！？\n])', text)
        parts = [p for p in parts if p.strip()]
        return parts
