"""L2: 基于 sentence-transformers 的敏感文本分类器。

计算查询与各敏感类别描述的语义相似度，返回 0-1 的敏感分数。
模拟 DistilBERT fine-tuned on toxicity + 企业敏感语料的行为。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class L2Result:
    """L2 检测结果。"""
    score: float            # 最高敏感分数 (0-1)
    top_category: str       # 最匹配的类别
    all_scores: dict[str, float]   # 各类别分数
    needs_l3: bool          # 是否需要 L3 裁决 (0.4-0.7)
    latency_us: float = 0.0


class L2TextClassifier:
    """L2 文本分类器。

    使用 sentence-transformers 计算查询与敏感类别描述的相似度。
    分数融合: embedding_similarity * 0.6 + keyword_overlap * 0.4
    """

    def __init__(self, categories, threshold: float = 0.7):
        self.categories = categories
        self.threshold = threshold
        self._model = None
        self._category_embeddings = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._model

    def _get_category_embeddings(self):
        if self._category_embeddings is None:
            descriptions = [
                f"{cat.name}: {cat.description}"
                for cat in self.categories.values()
            ]
            self._category_embeddings = self.model.encode(
                descriptions, normalize_embeddings=True
            )
        return self._category_embeddings

    def classify(self, query: str) -> L2Result:
        """分类查询。"""
        t0 = time.perf_counter()

        if not query.strip():
            return L2Result(score=0.0, top_category="", all_scores={},
                          needs_l3=False)

        cat_names = list(self.categories.keys())
        cat_list = list(self.categories.values())

        # 1. 关键词重叠分数
        keyword_scores = {}
        for cat in cat_list:
            hits = cat.match_keywords(query)
            unique_matches = len(set(hits))
            max_kw = max(len(cat.keywords), 1)
            keyword_scores[cat.name] = min(unique_matches / max_kw, 1.0)

        # 2. 嵌入相似度分数
        try:
            query_emb = self.model.encode([query], normalize_embeddings=True)[0]
            cat_embs = self._get_category_embeddings()
            sim_scores = np.dot(cat_embs, query_emb)

            for i, name in enumerate(cat_names):
                sim = float(np.clip(sim_scores[i], 0, 1))
                keyword_scores[name] = (
                    0.6 * sim + 0.4 * keyword_scores.get(name, 0)
                )
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}, using keyword only")
            # 纯关键词降级
            pass

        # 3. 找最高分
        best_cat = max(keyword_scores, key=keyword_scores.get)
        best_score = round(keyword_scores[best_cat], 4)

        latency = (time.perf_counter() - t0) * 1_000_000

        return L2Result(
            score=best_score,
            top_category=best_cat,
            all_scores={k: round(v, 4) for k, v in keyword_scores.items()},
            needs_l3=(0.4 <= best_score <= 0.7),
            latency_us=round(latency, 1),
        )
