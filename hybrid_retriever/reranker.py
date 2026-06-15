"""Cross-encoder 重排序。使用 sentence-transformers 的 CrossEncoder，batch 处理。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import RetrieverConfig

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder 重排序器。

    sentence-transformers CrossEncoder 的参数从 config 读取。
    模型延迟加载，避免模块导入时触发损坏的 torch。
    """

    def __init__(self, config: RetrieverConfig):
        self._config = config
        self._model = None

    @property
    def model(self):
        """延迟加载 CrossEncoder 模型。"""
        if self._model is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"Loading CrossEncoder: {self._config.reranker_model}")
            self._model = CrossEncoder(
                self._config.reranker_model,
                max_length=self._config.reranker_max_length,
            )
        return self._model

    def rerank(self, query: str,
               candidates: list[dict]) -> list[dict]:
        """对融合后的候选列表进行重排序。

        Args:
            query: 用户查询
            candidates: 融合后的候选列表 (含 chunk_id, content, fusion_score)

        Returns:
            按 rerank_score 降序排列的候选列表
        """
        if not candidates:
            return []

        batch_size = self._config.reranker_batch_size

        # 构造 (query, document) 对
        pairs = [[query, c["content"]] for c in candidates]

        # 批量推理
        all_scores = []
        try:
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                batch_scores = self.model.predict(batch, show_progress_bar=False)
                if hasattr(batch_scores, 'tolist'):
                    batch_scores = batch_scores.tolist()
                all_scores.extend(batch_scores)
        except Exception as e:
            logger.error(f"CrossEncoder rerank failed: {e}")
            # 降级：用融合分数排序
            ranked = sorted(candidates,
                          key=lambda x: x.get("fusion_score", 0),
                          reverse=True)
            for c in ranked:
                c["rerank_score"] = c.get("fusion_score", 0)
                c["rerank_source"] = "fallback_fusion"
            return ranked

        # 赋值
        for i, c in enumerate(candidates):
            c["rerank_score"] = round(
                float(all_scores[i]) if i < len(all_scores) else 0.0, 4
            )
            c["pre_rerank_score"] = c.get("fusion_score", 0)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates

    def is_loaded(self) -> bool:
        return self._model is not None
