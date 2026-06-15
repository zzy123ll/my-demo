"""NLI 蕴含检测：判断文档(premise)是否蕴含声明(hypothesis)。"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class NLIChecker:
    """基于 HuggingFace NLI 模型的蕴含检测器。

    使用 multilingual NLI 模型（如 mDeBERTa-v3-base-mnli-xnli），
    支持中英文。模型延迟加载，避免启动时阻塞。

    标签映射:
    - label=0: contradiction (矛盾)
    - label=1: neutral (中立)
    - label=2: entailment (蕴含)
    """

    # DeBERTa NLI 模型的标签顺序
    NLI_LABELS = ["contradiction", "neutral", "entailment"]

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                 entailment_threshold: float = 0.5,
                 contradiction_threshold: float = 0.3,
                 batch_size: int = 16,
                 max_length: int = 512):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        """延迟加载模型。"""
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info(f"Loading NLI model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.eval()

    def check_entailment(self, premise_docs: list[str],
                         hypothesis_statement: str) -> bool:
        """检查是否有文档蕴含该声明。

        Args:
            premise_docs: 检索到的文档块列表（前提）
            hypothesis_statement: 单个原子声明（假设）

        Returns:
            True: 至少有一个文档蕴含该声明
            False: 所有文档均不蕴含
        """
        self._ensure_loaded()

        if not premise_docs or not hypothesis_statement.strip():
            return False

        # 为每个 (premise, hypothesis) 对计算 NLI 分数
        for i in range(0, len(premise_docs), self.batch_size):
            batch = premise_docs[i:i + self.batch_size]
            pairs = [(doc, hypothesis_statement) for doc in batch]

            # Tokenize
            inputs = self._tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

                # 检查 entailment 分数
                entailment_scores = probs[:, 2].tolist()

                if any(s >= self.entailment_threshold for s in entailment_scores):
                    return True

        return False

    def check_entailment_detailed(self, premise_docs: list[str],
                                  hypothesis_statement: str) -> dict:
        """详细版：返回每条文档的 NLI 结果。"""
        self._ensure_loaded()

        if not premise_docs or not hypothesis_statement.strip():
            return {"entailed": False, "best_score": 0.0, "per_doc": []}

        best_score = 0.0
        per_doc = []

        for i in range(0, len(premise_docs), self.batch_size):
            batch = premise_docs[i:i + self.batch_size]
            pairs = [(doc, hypothesis_statement) for doc in batch]

            inputs = self._tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

                for j, doc in enumerate(batch):
                    prob = probs[j].tolist()
                    entail_score = prob[2]
                    best_score = max(best_score, entail_score)
                    per_doc.append({
                        "doc_preview": doc[:200],
                        "entailment": round(entail_score, 4),
                        "neutral": round(prob[1], 4),
                        "contradiction": round(prob[0], 4),
                    })

        return {
            "entailed": best_score >= self.entailment_threshold,
            "best_score": round(best_score, 4),
            "per_doc": per_doc,
        }
