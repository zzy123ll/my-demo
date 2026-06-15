"""离线评估运行器。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import EvalConfig
from .golden_dataset import GoldenDataset, TestCase
from .metrics import (
    compute_hit_at_k, compute_mrr, compute_faithfulness,
    compute_relevance, compute_rouge_l, EvalMetrics,
)


@dataclass
class EvaluationReport:
    dataset_name: str
    total_cases: int
    avg_hit_at_k: dict[int, float]
    avg_mrr: float
    avg_faithfulness: float
    avg_relevance: float
    avg_rouge_l: float
    per_case: list[EvalMetrics]
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "total_cases": self.total_cases,
            "avg_hit_at_k": self.avg_hit_at_k,
            "avg_mrr": round(self.avg_mrr, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_relevance": round(self.avg_relevance, 4),
            "avg_rouge_l": round(self.avg_rouge_l, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class OfflineEvaluator:
    """离线评估运行器。

    遍历 GoldenDataset，对每条用例模拟检索+生成，
    计算所有指标并汇总输出报告。
    """

    def __init__(self, config: EvalConfig = None,
                 retriever=None, generator=None, nli_checker=None):
        self.config = config or EvalConfig()
        self.retriever = retriever
        self.generator = generator
        self.nli_checker = nli_checker

    def evaluate(self, dataset: GoldenDataset) -> EvaluationReport:
        t0 = time.perf_counter()
        per_case = []

        for case in dataset:
            # 模拟检索
            retrieved = self._retrieve(case.question)
            retrieved_ids = [r.get("chunk_id", r.get("doc_id", ""))
                           for r in retrieved]
            retrieved_texts = [r.get("content", "") for r in retrieved]

            # 模拟生成
            answer = self._generate(case.question, retrieved_texts)

            # 计算指标
            hit = compute_hit_at_k(retrieved_ids, case.relevant_doc_ids,
                                   self.config.hit_k_values)
            mrr = compute_mrr(retrieved_ids, case.relevant_doc_ids,
                             self.config.mrr_k)
            faith = compute_faithfulness(answer, retrieved_texts,
                                        self.nli_checker)
            rel = compute_relevance(answer, case.question)
            rouge = compute_rouge_l(answer, case.ground_truth_answer)

            per_case.append(EvalMetrics(
                question=case.question,
                hit_at_k=hit,
                mrr=mrr,
                faithfulness=faith,
                relevance=rel,
                rouge_l=rouge,
            ))

        elapsed = time.perf_counter() - t0
        return self._aggregate(dataset, per_case, elapsed)

    def _retrieve(self, question: str) -> list[dict]:
        if self.retriever:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.retriever.search(question)
                )
            except Exception:
                pass
        return []

    def _generate(self, question: str, docs: list[str]) -> str:
        if self.generator:
            return self.generator.generate(question, docs)
        return ""

    def _aggregate(self, dataset, per_case, elapsed) -> EvaluationReport:
        n = len(per_case)
        if n == 0:
            return EvaluationReport(dataset_name="empty", total_cases=0,
                                     avg_hit_at_k={}, avg_mrr=0.0,
                                     avg_faithfulness=0.0, avg_relevance=0.0,
                                     avg_rouge_l=0.0, per_case=[],
                                     elapsed_seconds=0.0)

        # 汇总 Hit@k
        avg_hit = {}
        for k in self.config.hit_k_values:
            avg_hit[k] = sum(c.hit_at_k.get(k, 0) for c in per_case) / n

        return EvaluationReport(
            dataset_name=getattr(dataset, '_path', 'dataset'),
            total_cases=n,
            avg_hit_at_k=avg_hit,
            avg_mrr=sum(c.mrr for c in per_case) / n,
            avg_faithfulness=sum(c.faithfulness for c in per_case) / n,
            avg_relevance=sum(c.relevance for c in per_case) / n,
            avg_rouge_l=sum(c.rouge_l for c in per_case) / n,
            per_case=per_case,
            elapsed_seconds=elapsed,
        )
