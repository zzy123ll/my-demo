"""GoldenDataset: 从 JSONL 加载测试用例。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestCase:
    question: str
    ground_truth_answer: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class GoldenDataset:
    """黄金标准测试集。

    JSONL 格式:
    {"question": "...", "ground_truth_answer": "...", "relevant_doc_ids": ["d1","d2"]}
    """

    def __init__(self, path: str = None):
        self._cases: list[TestCase] = []
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._cases.append(TestCase(
                    question=obj["question"],
                    ground_truth_answer=obj.get("ground_truth_answer", ""),
                    relevant_doc_ids=obj.get("relevant_doc_ids", []),
                    metadata=obj.get("metadata", {}),
                ))

    def add(self, case: TestCase) -> None:
        self._cases.append(case)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for case in self._cases:
                f.write(json.dumps({
                    "question": case.question,
                    "ground_truth_answer": case.ground_truth_answer,
                    "relevant_doc_ids": case.relevant_doc_ids,
                    "metadata": case.metadata,
                }, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self):
        return iter(self._cases)

    def __getitem__(self, idx):
        return self._cases[idx]

    def filter(self, predicate) -> list[TestCase]:
        return [c for c in self._cases if predicate(c)]
