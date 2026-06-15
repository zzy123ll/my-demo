"""用户反馈收集接口 — POST /feedback。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeedbackRecord:
    feedback_id: str
    message_id: str
    session_id: str
    user_id: str
    rating: int   # 1-5
    comment: str = ""
    question: str = ""
    answer: str = ""
    source_docs: list[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "feedback_id": self.feedback_id,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "comment": self.comment,
            "question": self.question[:500],
            "answer": self.answer[:500],
            "created_at": self.created_at,
        }


class FeedbackCollector:
    """反馈收集器。

    API:
        POST /feedback
        {
            "message_id": "...",
            "session_id": "...",
            "rating": 4,
            "comment": "回答很准确",
            "question": "...",
            "answer": "..."
        }
    """

    def __init__(self, db_path: str = "data/feedback.jsonl"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[FeedbackRecord] = []

    def submit(self, message_id: str, rating: int,
               session_id: str = "", user_id: str = "",
               comment: str = "", question: str = "",
               answer: str = "",
               source_docs: list[str] = None) -> FeedbackRecord:
        if not 1 <= rating <= 5:
            raise ValueError("rating must be 1-5")

        record = FeedbackRecord(
            feedback_id=f"FB-{uuid.uuid4().hex[:8].upper()}",
            message_id=message_id,
            session_id=session_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            question=question,
            answer=answer,
            source_docs=source_docs or [],
            created_at=time.time(),
        )

        self._records.append(record)
        self._append_to_file(record)

        return record

    def _append_to_file(self, record: FeedbackRecord) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def get_stats(self) -> dict:
        if not self._records:
            return {"total": 0, "avg_rating": 0.0}
        ratings = [r.rating for r in self._records]
        return {
            "total": len(ratings),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "positive_rate": round(sum(1 for r in ratings if r >= 4) / len(ratings), 4),
            "negative_rate": round(sum(1 for r in ratings if r <= 2) / len(ratings), 4),
        }

    def load_history(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._records.append(FeedbackRecord(**obj))
