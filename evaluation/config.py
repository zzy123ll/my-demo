"""Evaluation 配置。"""

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    dataset_path: str = "data/golden_dataset.jsonl"
    hit_k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    mrr_k: int = 10
    faithfulness_threshold: float = 0.5
    relevance_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    feedback_db_path: str = "data/feedback.jsonl"


def load_eval_config() -> EvalConfig:
    return EvalConfig()
