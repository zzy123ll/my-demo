"""Escalation Handler 配置，从 YAML + .env 加载。"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path


def _load_yaml() -> dict:
    p = Path(__file__).resolve().parent / "escalation_rules.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_env_file() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = current / ".env"
        if env_path.exists():
            return env_path
        current = current.parent
    fallback = Path("E:/wolkplace/.env")
    if fallback.exists():
        return fallback
    return Path(".")


def _load_dotenv() -> None:
    env_path = _find_env_file()
    if env_path is None or not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


@dataclass
class EscalationConfig:
    escalation_keywords: list[str]
    hallucination_reject_threshold: int = 2
    negative_sentiment_threshold: int = 3
    sentiment_threshold: float = -0.3
    sentiment_model: str = "rule_based"
    ticket_ttl_minutes: int = 60
    max_queue_size: int = 1000
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL1", "deepseek-v4-flash"))

    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url)


def load_escalation_config() -> EscalationConfig:
    raw = _load_yaml()
    return EscalationConfig(
        escalation_keywords=raw.get("escalation_keywords", ["人工", "转人工"]),
        hallucination_reject_threshold=raw.get("hallucination_reject_threshold", 2),
        negative_sentiment_threshold=raw.get("negative_sentiment_threshold", 3),
        sentiment_threshold=raw.get("sentiment_threshold", -0.3),
    )
