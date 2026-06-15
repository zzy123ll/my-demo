"""Safety Enforcer 配置，从 YAML + .env 加载。"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


def _load_yaml_config() -> dict:
    yaml_path = Path(__file__).resolve().parent / "sensitive_categories.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
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
class SensitiveCategory:
    name: str
    risk_level: str   # critical / high / medium / low
    description: str
    keywords: list[str]
    regex_patterns: list[re.Pattern]
    action: str       # block / block_and_escalate / warn

    def match_keywords(self, text: str) -> list[str]:
        hits = []
        for kw in self.keywords:
            if kw in text:
                hits.append(kw)
        return hits

    def match_regex(self, text: str) -> list[str]:
        hits = []
        for pat in self.regex_patterns:
            m = pat.search(text)
            if m:
                hits.append(m.group())
        return hits


@dataclass
class EnforcerConfig:
    categories: dict[str, SensitiveCategory]
    access_control: dict
    notifications: dict
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    l2_similarity_threshold: float = 0.7
    l3_timeout: float = 2.0
    audit_enabled: bool = True


def load_enforcer_config() -> EnforcerConfig:
    raw = _load_yaml_config()

    categories = {}
    for name, cat_data in raw.get("categories", {}).items():
        patterns = [re.compile(p) for p in cat_data.get("regex", [])]
        categories[name] = SensitiveCategory(
            name=name,
            risk_level=cat_data.get("risk_level", "medium"),
            description=cat_data.get("description", ""),
            keywords=cat_data.get("keywords", []),
            regex_patterns=patterns,
            action=cat_data.get("action", "block"),
        )

    return EnforcerConfig(
        categories=categories,
        access_control=raw.get("access_control", {}),
        notifications=raw.get("notifications", {}),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_model=os.getenv("LLM_MODEL1", "deepseek-v4-flash"),
    )
