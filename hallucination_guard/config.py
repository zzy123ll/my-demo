"""Hallucination Guard 配置，全部从 wolkplace/.env 隐式读取。"""

import os
from dataclasses import dataclass, field
from pathlib import Path


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
    raise FileNotFoundError("Cannot find .env file")


def _load_dotenv() -> None:
    env_path = _find_env_file()
    if not env_path.exists():
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
class GuardConfig:
    """幻觉检测配置。"""

    # NLI 模型
    nli_model: str = field(
        default_factory=lambda: os.getenv(
            "NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        )
    )

    # 蕴含阈值：NLI entailment score >= 此值视为蕴含
    entailment_threshold: float = 0.5

    # 中立阈值：低于此值视为矛盾
    contradiction_threshold: float = 0.3

    # NLI 推理 batch size
    nli_batch_size: int = 16

    # NLI 最大输入长度
    nli_max_length: int = 512

    # 全部幻觉时的回复话术
    fallback_message: str = "抱歉，我无法根据现有知识确认这一点。"

    # LLM API（用于流式场景的撤回替换）
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "")
    )


def load_guard_config() -> GuardConfig:
    return GuardConfig()
