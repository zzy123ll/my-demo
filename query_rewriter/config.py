"""从 wolkplace/.env 文件中加载所有模型配置，不在代码中硬编码任何 key/model 名称。"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _find_env_file() -> Path:
    """向上查找 .env 文件，直到 wolkplace 根目录。"""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = current / ".env"
        if env_path.exists():
            return env_path
        current = current.parent
    # 兜底：直接在 wolkplace 下查找
    fallback = Path("E:/wolkplace/.env")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Cannot find .env file. Expected at E:/wolkplace/.env"
    )


def _load_dotenv() -> None:
    """手动解析 .env 文件，避免依赖 python-dotenv。"""
    env_path = _find_env_file()
    if env_path is None or not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# 模块加载时自动读取 .env
_load_dotenv()


@dataclass
class RewriterConfig:
    """Query Rewriter 的配置，全部从环境变量读取。"""

    # LLM 配置（使用 .env 中的 LLM_MODEL1，即轻量模型）
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL1", "deepseek-v4-flash")
    )

    # 句子相似度模型
    sentence_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # 一致性校验阈值
    similarity_threshold: float = 0.85

    # 历史消息窗口大小
    max_history_rounds: int = 5

    # LLM 请求超时（秒）
    llm_timeout: float = 10.0

    # LLM 最大输出 token
    llm_max_tokens: int = 256

    # 是否启用 LLM 改写（false 时仅用规则）
    enable_llm_rewrite: bool = True

    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url)


def load_config() -> RewriterConfig:
    """工厂方法：从环境变量加载配置。"""
    return RewriterConfig()
