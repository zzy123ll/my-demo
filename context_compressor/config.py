"""Context Compressor 配置，全部从 wolkplace/.env 隐式读取。"""

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
    return None


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
class CompressorConfig:
    """压缩器配置。"""

    # 压缩模式: "extractive" 或 "generative"
    mode: str = "extractive"

    # 模型上下文窗口大小
    context_window: int = 4096  # 或 8192

    # 预留给系统提示和回答的 token 数
    reserved_tokens: int = 500

    # 提取式压缩参数
    extractive_top_k_sentences: int = 5     # 每个文档块保留的句子数
    extractive_similarity_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # 生成式压缩参数 (使用 .env 中的 LLM_MODEL1)
    generative_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL1", "deepseek-v4-flash")
    )
    generative_max_tokens: int = 512
    generative_temperature: float = 0.0
    generative_timeout: float = 15.0  # 秒

    # LLM API 配置
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "")
    )

    # 完整性检查
    entity_preservation_threshold: float = 0.85  # 实体保留率阈值

    # 降级参数
    fallback_max_chars: int = 2000  # 截断降级时的字符数

    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url)


def load_compressor_config() -> CompressorConfig:
    return CompressorConfig()
