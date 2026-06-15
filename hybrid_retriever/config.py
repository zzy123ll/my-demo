"""Hybrid Retriever 配置，全部从 wolkplace/.env 隐式读取。"""

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
class RetrieverConfig:
    """检索器配置。"""

    # 融合策略: "rrf" 或 "weighted"
    fusion_strategy: str = "rrf"

    # RRF 参数
    rrf_k: int = 60

    # 加权融合参数 (仅 fusion_strategy="weighted" 时使用)
    weighted_alpha: float = 0.5   # 0=全BM25, 1=全向量

    # 各路召回数
    top_k_per_path: int = 20
    fusion_top_k: int = 100
    final_top_k: int = 10

    # Cross-encoder 模型 (sentence-transformers)
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
    )
    reranker_batch_size: int = 32
    reranker_max_length: int = 512

    # 向量检索模型
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
    )

    # ChromaDB 持久化路径
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR", "./chroma_data"
        )
    )

    # BM25 参数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # 索引更新检查间隔（秒）
    index_sync_interval: float = 10.0

    # 性能指标
    metrics_enabled: bool = True


def load_retriever_config() -> RetrieverConfig:
    return RetrieverConfig()
