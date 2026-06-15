"""ChromaDB 向量检索封装。支持 OpenAI 兼容 API 做 embedding，避免本地 DLL 依赖。"""

from __future__ import annotations

import os
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """向量索引中的文档。"""
    doc_id: str
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)


def _create_openai_ef():
    """从 .env 创建 OpenAI 兼容的 embedding function。"""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    if not api_key or not base_url:
        return None
    try:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base_url=base_url,
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
        )
    except Exception as e:
        logger.warning(f"Cannot create OpenAIEmbeddingFunction: {e}")
        return None


class VectorRetriever:
    """基于 ChromaDB 的向量检索引擎。

    Embedding 策略（按优先级）：
    1. 如果提供 embedding_fn 参数，直接使用
    2. 如果 .env 中配置了 LLM_API_KEY，使用 OpenAI 兼容 API
    3. 否则使用 ChromaDB 内置 ONNX embedding（需要 onnxruntime 可用）
    """

    def __init__(self, collection_name: str = "rag_docs",
                 persist_dir: str = "./chroma_data",
                 embedding_fn=None):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name

        # 自动选择 embedding function
        if embedding_fn is not None:
            self._embedding_fn = embedding_fn
        else:
            self._embedding_fn = _create_openai_ef()

        self._collection = None
        self._version: int = 0
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._collection = self._client.get_collection(
                self._collection_name,
                embedding_function=self._embedding_fn,
            )
        except Exception:
            self._collection = self._client.create_collection(
                self._collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    # ---- Index Management ----

    def add_document(self, doc: VectorDocument) -> None:
        self._collection.add(
            ids=[doc.chunk_id],
            documents=[doc.content],
            metadatas=[{**doc.metadata, "doc_id": doc.doc_id}],
        )
        self._version += 1

    def add_batch(self, docs: list[VectorDocument]) -> None:
        if not docs:
            return
        ids = [d.chunk_id for d in docs]
        documents = [d.content for d in docs]
        metadatas = [{**d.metadata, "doc_id": d.doc_id} for d in docs]
        self._collection.add(
            ids=ids, documents=documents, metadatas=metadatas,
        )
        self._version += 1

    def remove_document(self, chunk_id: str) -> None:
        self._collection.delete(ids=[chunk_id])
        self._version += 1

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._ensure_collection()
        self._version = 0

    # ---- Search ----

    def search(self, query: str, top_k: int = 20,
               where: dict = None) -> list[dict]:
        try:
            kwargs = {"query_texts": [query], "n_results": top_k}
            if where:
                kwargs["where"] = where

            results = self._collection.query(**kwargs)

            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            output = []
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(ids)):
                distance = distances[i] if i < len(distances) else 1.0
                similarity = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)

                output.append({
                    "chunk_id": ids[i],
                    "doc_id": metadatas[i].get("doc_id", "") if i < len(metadatas) else "",
                    "content": documents[i] if i < len(documents) else "",
                    "score": round(similarity, 4),
                    "source": "vector",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })
            return output
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def get_version(self) -> int:
        return self._version

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0
