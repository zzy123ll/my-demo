"""索引更新通知机制：后台异步监听文档变更，同步更新向量和 BM25 索引。"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .bm25_retriever import BM25Retriever, BM25Document
from .vector_retriever import VectorRetriever, VectorDocument


logger = logging.getLogger(__name__)


class IndexStatus(Enum):
    IDLE = "idle"
    SYNCING = "syncing"
    ERROR = "error"


@dataclass
class PendingDocument:
    """待索引的文档。"""
    doc_id: str
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    version: int = 0
    action: str = "add"
    timestamp: float = 0.0


class IndexManager:
    """索引管理器。"""

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever,
                 sync_interval: float = 10.0):
        self._bm25 = bm25
        self._vector = vector
        self._sync_interval = sync_interval
        self._pending: list[PendingDocument] = []
        self._lock = threading.Lock()
        self._status = IndexStatus.IDLE
        self._last_sync_time: float = 0.0
        self._processed_versions: set[int] = set()
        self._pending_versions: set[int] = set()
        self._chunk_to_doc: dict[str, str] = {}
        self._task: Optional[asyncio.Task] = None
        self._on_sync_callbacks: list[Callable] = []

    # ---- 文档操作 ----

    def enqueue_add(self, doc_id: str, chunk_id: str, content: str,
                    metadata: dict = None, version: int = 0) -> None:
        with self._lock:
            if version in self._processed_versions or version in self._pending_versions:
                return
            self._pending_versions.add(version)
            self._pending.append(PendingDocument(
                doc_id=doc_id, chunk_id=chunk_id, content=content,
                metadata=metadata or {}, version=version, action="add",
                timestamp=__import__('time').time(),
            ))
            self._chunk_to_doc[chunk_id] = doc_id

    def enqueue_delete(self, chunk_id: str, doc_id: str = "",
                       version: int = 0) -> None:
        with self._lock:
            self._pending.append(PendingDocument(
                doc_id=doc_id, chunk_id=chunk_id, content="",
                version=version, action="delete",
            ))

    # ---- 同步逻辑 ----

    async def start(self) -> None:
        if self._task is not None:
            return

        async def _loop():
            while True:
                await asyncio.sleep(self._sync_interval)
                await self._sync_pending()

        self._task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def sync_now(self) -> int:
        return await self._sync_pending()

    async def _sync_pending(self) -> int:
        with self._lock:
            if not self._pending:
                return 0
            batch = list(self._pending)
            self._pending.clear()
            self._pending_versions.clear()

        self._status = IndexStatus.SYNCING
        count = 0

        try:
            bm25_docs = []
            vector_docs = []
            to_delete = []

            for doc in batch:
                if doc.action == "delete":
                    to_delete.append(doc)
                else:
                    bm25_docs.append(BM25Document(
                        doc_id=doc.doc_id, chunk_id=doc.chunk_id,
                        content=doc.content,
                        metadata={**doc.metadata, "_version": doc.version},
                    ))
                    vector_docs.append(VectorDocument(
                        doc_id=doc.doc_id, chunk_id=doc.chunk_id,
                        content=doc.content,
                        metadata={**doc.metadata, "_version": doc.version},
                    ))

            if bm25_docs:
                self._bm25.add_batch(bm25_docs)
                count += len(bm25_docs)

            if vector_docs:
                self._vector.add_batch(vector_docs)

            for doc in to_delete:
                target_id = self._chunk_to_doc.get(doc.chunk_id, doc.doc_id)
                self._bm25.remove_document(target_id)
                self._vector.remove_document(doc.chunk_id)
                count += 1

            for doc in batch:
                self._processed_versions.add(doc.version)

            self._last_sync_time = __import__('time').time()
            self._status = IndexStatus.IDLE

            for cb in self._on_sync_callbacks:
                try:
                    cb(count)
                except Exception:
                    pass

        except Exception as e:
            self._status = IndexStatus.ERROR
            logger.error(f"Index sync failed: {e}")
            with self._lock:
                self._pending = batch + self._pending

        return count

    # ---- 状态查询 ----

    def get_status(self) -> IndexStatus:
        return self._status

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def get_versions(self) -> dict:
        return {
            "bm25": self._bm25.get_version(),
            "vector": self._vector.get_version(),
            "last_sync": self._last_sync_time,
            "pending": self.pending_count(),
            "status": self._status.value,
        }

    def on_sync(self, callback: Callable) -> None:
        self._on_sync_callbacks.append(callback)
