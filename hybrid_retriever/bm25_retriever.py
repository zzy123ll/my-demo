"""纯 Python BM25 实现（无外部依赖），用于稀疏检索。"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BM25Document:
    """BM25 索引中的文档。"""
    doc_id: str
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)


class BM25Retriever:
    """内存 BM25 检索引擎。

    支持增量添加文档、删除文档、重建索引。
    使用标准 BM25 公式，k1 和 b 参数可配置。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: dict[str, BM25Document] = {}
        self._doc_ids: list[str] = []          # 按添加顺序
        self._tokenized_docs: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._term_freqs: dict[str, dict[int, int]] = defaultdict(dict)
        self._total_docs: int = 0
        self._dirty: bool = True
        self._version: int = 0

    # ---- Tokenization ----

    _CHINESE_CHAR_PAT = re.compile(r'[\u4e00-\u9fff]')
    _WORD_PAT = re.compile(r'[a-zA-Z0-9]+')

    def _tokenize(self, text: str) -> list[str]:
        """简单的中英文混合分词。
        中文: 逐字切分 + 常见双字词
        英文: 按空格和字母数字切分
        """
        tokens = []
        i = 0
        while i < len(text):
            ch = text[i]
            if self._CHINESE_CHAR_PAT.match(ch):
                tokens.append(ch)
                i += 1
            elif self._WORD_PAT.match(ch):
                m = self._WORD_PAT.match(text, i)
                if m:
                    tokens.append(m.group().lower())
                    i = m.end()
                else:
                    i += 1
            else:
                i += 1

        # 生成双字词 (bi-gram) 增强中文召回
        chinese_chars = [t for t in tokens if len(t) == 1 and '\u4e00' <= t <= '\u9fff']
        for j in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[j] + chinese_chars[j + 1])

        return tokens

    # ---- Index Management ----

    def add_document(self, doc: BM25Document) -> None:
        """添加单个文档。"""
        self._documents[doc.doc_id] = doc
        self._doc_ids.append(doc.doc_id)
        tokens = self._tokenize(doc.content)
        self._tokenized_docs.append(tokens)
        self._doc_lengths.append(len(tokens))
        self._total_docs += 1
        self._dirty = True

    def add_batch(self, docs: list[BM25Document]) -> None:
        """批量添加文档。"""
        for doc in docs:
            self._documents[doc.doc_id] = doc
            self._doc_ids.append(doc.doc_id)
            tokens = self._tokenize(doc.content)
            self._tokenized_docs.append(tokens)
            self._doc_lengths.append(len(tokens))
        self._total_docs += len(docs)
        self._dirty = True

    def remove_document(self, doc_id: str) -> None:
        """删除文档。"""
        if doc_id in self._documents:
            idx = self._doc_ids.index(doc_id)
            self._documents.pop(doc_id)
            self._doc_ids.pop(idx)
            self._tokenized_docs.pop(idx)
            self._doc_lengths.pop(idx)
            self._total_docs -= 1
            self._dirty = True

    def clear(self) -> None:
        """清空索引。"""
        self._documents.clear()
        self._doc_ids.clear()
        self._tokenized_docs.clear()
        self._doc_lengths.clear()
        self._idf.clear()
        self._term_freqs.clear()
        self._total_docs = 0
        self._avgdl = 0.0
        self._dirty = True

    def _rebuild_if_needed(self) -> None:
        if not self._dirty:
            return
        self._build_index()
        self._dirty = False
        self._version += 1

    def _build_index(self) -> None:
        """(重)建 IDF 和词频索引。"""
        self._idf.clear()
        self._term_freqs.clear()

        if self._total_docs == 0:
            self._avgdl = 0.0
            return

        self._avgdl = sum(self._doc_lengths) / self._total_docs

        doc_freq: dict[str, int] = defaultdict(int)
        for doc_idx, tokens in enumerate(self._tokenized_docs):
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freq[term] += 1
            # 保存词频 (term freq per doc)
            freq = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            for t, f in freq.items():
                self._term_freqs[t][doc_idx] = f

        # 计算 IDF
        N = self._total_docs
        for term, df in doc_freq.items():
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # ---- Search ----

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 检索。"""
        self._rebuild_if_needed()

        if self._total_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self._total_docs

        for token in set(query_tokens):
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            qtf = query_tokens.count(token)

            for doc_idx, tf in self._term_freqs.get(token, {}).items():
                dl = self._doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_idx] += idf * (qtf * numerator / denominator)

        # 排序取 top_k
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for doc_idx, score in ranked:
            if score <= 0:
                continue
            doc_id = self._doc_ids[doc_idx]
            doc = self._documents[doc_id]
            results.append({
                "chunk_id": doc.chunk_id,
                "doc_id": doc.doc_id,
                "content": doc.content,
                "score": round(score, 4),
                "source": "bm25",
                "metadata": doc.metadata,
            })

        return results

    def get_version(self) -> int:
        return self._version

    def __len__(self) -> int:
        return self._total_docs
