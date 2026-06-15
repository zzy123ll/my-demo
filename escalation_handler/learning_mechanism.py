"""学习机制 — 设计层面的接口定义和伪代码。

当坐席 resolve 工单时，将解决方案与触发问题关联存储。
后续遇到相似问题时优先检索此方案，降低转接率。

实现方式:
1. 向量化: 将问题 + 解决方案拼接后计算 embedding
2. 存储: ChromaDB 或向量数据库 (复用 hybrid_retriever)
3. 检索: 新问题 → embedding → 相似度检索 → 返回最接近的历史方案
4. 反馈: 如果人工方案被采纳且有效，提升权重
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LearnedSolution:
    solution_id: str
    query: str
    solution: str
    solution_tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    use_count: int = 0
    success_rate: float = 0.0
    created_at: str = ""


class LearningStore(ABC):
    """学习存储抽象接口。

    可对接:
    - ChromaDB (向量检索)
    - PostgreSQL (结构化查询)
    - Elasticsearch (全文检索)
    """

    @abstractmethod
    def store(self, solution: LearnedSolution) -> None:
        """存储解决方案。"""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[LearnedSolution]:
        """检索历史解决方案。"""
        ...

    @abstractmethod
    def feedback(self, solution_id: str, was_helpful: bool) -> None:
        """用户/坐席反馈。"""
        ...


class ChromaLearningStore(LearningStore):
    """基于 ChromaDB 的学习存储实现 (伪代码)。"""

    def store(self, solution: LearnedSolution) -> None:
        # embedding_fn.encode(solution.query + " " + solution.solution)
        # collection.add(ids=[solution_id], documents=[text], embeddings=[emb])
        pass

    def search(self, query: str, top_k: int = 3) -> list[LearnedSolution]:
        # emb = embedding_fn.encode(query)
        # results = collection.query(query_embeddings=[emb], n_results=top_k)
        # return [LearnedSolution(...) for r in results]
        pass

    def feedback(self, solution_id: str, was_helpful: bool) -> None:
        # UPDATE solutions SET use_count++, success_rate = ... WHERE id = solution_id
        pass


class LearningPipeline:
    """学习管线接口 — 在 resolve 工单时触发。

    伪代码流程:

    def on_ticket_resolve(ticket):
        # 1. 提取问题-解决方案对
        query = ticket.user_query
        solution = ticket.resolution

        # 2. 向量化
        text = f"Q: {query}\nA: {solution}"
        emb = embedding_model.encode(text)

        # 3. 存储
        learned = LearnedSolution(
            solution_id=f"SOL-{uuid4().hex[:8]}",
            query=query,
            solution=solution,
            solution_tags=ticket.solution_tags,
            embedding=emb,
        )
        learning_store.store(learned)

        # 4. 后续检索 (在新问题到来时)
        # similar = learning_store.search(new_query, top_k=3)
        # if similar and similar[0].success_rate > 0.7:
        #     return similar[0].solution  # 直接复用，降低转接率
    """
    pass
