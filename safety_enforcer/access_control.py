"""基于角色的知识域访问控制。"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class AccessController:
    """基于 JWT 部门解析的文档访问控制。

    从用户 JWT 中解析 department 字段，
    在检索前对文档元数据过滤，只返回该部门有权查看的文档。
    """

    def __init__(self, access_policies: dict):
        """
        Args:
            access_policies: {
                "departments": {
                    "HR": {"allowed_doc_tags": [...], "min_level": 0},
                    ...
                }
            }
        """
        self._policies = access_policies.get("departments", {})

    def parse_jwt(self, token: str) -> dict:
        """解析 JWT（仅 payload 部分，不验证签名）。

        生产环境应使用 PyJWT 并验证签名。
        """
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return {"department": "unknown"}
            payload = parts[1]
            # 补齐 base64 padding
            payload += "=" * (4 - len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            return json.loads(decoded)
        except Exception:
            return {"department": "unknown"}

    def get_department(self, token: str = "",
                       user_context: dict = None) -> str:
        """从 JWT 或 user_context 中提取部门。"""
        if user_context and user_context.get("department"):
            return user_context["department"]
        if token:
            claims = self.parse_jwt(token)
            return claims.get("department", "unknown")
        return "unknown"

    def get_allowed_tags(self, department: str) -> list[str]:
        """获取部门有权访问的文档标签。"""
        dept_policy = self._policies.get(department, {})
        tags = dept_policy.get("allowed_doc_tags", [])

        # "all" 表示无限制
        if "all" in tags:
            return ["__ALL__"]

        return tags

    def filter_documents(self, docs: list[dict],
                         department: str) -> list[dict]:
        """过滤文档列表，只返回部门有权查看的。

        Args:
            docs: [{"doc_id": ..., "metadata": {"tags": [...], ...}}, ...]
            department: 用户部门

        Returns:
            过滤后的文档列表
        """
        allowed_tags = self.get_allowed_tags(department)

        # 全权限
        if allowed_tags == ["__ALL__"]:
            return docs

        # 未知部门 → 只返回 general 标签文档
        if department == "unknown":
            allowed_tags = ["general"]

        filtered = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            doc_tags = metadata.get("tags", [])

            # 无标签文档 → 默认视为 general
            if not doc_tags:
                doc_tags = ["general"]

            # 检查是否有交集
            if set(doc_tags) & set(allowed_tags):
                filtered.append(doc)

        logger.debug(
            f"Access filter: {department} → {len(filtered)}/{len(docs)} docs"
        )
        return filtered

    def get_chroma_where_filter(self, department: str) -> dict | None:
        """生成 ChromaDB 的 where 过滤条件。"""
        allowed_tags = self.get_allowed_tags(department)

        if allowed_tags == ["__ALL__"]:
            return None
        if department == "unknown":
            allowed_tags = ["general"]

        # ChromaDB where 条件
        if len(allowed_tags) == 1:
            return {"tags": {"$contains": allowed_tags[0]}}
        else:
            return {"$or": [
                {"tags": {"$contains": tag}} for tag in allowed_tags
            ]}
