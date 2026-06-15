"""TokenBudgetManager: 根据模型上下文窗口动态计算可用 token 数。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TokenBudgetManager:
    """Token 预算管理器。

    使用 tiktoken 做精确计数，降级时用字符近似估算。
    计算公式: available = context_window - reserved_tokens - system_prompt - current_input
    """

    def __init__(self, context_window: int = 4096,
                 reserved_tokens: int = 500,
                 encoding_name: str = "cl100k_base"):
        self.context_window = context_window
        self.reserved_tokens = reserved_tokens
        self.encoding_name = encoding_name
        self._encoder = None

    @property
    def encoder(self):
        """延迟加载 tiktoken encoder。"""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding(self.encoding_name)
            except Exception as e:
                logger.warning(f"tiktoken not available, using char-based estimate: {e}")
                self._encoder = None
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数。"""
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        # 降级：中英文混合估算
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """基于字符的粗略 token 估算。"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        # 中文: ~1.5 char/token, 英文: ~4 char/token
        return int(chinese_chars / 1.5 + other_chars / 4)

    def available_tokens(self, system_prompt: str = "",
                         current_input: str = "") -> int:
        """计算当前可用的 token 数。

        available = context_window - reserved - system_prompt - current_input

        Args:
            system_prompt: 系统提示词文本
            current_input: 当前已占用的输入文本（如用户问题）

        Returns:
            可用于文档上下文的 token 数，最小为 0
        """
        used = self.reserved_tokens
        if system_prompt:
            used += self.count_tokens(system_prompt)
        if current_input:
            used += self.count_tokens(current_input)

        available = self.context_window - used
        return max(0, available)

    def budget_for_chunks(self, chunks: list[str],
                          system_prompt: str = "",
                          query: str = "",
                          per_chunk_min: int = 50) -> list[int]:
        """为每个文档块分配 token 预算。

        策略：等比例分配，保证每块至少有 per_chunk_min token。

        Args:
            chunks: 文档块文本列表
            system_prompt: 系统提示词
            query: 用户查询
            per_chunk_min: 每块最小 token 数

        Returns:
            每块的 token 预算列表
        """
        total_available = self.available_tokens(system_prompt, query)

        if not chunks:
            return []

        chunk_sizes = [self.count_tokens(c) for c in chunks]
        total_chunk_tokens = sum(chunk_sizes)

        if total_chunk_tokens <= total_available:
            return chunk_sizes  # 全部保留

        # 等比例分配
        budgets = []
        remaining = total_available
        for i, size in enumerate(chunk_sizes[:-1]):
            proportion = size / total_chunk_tokens
            budget = max(per_chunk_min, int(proportion * total_available))
            budget = min(budget, size, remaining)
            budgets.append(budget)
            remaining -= budget

        # 最后一块拿到剩余的全部
        budgets.append(min(remaining, chunk_sizes[-1]))

        return budgets

    def fits_in_window(self, text: str, system_prompt: str = "",
                       query: str = "") -> bool:
        """检查文本是否适配上下文窗口。"""
        tokens = (self.count_tokens(system_prompt) +
                  self.count_tokens(query) +
                  self.count_tokens(text) +
                  self.reserved_tokens)
        return tokens <= self.context_window

    def summary(self) -> dict:
        return {
            "context_window": self.context_window,
            "reserved_tokens": self.reserved_tokens,
            "encoding": self.encoding_name,
        }
