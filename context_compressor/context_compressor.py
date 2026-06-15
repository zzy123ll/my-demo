"""Context Compressor 主编排器：集成提取式/生成式压缩 + 完整性校验 + 降级。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import CompressorConfig, load_compressor_config
from .token_budget import TokenBudgetManager
from .extractive import ExtractiveCompressor
from .generative import GenerativeCompressor
from .integrity_checker import IntegrityChecker, IntegrityResult

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """压缩操作的最终输出。"""
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    original_tokens: int
    compressed_tokens: int
    mode: str  # "extractive" | "generative" | "fallback_truncation"
    latency_ms: float
    integrity: Optional[IntegrityResult] = None
    fallback_reason: str = ""
    metadata: dict = field(default_factory=dict)


class ContextCompressor:
    """上下文压缩器。

    流程:
    1. TokenBudgetManager 计算可用 token
    2. 根据 mode 选择压缩策略:
       - extractive: 相似度选句
       - generative: LLM 提取
    3. IntegrityChecker 验证实体保留率
    4. 实体丢失时：触发更宽松策略（增加 k 或重试）
    5. 压缩失败时：截断降级，保证可用
    """

    def __init__(self, config: CompressorConfig = None, llm=None):
        self.config = config or load_compressor_config()
        self.token_budget = TokenBudgetManager(
            context_window=self.config.context_window,
            reserved_tokens=self.config.reserved_tokens,
        )
        self.extractive = ExtractiveCompressor(
            top_k=self.config.extractive_top_k_sentences,
            model_name=self.config.extractive_similarity_model,
        )
        self.generative = GenerativeCompressor(
            llm=llm, config=self.config
        ) if llm else None
        self.integrity = IntegrityChecker(
            threshold=self.config.entity_preservation_threshold,
        )

    async def compress(self, document: str, query: str = "",
                       system_prompt: str = "") -> CompressionResult:
        """主入口：压缩文档。

        Args:
            document: 待压缩的文档
            query: 用户问题（用于相关度评估）
            system_prompt: 系统提示词（用于 token 预算计算）

        Returns:
            CompressionResult: 压缩结果
        """
        t_start = time.perf_counter()

        # 空文档处理
        if not document or not document.strip():
            return CompressionResult(
                compressed_text="", original_length=0,
                compressed_length=0, compression_ratio=1.0,
                original_tokens=0, compressed_tokens=0,
                mode="passthrough", latency_ms=0,
            )

        # 计算 token 预算
        available = self.token_budget.available_tokens(system_prompt, query)
        doc_tokens = self.token_budget.count_tokens(document)
        original_len = len(document)

        # 文档本身就在预算内 → 无需压缩
        if doc_tokens <= available:
            latency = (time.perf_counter() - t_start) * 1000
            return CompressionResult(
                compressed_text=document,
                original_length=original_len,
                compressed_length=original_len,
                compression_ratio=1.0,
                original_tokens=doc_tokens,
                compressed_tokens=doc_tokens,
                mode="passthrough",
                latency_ms=round(latency, 1),
                metadata={"available_tokens": available},
            )

        # 选择压缩策略
        if self.config.mode == "generative" and self.generative:
            result = await self._compress_generative(document, query, available)
        else:
            result = await self._compress_extractive(document, query, available)

        # 完整性检查
        integrity = self.integrity.check(document, result.compressed_text)

        if not integrity.passed and integrity.missing_entities:
            logger.warning(
                f"Integrity check failed: missing {len(integrity.missing_entities)} entities. "
                f"Retrying with more lenient compression..."
            )
            # 更宽松的策略
            result = await self._compress_extractive(
                document, query, available, override_k=10, integrity=integrity
            )

        latency = (time.perf_counter() - t_start) * 1000

        return CompressionResult(
            compressed_text=result.compressed_text,
            original_length=original_len,
            compressed_length=len(result.compressed_text),
            compression_ratio=(len(result.compressed_text) / original_len
                              if original_len > 0 else 1.0),
            original_tokens=doc_tokens,
            compressed_tokens=self.token_budget.count_tokens(result.compressed_text),
            mode=result.mode,
            latency_ms=round(latency, 1),
            integrity=integrity,
            metadata={"available_tokens": available},
        )

    async def _compress_extractive(self, document: str, query: str,
                                   available_tokens: int,
                                   override_k: int = None,
                                   integrity: IntegrityResult = None
                                   ) -> CompressionResult:
        """提取式压缩。"""
        try:
            backup_k = self.extractive.top_k
            if override_k:
                self.extractive.top_k = override_k

            ext_result = self.extractive.compress_with_budget(
                document, query, self.token_budget, available_tokens
            )
            self.extractive.top_k = backup_k

            if ext_result.compressed_text.strip():
                return CompressionResult(
                    compressed_text=ext_result.compressed_text,
                    original_length=len(document),
                    compressed_length=len(ext_result.compressed_text),
                    compression_ratio=ext_result.compression_ratio,
                    original_tokens=self.token_budget.count_tokens(document),
                    compressed_tokens=self.token_budget.count_tokens(ext_result.compressed_text),
                    mode="extractive",
                    latency_ms=0,
                )
        except Exception as e:
            logger.warning(f"Extractive compression failed: {e}")

        # 降级：截断
        return self._fallback_truncate(document, "extractive failed")

    async def _compress_generative(self, document: str, query: str,
                                   available_tokens: int
                                   ) -> CompressionResult:
        """生成式压缩。"""
        try:
            gen_result = await self.generative.compress(document, query)

            if gen_result.compressed_text.strip():
                return CompressionResult(
                    compressed_text=gen_result.compressed_text,
                    original_length=gen_result.original_length,
                    compressed_length=gen_result.compressed_length,
                    compression_ratio=gen_result.compression_ratio,
                    original_tokens=self.token_budget.count_tokens(document),
                    compressed_tokens=self.token_budget.count_tokens(gen_result.compressed_text),
                    mode="generative",
                    latency_ms=gen_result.latency_ms,
                    metadata={"model": gen_result.model_used},
                )
        except TimeoutError as e:
            logger.warning(f"Generative compression timeout: {e}")
            return self._fallback_truncate(document, f"LLM timeout: {e}")
        except RuntimeError as e:
            logger.warning(f"Generative compression failed: {e}")
            return self._fallback_truncate(document, f"LLM error: {e}")

        return self._fallback_truncate(document, "generative returned empty")

    def _fallback_truncate(self, document: str, reason: str
                           ) -> CompressionResult:
        """截断降级：取前 N 个字符，保证系统可用。"""
        max_chars = self.config.fallback_max_chars
        truncated = document[:max_chars]

        logger.warning(f"Fallback truncation ({reason}): {len(document)} -> {len(truncated)} chars")

        return CompressionResult(
            compressed_text=truncated,
            original_length=len(document),
            compressed_length=len(truncated),
            compression_ratio=len(truncated) / len(document) if document else 1.0,
            original_tokens=self.token_budget.count_tokens(document),
            compressed_tokens=self.token_budget.count_tokens(truncated),
            mode="fallback_truncation",
            latency_ms=0,
            fallback_reason=reason,
        )

    def compress_sync(self, document: str, query: str = "",
                      system_prompt: str = "") -> CompressionResult:
        """同步压缩接口。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        return loop.run_until_complete(
            self.compress(document, query, system_prompt)
        )
