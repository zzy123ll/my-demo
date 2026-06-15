"""生成式压缩：面向问题的摘要，调用 LLM 保持原始措辞。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


GENERATIVE_SYSTEM_PROMPT = """你是一个信息压缩专家。你的任务是从文档中提取与用户问题直接相关的内容。

【核心规则】
1. 保持原始措辞：逐字保留原文句子，不要改写、概括或重新组织。
2. 只提取直接回答用户问题所需的信息。
3. 数字、日期、百分比、金额、姓名、政策编号 => 绝对保留，不要修改。
4. 不要添加原文中没有的信息。
5. 不要添加解释、评价或过渡语。

【输出格式】
按原文顺序输出提取的句子，每句保留原文措辞。如果没有相关内容，输出 [NO_RELEVANT_CONTENT]。"""


@dataclass
class GenerativeResult:
    """生成式压缩结果。"""
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    model_used: str
    latency_ms: float
    raw_response: str = ""


class GenerativeCompressor:
    """生成式压缩：调用 LLM 做面向问题的提取。

    使用 .env 中配置的 LLM_MODEL1（轻量模型）。
    支持超时和降级处理。
    """

    def __init__(self, llm=None, config=None):
        self.llm = llm
        self.config = config

    async def compress(self, document: str, query: str) -> GenerativeResult:
        """生成式压缩。

        Args:
            document: 原文文档
            query: 用户问题

        Returns:
            GenerativeResult: 压缩结果
        """
        t0 = time.perf_counter()

        if not self.llm or not self.config or not self.config.is_llm_configured():
            raise RuntimeError("LLM not configured for generative compression")

        from langchain_core.messages import SystemMessage, HumanMessage

        user_prompt = f"""【用户问题】
{query}

【文档内容】
{document}

请从以上文档中提取与用户问题直接相关的内容。"""

        messages = [
            SystemMessage(content=GENERATIVE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=self.config.generative_timeout,
            )
            extracted = response.content.strip()

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"LLM compression timed out after "
                f"{self.config.generative_timeout}s"
            )
        except Exception as e:
            raise RuntimeError(f"LLM compression failed: {e}")

        latency_ms = (time.perf_counter() - t0) * 1000

        # 清理输出
        if extracted == "[NO_RELEVANT_CONTENT]" or not extracted.strip():
            extracted = ""

        compressed_len = len(extracted)
        original_len = len(document)

        return GenerativeResult(
            compressed_text=extracted,
            original_length=original_len,
            compressed_length=compressed_len,
            compression_ratio=(compressed_len / original_len
                              if original_len > 0 else 1.0),
            model_used=self.config.generative_model,
            latency_ms=round(latency_ms, 1),
            raw_response=extracted,
        )

    def compress_with_truncation(self, document: str, query: str) -> str:
        """简单的规则提取降级（无 LLM 调用）。

        策略：提取含查询关键词的句子。
        """
        sentences = re.split(r'(?<=[。！？\n])', document)
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return document[:2000]

        # 提取查询中的关键词（去停顿词）
        keywords = set()
        for word in re.findall(r'[\u4e00-\u9fff]{2,}', query):
            keywords.add(word)
        for word in re.findall(r'[a-zA-Z]{3,}', query):
            keywords.add(word.lower())

        if not keywords:
            return document[:2000]

        relevant = [s for s in sentences
                   if any(kw in s.lower() for kw in keywords)]

        if relevant:
            return "".join(relevant)

        return document[:2000]
