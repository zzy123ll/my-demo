"""阶段二：LLM 改写。当规则引擎无法处理时，调用轻量级 LLM 做指代消解和省略补全。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from langchain_core.messages import SystemMessage, HumanMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


REWRITE_SYSTEM_PROMPT = """你是一个查询改写专家。你的任务是基于对话历史，将用户的当前问题改写为一个独立、完整的查询，以便搜索引擎或知识库检索。

【改写规则】
1. 将代词（它、他、她、这个、那个、这些、那些）替换为对话历史中明确的实体名称。
2. 将省略的部分补齐。例如"那A呢？"应改为"那A的[属性]呢？"，属性从历史中提取。
3. 如果问题引用了上文的某条信息，明确写出该信息。
4. 保留用户的原始语气和意图，不要添加用户没说过的信息。
5. 不要编造历史中没有的信息。
6. 如果问题与历史完全无关（是一个全新的话题），原样返回，不要强行关联。

【输出格式】
请严格按照以下 JSON 格式输出，不要输出任何其他内容：
{"rewritten_query": "改写后的查询语句", "strategy": "coreference|ellipsis|inherit|passthrough", "confidence": 0.0-1.0}

strategy 说明：
- coreference: 消解了指代词
- ellipsis: 补全了省略内容
- inherit: 继承了上文的主题或属性
- passthrough: 无需改写，直接透传
"""


@dataclass
class LLMRewriteResult:
    rewritten_query: str
    strategy: str
    confidence: float
    raw_response: str = ""


class LLMRewriter:
    """阶段二的 LLM 改写器。

    使用 .env 中配置的 LLM_MODEL1（轻量模型）做改写。
    模型初始化参数全部从环境变量读取，不在代码中硬编码。
    """

    def __init__(self, llm, config):
        self.llm = llm
        self.config = config
        self._system_prompt = REWRITE_SYSTEM_PROMPT

    async def rewrite(self, query: str, state) -> LLMRewriteResult | None:
        if not self.config.enable_llm_rewrite:
            return None
        if not self.config.is_llm_configured():
            return None

        history_text = self._format_history(state)

        user_prompt = f"""【对话历史】
{history_text}

【用户当前问题】
{query}

请改写当前问题。"""

        try:
            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await self.llm.ainvoke(messages)
            parsed = self._parse_response(response.content)

            return LLMRewriteResult(
                rewritten_query=parsed.get("rewritten_query", query),
                strategy=parsed.get("strategy", "passthrough"),
                confidence=parsed.get("confidence", 0.5),
                raw_response=response.content,
            )
        except Exception:
            return None

    def _format_history(self, state) -> str:
        recent = state.get_recent_messages(self.config.max_history_rounds)
        if not recent:
            return "（无历史对话）"
        lines = []
        for msg in recent:
            role = "用户" if msg.type == "human" else "助手"
            content = msg.content[:500]
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict:
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except (json.JSONDecodeError, ValueError):
                pass
        return {
            "rewritten_query": text.strip()[:512],
            "strategy": "passthrough",
            "confidence": 0.4,
        }
