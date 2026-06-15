"""阶段一：基于规则的指代消解。处理中文常见的代词和省略模式。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .conversation_state import ConversationState, EntityInfo


@dataclass
class RuleResult:
    """规则引擎的输出。"""
    resolved: bool
    rewritten_query: str
    method: str = "rule"
    confidence: float = 1.0
    matched_pattern: str = ""
    matched_pronoun: str = ""


# 中文代词映射 - 直接匹配字符，不用 `\b`（对中文无效）
PRONOUN_PATTERNS = {
    "它": {
        "chars": "它",
        "target_type": "any",
        "prefer_non_person": True,
    },
    "他": {
        "chars": "他",
        "target_type": "person",
        "prefer_non_person": False,
    },
    "她": {
        "chars": "她",
        "target_type": "person",
        "prefer_non_person": False,
    },
    "这个": {
        "chars": "这个",
        "target_type": "any",
        "prefer_non_person": True,
    },
    "那个": {
        "chars": "那个",
        "target_type": "any",
        "prefer_non_person": True,
    },
    "这些": {
        "chars": "这些",
        "target_type": "any",
        "prefer_non_person": True,
    },
    "那些": {
        "chars": "那些",
        "target_type": "any",
        "prefer_non_person": True,
    },
    "它们": {
        "chars": "它们",
        "target_type": "any",
        "prefer_non_person": True,
    },
}

# 省略句式 - 更具体的模式放前面
ELLIPSIS_PATTERNS = [
    # "还有吗？" / "还有别的吗？"
    (r'^(?:还\s*)?有\s*(?:别的|其它|其他)?[吗么]?\s*[？?]?\s*$',
     "supplement_request"),
    # "具体(怎么|如何)...?"
    (r'^具体\s*(?:怎么|如何|怎样)\s*([^？?]*)[？?]?\s*$',
     "detail_inquiry"),
    # "那X呢？" -> topic_ellipsis
    (r'^(?:那|那么)\s*(.+?)\s*[呢吗吧啊]?\s*[？?]?\s*$',
     "topic_ellipsis"),
    # "X呢？"
    (r'^(.+?)[呢吗吧啊]\s*[？?]?\s*$',
     "topic_ellipsis"),
]


class CoreferenceResolver:
    """阶段一的规则引擎：基于中文代词和句式的指代消解。"""

    def __init__(self):
        self._compiled_pronouns = {}
        for name, info in PRONOUN_PATTERNS.items():
            self._compiled_pronouns[name] = re.compile(
                info["chars"], re.IGNORECASE
            )

    def resolve(self, query: str, state: ConversationState) -> RuleResult | None:
        query = query.strip()

        # 1. 先检查代词（避免"他的...呢？"被省略句式拦截）
        result = self._handle_pronouns(query, state)
        if result:
            return result

        # 2. 再检查省略句式
        result = self._handle_ellipsis(query, state)
        if result:
            return result

        return None

    def _handle_ellipsis(self, query: str, state: ConversationState) -> Optional[RuleResult]:
        for pattern, strategy in ELLIPSIS_PATTERNS:
            match = re.match(pattern, query)
            if not match:
                continue

            if strategy == "supplement_request":
                if state.last_topic:
                    rewritten = f"关于{state.last_topic}还有哪些信息？"
                    return RuleResult(
                        resolved=True, rewritten_query=rewritten,
                        method="rule", confidence=0.85,
                        matched_pattern="supplement_request",
                        matched_pronoun="还/还有",
                    )

            elif strategy == "detail_inquiry":
                target = match.group(1).strip() if match.lastindex else ""
                topic = state.last_topic or ""
                rewritten = f"{topic}具体如何{target}？"
                return RuleResult(
                    resolved=True, rewritten_query=rewritten,
                    method="rule", confidence=0.88,
                    matched_pattern="detail_inquiry",
                    matched_pronoun="具体",
                )

            elif strategy == "topic_ellipsis":
                keyword = match.group(1).strip() if match.lastindex else ""
                if not keyword:
                    continue
                if len(keyword) <= 1 and keyword in ("还", "有"):
                    continue

                if state.last_topic:
                    attr = self._extract_attribute_question(state.last_user_query)
                    rewritten = f"{keyword}{attr}"
                    return RuleResult(
                        resolved=True, rewritten_query=rewritten,
                        method="rule", confidence=0.9,
                        matched_pattern="topic_ellipsis",
                        matched_pronoun=keyword,
                    )
                else:
                    rewritten = f"{keyword}是什么？"
                    return RuleResult(
                        resolved=True, rewritten_query=rewritten,
                        method="rule", confidence=0.7,
                        matched_pattern="topic_ellipsis",
                        matched_pronoun=keyword,
                    )

        return None

    def _handle_pronouns(self, query: str, state: ConversationState) -> Optional[RuleResult]:
        for pronoun_name, pattern_info in PRONOUN_PATTERNS.items():
            compiled = self._compiled_pronouns[pronoun_name]
            if compiled.search(query):
                target = self._find_referent(pattern_info, state)
                if target:
                    rewritten = compiled.sub(target.name, query)
                    return RuleResult(
                        resolved=True,
                        rewritten_query=rewritten,
                        method="rule",
                        confidence=0.9,
                        matched_pattern=f"pronoun_{pronoun_name}",
                        matched_pronoun=pronoun_name,
                    )
        return None

    def _find_referent(self, pattern_info: dict, state: ConversationState) -> Optional[EntityInfo]:
        if not state.entity_map:
            if state.last_topic:
                return EntityInfo(
                    name=state.last_topic, type="topic",
                    first_mentioned_round=0, last_mentioned_round=0,
                    last_mention=state.last_topic,
                )
            return None

        target_type = pattern_info.get("target_type", "any")
        prefer_non_person = pattern_info.get("prefer_non_person", True)

        if target_type == "person":
            return state.find_latest_entity_by_type("person")

        entities = list(state.entity_map.values())
        if prefer_non_person:
            non_person = [e for e in entities if e.type != "person"]
            if non_person:
                return max(non_person, key=lambda e: e.last_mentioned_round)

        if entities:
            return max(entities, key=lambda e: e.last_mentioned_round)
        return None

    def _extract_attribute_question(self, query: str) -> str:
        attr_patterns = [
            r'有(?:多少|几[个天次]|什么|哪些).*[？?]?',
            r'的(?:流程|标准|规定|政策|制度|方法|步骤|要求).*[？?]?',
            r'怎么(?:申请|办理|操作|处理|计算).*[？?]?',
            r'是什么.*[？?]?',
            r'如何(?:申请|办理|操作|处理|计算).*[？?]?',
        ]
        for pat in attr_patterns:
            m = re.search(pat, query)
            if m:
                return m.group()
        return "是什么？"
