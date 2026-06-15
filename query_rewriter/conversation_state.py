"""ConversationState: 维护多轮对话的状态信息，集成 LangChain BaseMessage 历史管理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class UserProfile:
    user_id: str = ""
    department: str = ""
    level: str = ""
    name: str = ""


@dataclass
class EntityInfo:
    name: str
    type: str
    first_mentioned_round: int
    last_mentioned_round: int = 0
    last_mention: str = ""
    description: str = ""


@dataclass
class ConversationState:
    """多轮对话的完整状态。"""

    messages: list[BaseMessage] = field(default_factory=list)
    current_entity: Optional[EntityInfo] = None
    last_topic: str = ""
    user_profile: UserProfile = field(default_factory=UserProfile)
    entity_map: dict[str, EntityInfo] = field(default_factory=dict)
    round_count: int = 0
    last_user_query: str = ""
    last_assistant_answer: str = ""
    clarification_history: list[dict] = field(default_factory=list)
    _entity_counter: int = field(default=0, repr=False)

    # ---- 消息管理 ----
    def add_user_message(self, content: str) -> HumanMessage:
        msg = HumanMessage(content=content)
        self.messages.append(msg)
        self.round_count += 1
        self.last_user_query = content
        return msg

    def add_assistant_message(self, content: str) -> AIMessage:
        msg = AIMessage(content=content)
        self.messages.append(msg)
        self.last_assistant_answer = content
        return msg

    def get_recent_messages(self, n: int = 5) -> list[BaseMessage]:
        return self.messages[-(n * 2):]

    def get_last_user_message(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return None

    # ---- 实体管理 ----
    def track_entity(self, name: str, entity_type: str,
                     description: str = "") -> EntityInfo:
        self._entity_counter += 1
        if name in self.entity_map:
            entity = self.entity_map[name]
            entity.last_mentioned_round = self._entity_counter
            entity.last_mention = name
            self.current_entity = entity
            return entity

        entity = EntityInfo(
            name=name,
            type=entity_type,
            first_mentioned_round=self._entity_counter,
            last_mentioned_round=self._entity_counter,
            last_mention=name,
            description=description,
        )
        self.entity_map[name] = entity
        self.current_entity = entity
        return entity

    def find_entity(self, name: str) -> Optional[EntityInfo]:
        return self.entity_map.get(name)

    def find_latest_entity_by_type(self, entity_type: str) -> Optional[EntityInfo]:
        latest = None
        latest_round = -1
        for entity in self.entity_map.values():
            if entity.type == entity_type and entity.last_mentioned_round > latest_round:
                latest = entity
                latest_round = entity.last_mentioned_round
        return latest

    # ---- 主题管理 ----
    def update_topic(self, topic: str) -> None:
        if topic and topic != self.last_topic:
            self.last_topic = topic

    # ---- 澄清记录 ----
    def add_clarification(self, query: str, reason: str) -> None:
        self.clarification_history.append({
            "query": query,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    # ---- 摘要 ----
    def summarize(self) -> str:
        parts = []
        if self.last_topic:
            parts.append(f"当前话题: {self.last_topic}")
        if self.current_entity:
            parts.append(f"当前实体: {self.current_entity.name} ({self.current_entity.type})")
        entities = list(self.entity_map.keys())
        if entities:
            parts.append(f"已知实体: {', '.join(entities)}")
        parts.append(f"对话轮次: {self.round_count}")
        return " | ".join(parts)
