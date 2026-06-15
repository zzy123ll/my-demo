"""转接触发规则引擎。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TriggerReason(Enum):
    KEYWORD = "user_keyword"
    HALLUCINATION = "hallucination_reject"
    SAFETY_APPEAL = "safety_appeal"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    MANUAL = "manual"


@dataclass
class TriggerResult:
    should_escalate: bool
    reason: str = ""
    trigger_type: TriggerReason | None = None
    context: dict = field(default_factory=dict)


class TriggerEngine:
    """转接触发规则引擎。

    检测条件 (可配置):
    1. 用户消息含转接关键词
    2. 同一对话中 Hallucination Guard 拒答达到阈值
    3. Safety Enforcer 拦截但用户申诉
    4. 连续 N 轮负面情绪
    """

    def __init__(self, config):
        self.config = config
        self._session_reject_count: dict[str, int] = {}
        self._session_sentiment_history: dict[str, list[float]] = {}

    def evaluate(self, session_id: str, user_message: str,
                 safety_verdict: dict = None,
                 hallucination_verdict: dict = None,
                 sentiment_score: float = 0.0
                 ) -> TriggerResult:
        """评估所有触发条件，返回是否需要转接。"""

        # 1. 关键词触发
        for kw in self.config.escalation_keywords:
            if kw in user_message:
                return TriggerResult(
                    should_escalate=True,
                    reason=f'用户消息包含转接关键词: "{kw}"',
                    trigger_type=TriggerReason.KEYWORD,
                    context={"keyword": kw},
                )

        # 2. Hallucination Guard 连续拒答
        if hallucination_verdict and hallucination_verdict.get("verdict") == "reject":
            cnt = self._session_reject_count.get(session_id, 0) + 1
            self._session_reject_count[session_id] = cnt
            if cnt >= self.config.hallucination_reject_threshold:
                return TriggerResult(
                    should_escalate=True,
                    reason=f"Hallucination Guard 拒答 {cnt} 次 (阈值 {self.config.hallucination_reject_threshold})",
                    trigger_type=TriggerReason.HALLUCINATION,
                    context={"reject_count": cnt},
                )
        else:
            self._session_reject_count[session_id] = 0

        # 3. Safety Enforcer 拦截 + 用户申诉
        if safety_verdict and safety_verdict.get("decision") == "BLOCK":
            appeal_keywords = ["为什么", "凭什么", "我要投诉", "不合理", "哪里敏感"]
            if any(kw in user_message for kw in appeal_keywords):
                return TriggerResult(
                    should_escalate=True,
                    reason="Safety Enforcer 拦截后用户申诉",
                    trigger_type=TriggerReason.SAFETY_APPEAL,
                    context={"safety_category": safety_verdict.get("matched_category", "")},
                )

        # 4. 负面情绪连续 N 轮
        if sentiment_score <= self.config.sentiment_threshold:
            history = self._session_sentiment_history.get(session_id, [])
            history.append(sentiment_score)
            self._session_sentiment_history[session_id] = history
            if len(history) >= self.config.negative_sentiment_threshold:
                if all(s <= self.config.sentiment_threshold for s in history[-self.config.negative_sentiment_threshold:]):
                    return TriggerResult(
                        should_escalate=True,
                        reason=f"连续 {self.config.negative_sentiment_threshold} 轮负面情绪",
                        trigger_type=TriggerReason.NEGATIVE_SENTIMENT,
                        context={"sentiment_history": history[-5:]},
                    )
        else:
            history = self._session_sentiment_history.get(session_id, [])
            history.append(sentiment_score)
            self._session_sentiment_history[session_id] = history[-self.config.negative_sentiment_threshold:]

        return TriggerResult(should_escalate=False)

    def reset_session(self, session_id: str) -> None:
        self._session_reject_count.pop(session_id, None)
        self._session_sentiment_history.pop(session_id, None)
