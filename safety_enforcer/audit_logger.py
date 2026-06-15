"""审计日志 + 异步 Slack/邮件通知。"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class AuditLogger:
    """审计日志记录器。

    记录格式:
    {
      "timestamp": "ISO 8601",
      "user_id": "...",
      "department": "...",
      "original_query": "...",
      "decision": "PASS|BLOCK|WARN",
      "triggered_by": "L1|L2|L3",
      "matched_category": "...",
      "score": 0.85,
      "action": "block|block_and_escalate|warn",
      "escalated": bool,
      "session_id": "...",
    }
    """

    def __init__(self, enabled: bool = True, webhook_config: dict = None):
        self.enabled = enabled
        self.webhook_config = webhook_config or {}
        self._logs: list[dict] = []  # 内存日志（生产环境应写 DB）

    def log(self, user_id: str, department: str,
            original_query: str, decision: str,
            triggered_by: str, matched_category: str,
            score: float, action: str,
            session_id: str = "", extra: dict = None) -> None:
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "department": department,
            "original_query": original_query[:500],
            "decision": decision,
            "triggered_by": triggered_by,
            "matched_category": matched_category,
            "score": round(score, 4),
            "action": action,
            "escalated": action == "block_and_escalate",
            "session_id": session_id,
            **(extra or {}),
        }

        self._logs.append(entry)
        logger.info(
            f"SAFETY [{decision}] user={user_id} cat={matched_category} "
            f"score={score:.2f} trigger={triggered_by}"
        )

        # 异步通知
        if action in ("block", "block_and_escalate"):
            try:
                asyncio.ensure_future(self._notify(entry))
            except RuntimeError:
                pass  # no event loop (e.g. in tests)

    async def _notify(self, entry: dict) -> None:
        """异步发送通知到 webhook。"""
        webhook_url = self.webhook_config.get("slack_webhook", "")
        if not webhook_url:
            return

        try:
            import aiohttp
            payload = {
                "text": (
                    f"*Safety Alert* [{entry['decision']}]\n"
                    f"User: {entry['user_id']} ({entry['department']})\n"
                    f"Query: {entry['original_query'][:200]}\n"
                    f"Category: {entry['matched_category']}\n"
                    f"Score: {entry['score']:.2f} | Trigger: {entry['triggered_by']}"
                )
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload,
                                       timeout=5) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Webhook failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    def get_recent_logs(self, n: int = 100) -> list[dict]:
        return self._logs[-n:]

    def get_stats(self) -> dict:
        if not self._logs:
            return {"total": 0}
        blocks = sum(1 for l in self._logs if l["decision"] == "BLOCK")
        return {
            "total": len(self._logs),
            "blocks": blocks,
            "block_rate": round(blocks / len(self._logs), 4),
        }
