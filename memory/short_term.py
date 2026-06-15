
from collections import deque, defaultdict

class ShortTermMemory:
    """滑动窗口短时记忆 (per session)."""
    def __init__(self, max_turns=10):
        self._store = defaultdict(lambda: deque(maxlen=max_turns))
        self._max = max_turns

    def add(self, session_id: str, role: str, content: str):
        self._store[session_id].append({"role": role, "content": content})

    def get(self, session_id: str, n: int = None):
        items = list(self._store.get(session_id, []))
        return items[-n:] if n else items

    def get_context(self, session_id: str) -> str:
        turns = self.get(session_id)
        if not turns: return ""
        return "\n".join(f"[{t['role']}]: {t['content'][:300]}" for t in turns)

    def exceeds_limit(self, session_id: str) -> bool:
        return len(self._store.get(session_id, [])) >= self._max

    def clear(self, session_id: str):
        self._store.pop(session_id, None)

    def summary_for_compression(self, session_id: str):
        return self.get_context(session_id)
