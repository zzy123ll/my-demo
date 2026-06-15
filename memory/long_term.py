
class LongTermMemory:
    """长时记忆: LLM 压缩超量上下文为摘要."""
    def __init__(self, llm=None):
        self._summaries = {}
        self._llm = llm

    async def compress(self, session_id: str, context: str) -> str:
        if self._llm:
            try:
                from langchain_core.messages import HumanMessage
                prompt = f"请用一段话总结以下对话:
{context}

只输出摘要(不超过80字):"
                r = await self._llm.ainvoke([HumanMessage(content=prompt)])
                summary = r.content.strip()[:120]
                self._summaries[session_id] = summary
                return summary
            except: pass
        summary = context[:200] + "..."
        self._summaries[session_id] = summary
        return summary

    def get(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")
