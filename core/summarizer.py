"""
summarizer.py
-------------
Summarizes top-ranked chunks into concise insights or answers.
"""

from loguru import logger
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI


class Summarizer:
    def __init__(self, temperature: float = 0.4):
        self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
        logger.info("✅ Summarizer LLM initialized (Gemini 2.5 Flash).")

    def summarize(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return "No context found to summarize."

        context_text = "\n\n".join(contexts[:10])
        logger.info(f"🧠 Summarizing {len(contexts)} chunks for query: {query}")

        prompt = f"""
You are a research summarization agent.
Summarize the following context to directly answer the user's query.

Query:
{query}

Context:
{context_text}

Return a concise, factual summary (2–3 paragraphs max).
"""
        try:
            response = self.llm.invoke(prompt)
            summary = response.content.strip()
            return summary
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            return "Failed to summarize context."


if __name__ == "__main__":
    from core.reranker import Reranker
    from core.retriever_selector import RetrieverSelector

    query = "Emerging AI trends for 2025 in developer tools"
    retriever = RetrieverSelector()
    reranker = Reranker()
    summarizer = Summarizer()

    docs = retriever.retrieve(query)
    ranked = reranker.rerank(query, docs)
    contexts = [r["text"] for r in ranked]
    summary = summarizer.summarize(query, contexts)
    print("\n--- FINAL SUMMARY ---\n", summary)
