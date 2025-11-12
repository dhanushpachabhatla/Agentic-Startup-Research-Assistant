"""
reranker.py
-----------
Re-ranks retrieved documents by semantic relevance to query.
"""

import os
import json
import re
from typing import List, Dict, Any
from loguru import logger
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import config

# Load Gemini key safely
if getattr(config, "GEMINI_API_KEY3", None):
    os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY3


class Reranker:
    def __init__(self, temperature: float = 0.3):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_tokens=None,
            timeout=120,
            max_retries=2,
        )
        logger.info("✅ Gemini reranker initialized (manual JSON parsing mode).")

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            logger.warning("No documents provided for reranking.")
            return []

        texts = [d.page_content[:500] for d in docs[:10]]
        logger.info(f"🔎 Reranking {len(texts)} documents for query: {query}")

        prompt = f"""
You are a ranking assistant. Rank these text chunks by how relevant they are to the query.

Output ONLY valid JSON in this exact format:
[
  {{ "text": "chunk content...", "score": 92 }},
  ...
]

Query:
{query}

Chunks:
{json.dumps(texts, ensure_ascii=False, indent=2)}
"""

        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", "") or str(response)

            # Try clean JSON parsing
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Extract JSON from messy text
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    raise ValueError("Could not extract valid JSON")

            # Normalize
            reranked = []
            for item in parsed:
                if isinstance(item, dict) and "text" in item and "score" in item:
                    reranked.append({
                        "text": item["text"],
                        "score": float(item["score"])
                    })

            if not reranked:
                raise ValueError("Empty reranked list")

            sorted_ranks = sorted(reranked, key=lambda x: x["score"], reverse=True)
            logger.info("✅ Successfully reranked results.")
            return sorted_ranks[:top_k]

        except Exception as e:
            logger.error(f"❌ Reranker failed: {e}")
            return [{"text": d.page_content, "score": 50} for d in docs[:top_k]]


# Quick local test
if __name__ == "__main__":
    from core.retriever_selector import RetrieverSelector
    retriever = RetrieverSelector()
    reranker = Reranker()
    query = "What are the competitors of SonarQube?"
    docs = retriever.retrieve(query)
    ranked = reranker.rerank(query, docs, 5)
    for r in ranked:
        print(f"{r['score']:.1f}: {r['text'][:120]}...\n")
