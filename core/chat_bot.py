"""
chat_bot.py (persistent + context-aware)
----------------------------------------
Conversational RAG chatbot with:
 - Persistent user memory (saved to disk)
 - Context-aware reasoning
 - Hybrid context (RAG + conversation)
"""

import json
import argparse
from loguru import logger
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from core.retriever_selector import RetrieverSelector
from core.summarizer import Summarizer
from core.reranker import Reranker


# Persistent Chat Memory
class ChatMemory:
    def __init__(self, user_id="user_1"):
        self.user_id = user_id
        self.file_path = Path(f"data/memory_store/{user_id}_chat_memory.json")
        self.history = self._load()

    def _load(self):
        if not self.file_path.exists():
            logger.info(f"💾 Memory file not found for {self.user_id}, creating new one.")
            return []
        try:
            content = self.file_path.read_text(encoding="utf-8").strip()
            if not content:
                logger.warning("Memory file is empty — starting fresh.")
                return []
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to load chat memory: {e}")
            return []

    def _save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save chat memory: {e}")

    def add(self, user_query: str, bot_answer: str):
        self.history.append({
            "user": user_query,
            "bot": bot_answer,
            "timestamp": datetime.now().isoformat()
        })
        self._save()

    def get_context(self) -> str:
        recent = self.history[-3:]
        return "\n".join(
            [f"User: {h['user']}\nBot: {h['bot']}" for h in recent]
        ).strip()

    def print_history(self):
        print("\n--- Conversation History ---")
        for h in self.history:
            print(f"🧍 {h['user']}\n🤖 {h['bot']}\n")


# Core Query Answering
def answer_query(user_id: str, query: str, memory: ChatMemory,
                 use_rag=True, rag_k=6) -> Dict[str, any]:
    retriever = RetrieverSelector()
    summarizer = Summarizer()
    reranker = Reranker()

    # Combine user query with conversational context
    memory_context = memory.get_context()
    combined_query = (
        f"User question: {query}\n\nConversation context:\n{memory_context}"
        if memory_context else query
    )

    logger.info(f"User '{user_id}' asked: {query}")

    # RAG retrieval
    if use_rag:
        docs = retriever.retrieve(combined_query)
        if not docs:
            logger.warning("No docs retrieved, falling back to summarizer only.")
            summary = summarizer.summarize(query, [])
            memory.add(query, summary)
            return {"answer": summary, "retrieved_docs": 0}

        # Re-rank docs by relevance
        ranked = reranker.rerank(combined_query, docs, top_k=rag_k)
        contexts = [r["text"] for r in ranked]
    else:
        contexts = []

    # Merge retrieved context and memory for summarization
    augmented_context = (
        "\n\n--- Memory Context ---\n" + memory_context if memory_context else ""
    )
    query_with_context = f"{query}\n\nUse the following context:\n{augmented_context}"

    summary = summarizer.summarize(query_with_context, contexts)

    # Store turn in memory
    memory.add(query, summary)
    return {"answer": summary, "retrieved_docs": len(contexts)}


# Command-line Chat Interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="user_1", help="User ID for persistent chat")
    parser.add_argument("--query", help="Single query (optional)")
    parser.add_argument("--no-rag", dest="use_rag", action="store_false")
    args = parser.parse_args()

    memory = ChatMemory(args.user)

    if args.query:
        # Single-shot mode
        result = answer_query(args.user, args.query, memory, use_rag=args.use_rag)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Interactive CLI loop
        print("🤖 Chatbot is ready! Type your questions (or 'exit' to quit).")
        while True:
            user_input = input("\n🧍 You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Exiting chatbot. Goodbye!")
                break

            result = answer_query(args.user, user_input, memory, use_rag=True)
            print(f"\n🤖 Bot: {result['answer']}")
