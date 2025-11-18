"""
pipeline.py
------------
Unified pipeline for the Agentic Startup Research Assistant:
1. Intent Parsing
2. Dynamic Task Planning
3. Orchestration & Multi-Agent Research
4. RAG Index Building (raw docs)
5. Strategy Generation
6. Final Report Assembly
7. Strategic Knowledge Indexing
8. Interactive Chatbot (optional, at end)
"""

import json
from loguru import logger
from pathlib import Path
import traceback

# === Import each stage ===
from core.intent_parser import IntentParser
from core.dynamic_task_planner import DynamicTaskPlanner
from core.orchestrator import app  # Graph-based orchestrator already compiled
from core.rag_manager import VectorStoreManager
from langchain_core.documents import Document
from core.strategy_engine import generate_strategy
from core.report_builder import build_final_report
from core.index_strategic_knowledge import load_texts, _normalize_to_text
from core.chat_bot import ChatMemory, answer_query


def run_intent_parser(user_query: str):
    try:
        logger.info("🧩 [1] Running Intent Parser...")
        parser = IntentParser()
        intent = parser.parse(user_query)
        logger.success("✅ Intent Parser completed.")
        return intent
    except Exception as e:
        logger.error(f"❌ Intent Parser failed: {e}")
        traceback.print_exc()
        return None




def run_task_planner(intent):
    try:
        logger.info("🧠 [2] Running Dynamic Task Planner...")
        planner = DynamicTaskPlanner(use_llm=True)
        result = planner.plan({"intent": intent})
        task_plan = result.get("task_plan")

        # ✅ Save plan output for downstream modules
        import os, json
        os.makedirs("data/memory_store", exist_ok=True)
        plan_path = "data/memory_store/plan.json"

        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(task_plan, f, indent=2, ensure_ascii=False)
        logger.success(f"✅ Task Planner completed and plan saved to {plan_path}")

        return task_plan
    except Exception as e:
        logger.error(f"❌ Dynamic Task Planner failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_orchestrator(task_plan):
    try:
        logger.info("🤖 [3] Running Multi-Agent Orchestrator...")
        inputs = {
            "plan": task_plan,
            "completed_tasks": set(),
            "raw_documents": [],
            "agent_summaries": [],
            "final_report": ""
        }
        final_state = app.invoke(inputs, {"recursion_limit": 15})

        if not final_state:
            raise RuntimeError("No final state returned by orchestrator.")

        # Save agent summaries
        agent_summaries = final_state.get("agent_summaries", [])
        Path("data/memory_store").mkdir(parents=True, exist_ok=True)
        with open("data/memory_store/agent_summaries.json", "w", encoding="utf-8") as f:
            json.dump(agent_summaries, f, indent=2, ensure_ascii=False)

        # Save raw documents
        raw_docs = []
        for d in final_state.get("raw_documents", []):
            raw_docs.append({
                "page_content": getattr(d, "page_content", ""),
                "metadata": getattr(d, "metadata", {})
            })
        Path("data/raw_docs").mkdir(parents=True, exist_ok=True)
        with open("data/raw_docs/raw_docs.json", "w", encoding="utf-8") as f:
            json.dump(raw_docs, f, indent=2, ensure_ascii=False)

        logger.success(f"✅ Orchestrator finished — {len(agent_summaries)} summaries, {len(raw_docs)} docs.")
        return {"summaries": agent_summaries, "raw_docs": raw_docs}

    except Exception as e:
        logger.error(f"❌ Orchestrator failed: {e}")
        traceback.print_exc()
        return None


def run_rag_indexer():
    try:
        logger.info("📚 [4] Building initial RAG index from raw documents...")
        json_input_file = "data/raw_docs/raw_docs.json"
        with open(json_input_file, "r", encoding="utf-8") as f:
            docs_json = json.load(f)
        documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_json]

        manager = VectorStoreManager()
        manager.clear_store()
        manager.add_documents(documents)

        logger.success("✅ RAG vector store built from raw_docs.")
        return True
    except Exception as e:
        logger.error(f"❌ RAG Manager failed: {e}")
        traceback.print_exc()
        return False


def run_strategy_engine():
    try:
        logger.info("🎯 [5] Running Strategy Engine (RAG synthesis)...")
        strategy = generate_strategy(
            agent_summaries_path="data/memory_store/agent_summaries.json",
            raw_docs_query="AI-powered GitHub repository analysis tools and developer productivity trends",
            use_rag=True,
        )
        logger.success("✅ Strategy Engine completed.")
        return strategy
    except Exception as e:
        logger.error(f"❌ Strategy Engine failed: {e}")
        traceback.print_exc()
        return None


def run_report_builder():
    try:
        logger.info("📝 [6] Running Report Builder...")
        result = build_final_report()
        logger.success("✅ Final Report Builder completed.")
        return result
    except Exception as e:
        logger.error(f"❌ Report Builder failed: {e}")
        traceback.print_exc()
        return None


def run_strategic_indexer():
    try:
        logger.info("📘 [7] Indexing strategic documents (without overwriting existing data)...")
        from core.index_strategic_knowledge import FILES_TO_INDEX
        from core.rag_manager import VectorStoreManager

        docs = load_texts()
        if not docs:
            logger.warning("No new docs found for strategic index.")
            return False

        manager = VectorStoreManager()
        db = manager._get_db()

        existing_sources = set()
        try:
            collection = db.get(include=["metadatas"])
            if "metadatas" in collection:
                for meta in collection["metadatas"]:
                    if isinstance(meta, dict) and meta.get("source"):
                        existing_sources.add(meta["source"])
        except Exception as e:
            logger.warning(f"Could not fetch existing metadatas: {e}")

        new_docs = [d for d in docs if d.metadata.get("source") not in existing_sources]
        if not new_docs:
            logger.info("No new strategy docs — index already up to date.")
        else:
            manager.add_documents(new_docs)
            logger.success(f"✅ Added {len(new_docs)} new strategy docs.")
        return True
    except Exception as e:
        logger.error(f"❌ Strategic Indexer failed: {e}")
        traceback.print_exc()
        return False


def run_chatbot():
    try:
        logger.info("💬 [8] Launching Chatbot (context-aware RAG)...")
        memory = ChatMemory("user_1")
        print("\n🤖 Chatbot ready! Type 'exit' to stop.\n")
        while True:
            q = input("🧍 You: ").strip()
            if q.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!\n")
                break
            response = answer_query("user_1", q, memory)
            print(f"\n🤖 Bot: {response['answer']}\n")
    except Exception as e:
        logger.error(f"❌ Chatbot failed: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    memory_file = Path("data/memory_store/user_1_chat_memory.json")
    if memory_file.exists():
        try:
            memory_file.unlink()  # deletes the file
            print(f"🗑️ Old chat memory deleted: {memory_file}")
        except Exception as e:
            print(f"❌ Could not delete chat memory: {e}")
    
    logger.info("🚀 Starting Full Agentic Research Pipeline...\n")
    user_query = ( "Create AI-driven fitness apps which tracks user health and his dialy activities " "and give healthy insights to maintain a good lifestyle. " "both physical and mental health." )
    if not user_query:
        logger.error("No query provided. Exiting.")
        exit()

    # Intent Parsing
    intent = run_intent_parser(user_query)
    if not intent: exit()

    # Task Planning
    plan = run_task_planner(intent)
    if not plan: exit()

    # Multi-Agent Orchestrator
    orch = run_orchestrator(plan)
    if not orch: exit()

    # RAG Indexer
    if not run_rag_indexer(): exit()

    # Strategy Engine
    strategy = run_strategy_engine()
    if not strategy: exit()

    # Report Builder
    report = run_report_builder()
    if not report: exit()

    # Index Strategic Knowledge
    run_strategic_indexer()

    # Chatbot (manual interaction)
    run_chatbot()
