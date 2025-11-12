# frontend/app_frontend.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from loguru import logger
import json
import traceback

# === Import backend modules ===
from core.pipeline import (
    run_intent_parser,
    run_task_planner,
    run_orchestrator,
    run_rag_indexer,
    run_strategy_engine,
    run_report_builder,
    run_strategic_indexer,
)
from core.chat_bot import ChatMemory, answer_query


# ============================================================
# 🎨 Streamlit Configuration
# ============================================================
st.set_page_config(
    page_title="Agentic Startup Research Assistant",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Agentic Startup Research Assistant")
st.caption("A full AI-powered research pipeline with agents, RAG, and strategy synthesis.")


# ============================================================
# 🧩 TAB NAVIGATION
# ============================================================
tabs = st.tabs(["🏁 Full Pipeline", "💬 Chatbot Assistant", "📊 Reports & Outputs"])


# ============================================================
# TAB 1 — RUN FULL PIPELINE
# ============================================================
with tabs[0]:
    st.header("🏁 Run the End-to-End Research Pipeline")

    user_query = st.text_area(
        "💡 Enter your startup idea or research problem:",
        placeholder="e.g., AI-powered GitHub repository analysis tool for developers...",
        height=120
    )

    if st.button("🚀 Run Full Pipeline"):
        if not user_query.strip():
            st.warning("Please enter a valid query first.")
        else:
            try:
                st.subheader("🧠 Query Summary")
                st.write(f"**User Query:** {user_query}")

                st.info("1️⃣ Running Intent Parser...")
                intent_parser = run_intent_parser(user_query)
                st.json(intent_parser)

                st.info("2️⃣ Running Task Planner...")
                task_plan = run_task_planner(intent_parser)
                st.json(task_plan)

                st.info("3️⃣ Running Multi-Agent Orchestrator...")
                orch_output = run_orchestrator(task_plan)
                st.success(f"✅ Orchestrator completed. Found {len(orch_output['raw_docs'])} raw docs.")
                st.download_button(
                    "⬇️ Download Agent Summaries",
                    json.dumps(orch_output['summaries'], indent=2),
                    "agent_summaries.json"
                )

                st.info("4️⃣ Building RAG Index...")
                run_rag_indexer()
                st.success("✅ RAG index built successfully.")

                st.info("5️⃣ Running Strategy Engine...")
                strategy = run_strategy_engine()
                st.json(strategy)

                st.info("6️⃣ Building Final Report...")
                report_result = run_report_builder()
                st.success("✅ Final report created.")
                st.markdown(f"**Markdown Path:** `{report_result['markdown_path']}`")

                st.info("7️⃣ Indexing Strategic Knowledge...")
                run_strategic_indexer()
                st.success("✅ Strategic knowledge added to vector DB.")

                st.success("🎉 Full pipeline completed successfully!")

            except Exception as e:
                st.error(f"❌ Pipeline crashed: {e}")
                st.code(traceback.format_exc())



# ============================================================
# TAB 2 — INTERACTIVE CHATBOT (GPT-style)
# ============================================================
with tabs[1]:
    st.header("💬 Chat with the AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemory("streamlit_user")

    # --- Chat container (like GPT) ---
    chat_container = st.container()

    for chat in st.session_state.chat_history:
        with chat_container:
            st.markdown(f"🧑 **You:** {chat['user']}")
            st.markdown(f"🤖 **Assistant:** {chat['bot']}")

    # --- Input box with send arrow ---
    user_input = st.chat_input("Type your question here...")

    if user_input:
        try:
            result = answer_query("streamlit_user", user_input, st.session_state.memory)
            bot_response = result["answer"]

            # Add to history
            st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

            # Immediately display new message
            with chat_container:
                st.markdown(f"🧑 **You:** {user_input}")
                st.markdown(f"🤖 **Assistant:** {bot_response}")

        except Exception as e:
            st.error(f"Chatbot failed: {e}")
            st.code(traceback.format_exc())

    if st.button("🧠 Show Chat Memory"):
        st.write(st.session_state.memory.history)



# ============================================================
# TAB 3 — REPORTS & OUTPUTS (Markdown preview)
# ============================================================
with tabs[2]:
    st.header("📊 Reports and Data Outputs")

    data_dir = Path("data/memory_store")
    raw_dir = Path("data/raw_docs")

    report_md_path = data_dir / "final_report.md"

    # --- Markdown preview ---
    if report_md_path.exists():
        st.subheader("📝 Final Report Preview")
        with open(report_md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        st.markdown(md_content)
    else:
        st.warning("No final report found yet. Run the pipeline first!")

    # --- Download buttons ---
    st.divider()
    st.subheader("⬇️ Download Files")

    files = {
        "Agent Summaries": data_dir / "agent_summaries.json",
        "Strategy Report": data_dir / "strategy_report.json",
        "Final Report (Markdown)": report_md_path,
        "Final Report (JSON)": data_dir / "final_report.json",
        "Raw Docs": raw_dir / "raw_docs.json"
    }

    for name, path in files.items():
        if path.exists():
            st.download_button(
                f"⬇️ Download {name}",
                path.read_bytes(),
                file_name=path.name,
                mime="application/json" if path.suffix == ".json" else "text/plain"
            )
        else:
            st.warning(f"{name} not found yet.")
