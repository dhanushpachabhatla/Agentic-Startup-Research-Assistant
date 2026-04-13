# Agentic Startup Research Assistant

An autonomous research assistant that turns a startup idea or research question into a structured market, technical, trend, and strategy report.

The project combines Gemini-powered planning and synthesis, specialized research agents, LangGraph orchestration, Chroma-based retrieval, and a Streamlit interface for running the full workflow and chatting with the generated knowledge base.

## What It Does

Given a startup idea, the system can:

- Parse the idea into structured intent such as industry, audience, business model, problem, solution, data needs, and agent triggers.
- Build a dynamic research plan with tasks assigned to specialist agents.
- Run multiple research agents for competitor, paper, and trend discovery.
- Store raw source material and agent summaries as JSON artifacts.
- Build a local Chroma vector store from collected documents.
- Generate a strategy report with findings, opportunities, risks, recommendations, KPIs, and roadmap items.
- Build a final Markdown and JSON report.
- Index strategic outputs back into the vector store.
- Answer follow-up questions through a persistent RAG chatbot.

## Architecture

```text
User idea
  -> IntentParser
  -> DynamicTaskPlanner
  -> LangGraph orchestrator
       -> CompetitorScout
       -> TechPaperMiner
       -> TrendsScraper
  -> raw_docs.json + agent_summaries.json
  -> Chroma RAG index
  -> StrategyEngine
  -> ReportBuilder
  -> StrategicKnowledgeIndexer
  -> Chatbot Assistant
```

## Main Components

### Agents

The `agents/` directory contains the specialist collectors used by the orchestrator.

| File | Agent | Purpose |
| --- | --- | --- |
| `agents/competitor_scout.py` | `CompetitorScoutAgent` | Finds competitor companies through Tavily, scrapes competitor websites, and returns structured competitor summaries plus raw documents for RAG. |
| `agents/tech_paper_miner_3.py` | `TechPaperMinerAgent` | Searches Semantic Scholar, Tavily, and scraped web pages for technical papers, research articles, and libraries. |
| `agents/trend_scraper.py` | `TrendsScraperAgent` | Uses NewsAPI, Reddit public JSON endpoints, and Tavily to discover market, community, and technology trends. |

Each agent returns:

- `success`
- `output_summary`
- `output_raw_docs`
- `output_type`
- `meta`

### Core Pipeline

The `core/` directory contains orchestration, RAG, synthesis, reporting, and chat logic.

| File | Purpose |
| --- | --- |
| `core/pipeline.py` | End-to-end pipeline wrapper for intent parsing, planning, orchestration, RAG indexing, strategy generation, report building, strategic indexing, and optional chatbot loop. |
| `core/intent_parser.py` | Extracts structured startup intent using Gemini, with a rule-based fallback. |
| `core/dynamic_task_planner.py` | Generates a JSON task plan and assigns tasks to available agents. |
| `core/orchestrator.py` | Defines the LangGraph state machine and routes runnable tasks to the correct agent. |
| `core/rag_manager.py` | Chunks documents, embeds them with Gemini embeddings, and stores them in Chroma. |
| `core/retriever_selector.py` | Combines dense Chroma retrieval with optional BM25 sparse retrieval. |
| `core/reranker.py` | Re-ranks retrieved chunks with Gemini. |
| `core/summarizer.py` | Summarizes retrieved context for answers and strategy synthesis. |
| `core/strategy_engine.py` | Creates a structured strategy report from agent summaries and optional RAG context. |
| `core/report_builder.py` | Builds final Markdown and JSON reports from strategy, agent summaries, and raw docs. |
| `core/index_strategic_knowledge.py` | Adds final strategic artifacts back into Chroma without clearing existing indexed data. |
| `core/chat_bot.py` | Provides a persistent RAG chatbot with user-specific memory files. |
| `core/self_evaluate_output.py` | Checks expected task outputs and can re-run incomplete tasks. |

### App And Frontend

| File | Purpose |
| --- | --- |
| `app/config.py` | Loads `.env`, API keys, model defaults, data paths, and logging. |
| `frontend/app_frontend.py` | Streamlit app with tabs for full pipeline execution, chatbot, and report downloads. |
| `app/main.py` | Currently empty; the Streamlit app and `core.pipeline` are the active entry points. |

## Project Layout

```text
.
|-- agents/                  # Specialist research agents
|-- app/                     # Configuration and app-level entry points
|-- core/                    # Pipeline, orchestration, RAG, strategy, report, chat
|-- data/
|   |-- memory_store/        # Plans, summaries, reports, chat memory
|   `-- raw_docs/            # Raw retrieved documents
|-- frontend/                # Streamlit UI
|-- testing/                 # Local API and integration experiments
|-- chroma_db/               # Local Chroma vector store
|-- logs/                    # Runtime logs
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.10 or newer
- A Google Gemini API key
- Tavily API key for competitor and web search collection
- NewsAPI key for trend collection
- Internet access for live search, scraping, Semantic Scholar, Reddit, and NewsAPI calls

Optional:

- Cohere, Grok, SerpAPI, and LangSmith keys if you are experimenting with the testing utilities or tracing.

## Installation

From the repository root:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the repository root. The implementation reads many named Gemini keys from `app/config.py`; the current pipeline mainly uses keys `3`, `4`, `9`, `10`, `20`, `21`, and `22`.

You can reuse the same Gemini key for development:

```env
GEMINI_API_KEY3=your_gemini_key
GEMINI_API_KEY4=your_gemini_key
GEMINI_API_KEY9=your_gemini_key
GEMINI_API_KEY10=your_gemini_key
GEMINI_API_KEY20=your_gemini_key
GEMINI_API_KEY21=your_gemini_key
GEMINI_API_KEY22=your_gemini_key

# Used by some LangChain / Google SDK paths.
GOOGLE_API_KEY=your_gemini_key
GOOGLE_API_KEY_4=your_gemini_key

TAVILY_API_KEY=your_tavily_key
NEWS_API_KEY=your_newsapi_key

# Optional
COHERE_API_KEY=
GROK_API_KEY=
SERPAPI_KEY=
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=agentic-startup-assistant
```

## Running The Project

### Streamlit UI

The easiest way to use your own startup idea is the Streamlit frontend:

```bash
streamlit run frontend/app_frontend.py
```

The UI includes:

- Full Pipeline: run intent parsing, planning, agents, RAG indexing, strategy generation, report building, and strategic indexing.
- Chatbot Assistant: ask follow-up questions using the indexed research and persistent chat memory.
- Reports & Outputs: preview and download generated artifacts.

### CLI Pipeline

You can run the end-to-end pipeline from the command line:

```bash
python -m core.pipeline
```

Note: `core/pipeline.py` currently uses a sample `user_query` inside the `__main__` block and then launches an interactive chatbot. Use the Streamlit app for ad hoc queries, or edit `user_query` in `core/pipeline.py` for a CLI run.

### Chatbot Only

After building the Chroma index, you can query the chatbot directly:

```bash
python -m core.chat_bot --user user_1 --query "What are the main competitor gaps?"
```

For an interactive CLI chat:

```bash
python -m core.chat_bot --user user_1
```

### Individual Modules

Useful module-level commands:

```bash
python -m core.intent_parser
python -m core.dynamic_task_planner
python -m core.orchestrator
python -m core.rag_manager
python -m core.report_builder
python -m core.index_strategic_knowledge
```

Agent-level local tests:

```bash
python agents/competitor_scout.py
python agents/tech_paper_miner_3.py
python agents/trend_scraper.py
```

## Generated Outputs

The pipeline writes its main artifacts under `data/`:

| Path | Description |
| --- | --- |
| `data/memory_store/plan.json` | Dynamic task plan generated from parsed intent. |
| `data/memory_store/agent_summaries.json` | Structured summaries returned by research agents. |
| `data/raw_docs/raw_docs.json` | Raw documents collected by agents and serialized for RAG indexing. |
| `data/memory_store/strategy_report.json` | Strategy engine output with findings, opportunities, risks, KPIs, and roadmap. |
| `data/memory_store/final_report.md` | Human-readable final report. |
| `data/memory_store/final_report.json` | Structured final report export. |
| `data/memory_store/*_chat_memory.json` | Persistent chatbot memory by user/session. |
| `chroma_db/` | Local Chroma vector database. |
| `logs/system.log` | Runtime logs from the configuration logger. |



## Development Notes

- Run commands from the repository root so imports like `from app.config import config` resolve correctly.
- The first full run may take time because agents call live APIs, scrape pages, embed chunks, and write Chroma data.
- `run_rag_indexer()` clears `chroma_db/` before indexing raw documents. The strategic indexer later appends report artifacts without clearing existing data.
- If an agent is missing its Gemini key, several collectors fall back to deterministic sample output, but later synthesis components generally expect a working Gemini configuration.
- The README intentionally does not document values from the local `.env`; keep API keys out of version control.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| Import errors | Run from the repo root and activate the virtual environment. 
| Slow execution | Live web collection, scraping, embedding, and LLM synthesis are network-bound and rate-limit sensitive. |

