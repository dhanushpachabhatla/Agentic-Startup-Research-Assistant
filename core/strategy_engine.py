"""
strategy_engine.py

Produces a structured strategic analysis from:
 - agent_summaries.json (collected agent outputs)
 - Optionally augmented with RAG retrieval over raw_docs (via RetrieverSelector + Summarizer)

Outputs:
 - strategy_report (dict) with keys:
    - executive_summary
    - key_findings
    - market_opportunities
    - risks_and_challenges
    - strategic_recommendations
    - suggested_kpis
    - roadmap (short-term, mid-term, long-term)
 - saved to strategy_report.json
"""

import os
import json
from typing import List, Dict, Any
from loguru import logger
from pathlib import Path

from app.config import config
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional enrichers
try:
    from core.retriever_selector import RetrieverSelector
    from core.summarizer import Summarizer
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False
    RetrieverSelector = None
    Summarizer = None

logger.info("Initializing Strategy Engine...")

# pick a Gemini key available
GEMINI_KEY = getattr(config, "GEMINI_API_KEY10", None)
if GEMINI_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

# Initialize LLM
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def load_json_file(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        logger.warning(f"{path} not found.")
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None


def _safe_json_parse(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    # Trim code fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    # Extract first JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # As fallback try full text parse
    try:
        return json.loads(text)
    except Exception:
        return None


def _compose_context(agent_summaries: List[Dict[str, Any]], top_rag_contexts: List[str]) -> str:
    parts = []
    parts.append("AGENT SUMMARIES (brief):")
    for s in agent_summaries:
        # Each agent summary is expected to be dict with at least agent and summary/text
        agent_name = s.get("agent", "<agent>")
        tid = s.get("task_id", "")
        summ = s.get("summary") or s.get("text") or s.get("output") or ""
        # limit
        parts.append(f"- [{agent_name} | task {tid}]: {str(summ)[:800]}")
    if top_rag_contexts:
        parts.append("\nTOP RAG CONTEXTS (snippets):")
        for i, c in enumerate(top_rag_contexts, 1):
            parts.append(f"Snippet {i}:\n{c[:1000]}")
    return "\n\n".join(parts)


def generate_strategy(agent_summaries_path: str = "data/memory_store/agent_summaries.json",
                      raw_docs_query: str = None,
                      use_rag: bool = True) -> Dict[str, Any]:
    """
    Main function to generate strategy report.

    - agent_summaries_path: where orchestrator saved agent summaries (list)
    - raw_docs_query: optional query string to retrieve supporting context
    - use_rag: whether to call retriever + summarizer to enrich the prompt
    """
    agent_summaries = load_json_file(agent_summaries_path) or []
    if not isinstance(agent_summaries, list):
        logger.warning("agent_summaries.json is not a list — coercing.")
        # attempt to coerce dict->list
        if isinstance(agent_summaries, dict):
            agent_summaries = [agent_summaries]
        else:
            agent_summaries = []

    top_rag_contexts = []
    if use_rag and raw_docs_query and RAG_AVAILABLE:
        try:
            logger.info("Running hybrid retrieval to get supporting contexts...")
            retriever = RetrieverSelector()
            docs = retriever.retrieve(raw_docs_query)
            # convert to plain strings; limit to top K
            snippets = [d.page_content[:1500] for d in docs[:6]]
            # Summarize these via summarizer for compact context
            summarizer = Summarizer()
            rag_summary = summarizer.summarize(raw_docs_query, snippets)
            top_rag_contexts = snippets
            logger.info("Hybrid retrieval + summary complete.")
            # also include rag_summary as a supporting snippet
            top_rag_contexts.insert(0, f"RAG summary:\n{rag_summary}")
        except Exception as e:
            logger.warning(f"RAG enrichment failed: {e}")
            top_rag_contexts = []

    # Compose LLM prompt
    context = _compose_context(agent_summaries, top_rag_contexts)
    prompt = f"""
You are a strategic product/market analyst for early-stage startups.
Given the aggregated research below (agent summaries + supporting document snippets), produce a structured JSON strategy report.

CONTEXT:
{context}

Return JSON with EXACT keys:
- executive_summary: short 3-4 sentence overview
- key_findings: list of bullet-finding strings (5 max)
- market_opportunities: list of opportunity objects {{ "opportunity": "...", "impact": "High|Medium|Low", "evidence": ["..."] }}
- risks_and_challenges: list of risk strings (5 max)
- strategic_recommendations: list of recommendation objects {{ "area": "...", "action": "...", "priority": "High|Medium|Low", "owner": "Product/Engine/BD" }}
- suggested_kpis: list of KPI objects {{ "name": "...", "target": "...", "rationale": "..." }}
- roadmap: object with keys short_term (0-3m), mid_term (3-9m), long_term (9-24m) each a list of items
- supporting_references: list of source strings (URLs or agent names)

Be concise. Return ONLY valid JSON (no markdown, no extra commentary).
"""

    logger.info("Invoking LLM to synthesize strategy...")
    resp = LLM.invoke(prompt)
    text = getattr(resp, "content", None) or str(resp)
    parsed = _safe_json_parse(text)
    if parsed:
        logger.info("✅ Strategy JSON parsed successfully.")
        strategy = parsed
    else:
        logger.warning("LLM did not return clean JSON — attempting a best-effort parse into fallback structure.")
        # fallback: wrap raw text into executive summary + raw_notes
        strategy = {
            "executive_summary": text[:800],
            "key_findings": [],
            "market_opportunities": [],
            "risks_and_challenges": [],
            "strategic_recommendations": [],
            "suggested_kpis": [],
            "roadmap": {"short_term": [], "mid_term": [], "long_term": []},
            "supporting_references": []
        }

    # Save strategy artifact
    out_path = Path("data/memory_store").joinpath("strategy_report.json")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved strategy report to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save strategy report: {e}")

    return strategy


if __name__ == "__main__":
    import json

    logger.info("🚀 Running Strategy Engine (with RAG enrichment)...")

    # === Default query for RAG ===
    DEFAULT_QUERY = (
        "What are the latest market trends, competitor offerings, and emerging opportunities "
        "for AI-powered GitHub repository analysis tools and developer productivity platforms "
        "that use LLMs for code quality, insights, and vulnerability detection?"
    )

    # Paths (orchestrator outputs)
    AGENT_SUMMARIES_PATH = "data/memory_storeagent_summaries.json"

    # Run synthesis
    strategy = generate_strategy(
        agent_summaries_path=AGENT_SUMMARIES_PATH,
        raw_docs_query=DEFAULT_QUERY,
        use_rag=True,
    )

    # Print short summary for console
    print("\n================== STRATEGY SNAPSHOT ==================\n")
    print(f"Executive Summary:\n{strategy.get('executive_summary', '')}\n")

    key_findings = strategy.get("key_findings", [])
    if key_findings:
        print("Key Findings:")
        for k in key_findings[:5]:
            print(f" - {k}")
        print()

    print("✅ Strategy JSON saved to: data/memory_store/strategy_report.json")
    print("========================================================\n")

