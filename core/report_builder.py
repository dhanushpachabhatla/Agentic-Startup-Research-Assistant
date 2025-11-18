"""
report_builder.py (enhanced version)
-----------------------------------
Builds a comprehensive final startup report by merging:
 - strategy_report.json (strategy_engine output)
 - agent_summaries.json (detailed agent insights)
 - raw_docs.json (retrieval content)
 
Outputs:
 - final_report.md  (Markdown narrative report)
 - final_report.json (structured full data export)
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger
from app.config import config

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# Helper file paths
DATA_DIR = Path("data")
STRATEGY_PATH = DATA_DIR / "memory_store" / "strategy_report.json"
AGENT_SUMMARIES_PATH = DATA_DIR / "memory_store" / "agent_summaries.json"
RAW_DOCS_PATH = DATA_DIR /"raw_docs" / "raw_docs.json"
FINAL_MD_PATH = DATA_DIR /"memory_store" / "final_report.md"
FINAL_JSON_PATH = DATA_DIR /"memory_store" / "final_report.json"


# Utils
def safe_load_json(path: Path) -> Any:
    if not path.exists():
        logger.warning(f"{path} not found.")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None


def _heading(text: str, level: int = 2) -> str:
    return f"{'#' * level} {text}\n\n"


def _summarize_with_llm(title: str, items: List[str], model="gemini-1.5-flash") -> str:
    """Optional: summarize bullet points into a paragraph via Gemini."""
    if not GEMINI_AVAILABLE or not items:
        return "\n".join(f"- {i}" for i in items)
    
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        if getattr(config, "GEMINI_API_KEY9", None):
             os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY9
        else:
            logger.warning("No Gemini API key. Skipping LLM summary.")
            return "\n".join(f"- {i}" for i in items)

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
    joined = "\n".join(f"- {i}" for i in items)
    prompt = f"Summarize these key points into a short, analytical paragraph titled '{title}':\n{joined}"
    try:
        resp = llm.invoke(prompt)
        return getattr(resp, "content", "") or joined
    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        return joined


# -----------------------------
# NEW: Formatting Helpers
# -----------------------------

def format_competitor_summary(summary_list):
    """Formats CompetitorScout output for Markdown."""
    parts = []
    if isinstance(summary_list, dict):
        summary_list = [summary_list]
    elif isinstance(summary_list, str):
        return f"- {summary_list}\n\n"

    for item in summary_list:
        if isinstance(item, str):
            parts.append(f"- {item}\n")
            continue
        parts.append(f"#### {item.get('name', 'Unknown Competitor')}\n")
        parts.append(f"- **Domain:** {item.get('domain', 'N/A')}\n")
        parts.append(f"- **Website:** {item.get('website', 'N/A')}\n")
        parts.append(f"- **Summary:** {item.get('summary', 'N/A')}\n")

        features = item.get('key_features', [])
        if features:
            parts.append("- **Key Features:**\n")
            for f in features:
                parts.append(f"  - {f}\n")
        parts.append("\n")
    return "".join(parts)


def format_trends_summary(summary_list):
    """Formats TrendsScraper output for Markdown."""
    parts = []
    if isinstance(summary_list, dict):
        summary_list = [summary_list]
    elif isinstance(summary_list, str):
        return f"- {summary_list}\n\n"

    for item in summary_list:
        if isinstance(item, str):
            parts.append(f"- {item}\n")
            continue
        parts.append(f"#### {item.get('trend_name', 'Unknown Trend')}\n")
        parts.append(f"**Summary:** {item.get('short_summary', 'N/A')}\n")

        sources = item.get('supporting_sources', [])
        if sources:
            parts.append("**Supporting Sources:**\n")
            for s in sources:
                parts.append(f"- {s}\n")
        parts.append("\n")
    return "".join(parts)


def format_tech_summary(summary_list):
    """Formats TechPaperMiner output for Markdown."""
    parts = []
    if isinstance(summary_list, dict):
        summary_list = [summary_list]
    elif isinstance(summary_list, str):
        return f"- {summary_list}\n\n"

    for item in summary_list:
        if isinstance(item, str):
            parts.append(f"- {item}\n")
            continue
        parts.append(f"#### {item.get('title', 'Unknown Paper')}\n")
        parts.append(f"**Authors:** {', '.join(item.get('authors', ['N/A']))}\n")
        parts.append(f"**Source:** {item.get('source_url', 'N/A')}\n")
        parts.append(f"**Summary:** {item.get('summary', 'N/A')}\n")

        findings = item.get('key_findings', [])
        if findings:
            parts.append("**Key Findings:**\n")
            for f in findings:
                parts.append(f"  - {f}\n")
        parts.append("\n")
    return "".join(parts)



# Core Builder
def build_final_report() -> Dict[str, Any]:
    logger.info("Building enhanced final report...")

    # === Load all inputs ===
    strategy = safe_load_json(STRATEGY_PATH) or {}
    agent_summaries = safe_load_json(AGENT_SUMMARIES_PATH) or []
    raw_docs = safe_load_json(RAW_DOCS_PATH) or []

    # === Group agent summaries ===
    # We are not flattening anymore, we keep the original structure
    grouped = {}
    if isinstance(agent_summaries, list):
        for s in agent_summaries:
            agent = s.get("agent", "Unknown")
            summary_data = s.get("summary", "")
            grouped.setdefault(agent, []).append(summary_data)
    else:
        logger.warning("agent_summaries.json not a list. Skipping grouping.")
        grouped = {"Misc": [str(agent_summaries)]}

    # === Compose Markdown ===
    md_parts = []
    md_parts.append(_heading("Agentic Startup Research Assistant — Final Report", 1))
    md_parts.append(f"*Generated: {datetime.now().astimezone().isoformat()}*\n\n")

    # ---- Executive Summary ----
    md_parts.append(_heading("Executive Summary", 2))
    exec_sum = strategy.get("executive_summary", "")
    if not exec_sum and grouped:
        # Create a summary from the *first* item of each agent's summary list
        summary_items = []
        for agent, summaries in grouped.items():
            if summaries:
                summary_items.append(f"From {agent}: {json.dumps(summaries[0])[:300]}...")
        exec_sum = _summarize_with_llm("Executive Summary", summary_items)
    md_parts.append(exec_sum + "\n\n")

    # ---- Key Findings ----
    md_parts.append(_heading("Key Findings", 2))
    findings = strategy.get("key_findings", [])
    if not findings:
        findings = []
        for agent, summaries in grouped.items():
            if summaries and isinstance(summaries[0], list) and summaries[0]:
                item = summaries[0][0] # Get the first item from the first summary list
                if 'trend_name' in item:
                    findings.append(f"**{item.get('trend_name')}:** {item.get('short_summary', 'N/A')}")
                elif 'name' in item:
                    findings.append(f"**{item.get('name')}:** {item.get('summary', 'N/A')}")
                elif 'title' in item:
                    findings.append(f"**{item.get('title')}:** {item.get('summary', 'N/A')}")
        
    for i, f in enumerate(findings, 1):
        md_parts.append(f"{i}. {f}\n") # Removed snippet `[:]`
    md_parts.append("\n")

    # ---- Agent-specific Insights ----
    
    md_parts.append(_heading("Detailed Agent Insights", 2))
    for agent, summaries_list in grouped.items():
        md_parts.append(_heading(agent.replace("_", " "), 3))
        
        if not summaries_list:
            md_parts.append("No data.\n\n")
            continue
            
        # summaries_list is a list of summaries (e.g., from multiple runs)
        # We'll format each summary.
        for summary_data in summaries_list:
            if not summary_data:
                continue

            if agent == "CompetitorScout":
                md_parts.append(format_competitor_summary(summary_data))
            elif agent == "TrendsScraper":
                md_parts.append(format_trends_summary(summary_data))
            elif agent == "TechPaperMiner":
                md_parts.append(format_tech_summary(summary_data))
            else:
                # Fallback for unknown agent types
                if isinstance(summary_data, (list, dict)):
                    md_parts.append(f"```json\n{json.dumps(summary_data, indent=2, ensure_ascii=False)}\n```\n\n")
                else:
                    md_parts.append(f"- {str(summary_data)}\n\n")
    

    # ---- Market Opportunities ----
    md_parts.append(_heading("Market Opportunities", 2))
    for i, op in enumerate(strategy.get("market_opportunities", []), 1):
        if isinstance(op, dict):
            md_parts.append(f"**{i}. {op.get('opportunity')}** — Impact: {op.get('impact','?')}\n")
            ev = op.get("evidence", [])
            if ev:
                md_parts.append("Evidence:\n" + "\n".join(f"- {e}" for e in ev) + "\n\n") # Removed snippet `[:]`
        else:
            md_parts.append(f"- {op}\n")
    md_parts.append("\n")

    # ---- Risks ----
    md_parts.append(_heading("Risks & Challenges", 2))
    for i, r in enumerate(strategy.get("risks_and_challenges", []), 1):
        md_parts.append(f"{i}. {r}\n")
    md_parts.append("\n")

    # ---- Recommendations ----
    md_parts.append(_heading("Strategic Recommendations", 2))
    recs = strategy.get("strategic_recommendations", [])
    if recs:
        for r in recs:
            if isinstance(r, dict):
                md_parts.append(f"- **{r.get('area')}** — {r.get('action')} (Priority: {r.get('priority')})\n")
            else:
                md_parts.append(f"- {r}\n")
    md_parts.append("\n")

    # ---- KPIs ----
    md_parts.append(_heading("Suggested KPIs", 2))
    kpis = strategy.get("suggested_kpis", [])
    if kpis:
        for k in kpis:
            if isinstance(k, dict):
                md_parts.append(f"- **{k.get('name')}** — Target: {k.get('target')}\n")
            else:
                md_parts.append(f"- {k}\n")
    md_parts.append("\n")

    # ---- Roadmap ----
    md_parts.append(_heading("Product Roadmap", 2))
    roadmap = strategy.get("roadmap", {})
    for phase in ["short_term", "mid_term", "long_term"]:
        md_parts.append(_heading(phase.replace("_", " ").title(), 3))
        for i, item in enumerate(roadmap.get(phase, []), 1):
            md_parts.append(f"{i}. {item}\n")
        md_parts.append("\n")

    # ---- Raw Document Snippets ----
    md_parts.append(_heading("Selected Raw Document Snippets", 2))
    for i, d in enumerate(raw_docs[:5], 1): # Show 5 snippets
        if isinstance(d, dict):
            src = d.get("metadata", {}).get("source", "N/A")
            content = d.get("page_content", "")
        else:
            src = getattr(d, "metadata", {}).get("source", "N/A")
            content = getattr(d, "page_content", "")
        md_parts.append(f"**Snippet {i} (Source: {src})**\n{content}\n\n")

    # ---- References ----
    refs = strategy.get("supporting_references", [])
    md_parts.append(_heading("References", 2))
    if refs:
        for r in refs:
            md_parts.append(f"- {r}\n")
    else:
        logger.info("No explicit references found in strategy, populating from raw docs...")
        doc_sources = set()
        for d in raw_docs:
            if isinstance(d, dict):
                src = d.get("metadata", {}).get("source")
            else:
                src = getattr(d, "metadata", {}).get("source")
            
            if src and src not in doc_sources:
                md_parts.append(f"- {src}\n")
                doc_sources.add(src)
        if not doc_sources:
            md_parts.append("No references found.\n")

    
    # SAVE OUTPUTS
    md_text = "".join(md_parts)
    FINAL_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MD_PATH.write_text(md_text, encoding="utf-8")
    logger.info(f"✅ Saved Markdown report to {FINAL_MD_PATH}")

    structured = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "strategy": strategy,
        "agent_groups": grouped,
        "raw_docs_count": len(raw_docs),
        "markdown_path": str(FINAL_MD_PATH)
    }

    with FINAL_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Saved structured report to {FINAL_JSON_PATH}")

    return structured



# Entry point
if __name__ == "__main__":
    result = build_final_report()
    print(json.dumps({
        "markdown": result["markdown_path"],
        "agents": list(result["agent_groups"].keys()),
        "raw_docs_count": result["raw_docs_count"]
    }, indent=2))