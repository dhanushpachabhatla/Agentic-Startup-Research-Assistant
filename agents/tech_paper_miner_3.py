"""
tech_paper_miner.py
-------------------------------------
(Hybrid "Smart Collector") Agent that finds and returns technical papers.
Returns BOTH a final JSON summary AND the raw documents for RAG.

- Fetches data from:
    • Semantic Scholar API (for academic papers)
    • Tavily Search (for blogs, news, and other papers)
    • Web Scraper (to get text from non-academic links)
- Uses Gemini LLM to plan collection AND summarize.
"""

import os
import json
import requests
from typing import Dict, Any, List
from loguru import logger
from datetime import datetime

from app.config import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.documents import Document
from bs4 import BeautifulSoup


# =====================================================
# 📘 Pydantic Output Models
# =====================================================
class PaperItem(BaseModel):
    """Details of a single research paper."""
    title: str = Field(..., description="The full title of the paper.")
    authors: List[str] = Field(..., description="A list of the primary authors' names.")
    summary: str = Field(..., description="The paper's abstract or a concise summary.")
    source_url: str = Field(..., description="The URL to the paper's abstract page or PDF.")
    key_findings: List[str] = Field(..., description="A 2-3 bullet point list of the paper's key findings.")


class PaperList(BaseModel):
    """A list of relevant technical papers. This is the REQUIRED format for the final summary."""
    papers: List[PaperItem]


# =====================================================
# 🔍 Utility Tools
# =====================================================
def semantic_scholar_search(query: str, limit: int = 5) -> str:
    """Search Semantic Scholar for academic papers."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,authors,abstract,url,year"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json().get("data", [])
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Semantic Scholar search failed: {e}"


def tavily_search(query: str, num_results: int = 5) -> str:
    """Search the web for research blogs, news, and non-academic papers."""
    if not getattr(config, "TAVILY_API_KEY", None):
        return "Tavily API key not configured."
    try:
        url = "https://api.tavily.com/search"
        payload = {"api_key": config.TAVILY_API_KEY, "query": query, "num_results": num_results}
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return json.dumps(data.get("results", []), indent=2)
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return f"Tavily search failed: {e}"


def scrape_website(url: str) -> str:
    """Scraper that extracts clean text from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        body_text = soup.body.get_text(separator=" ", strip=True)
        clean_text = body_text[:8000]
        return clean_text
    except Exception as e:
        return f"Failed to scrape {url}: {e}"


# =====================================================
# 🤖 Tech Paper Miner Agent
# =====================================================
class TechPaperMinerAgent:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        self.llm_with_tools = None

        # ---- Define Tools ----
        semantic_tool = StructuredTool.from_function(
            func=semantic_scholar_search,
            name="semantic_scholar_search",
            description="Search Semantic Scholar for academic and technical papers."
        )
        tavily_tool = StructuredTool.from_function(
            func=tavily_search,
            name="tavily_search",
            description="Search the web for research blogs, news, and non-academic papers."
        )
        scrape_tool = StructuredTool.from_function(
            func=scrape_website,
            name="scrape_website",
            description="Scrapes the clean text content from a URL (e.g., a blog or news article)."
        )

        # ---- Tool Maps ----
        self.tools_map = {
            semantic_tool.name: semantic_tool,
            tavily_tool.name: tavily_tool,
            scrape_tool.name: scrape_tool
        }
        all_tools_for_llm = [semantic_tool, tavily_tool, scrape_tool, PaperList]

        # ---- Initialize Gemini ----
        if self.use_llm and getattr(config, "GEMINI_API_KEY21", None):
            os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY21
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            logger.info("✅ Gemini LLM initialized for TechPaperMiner.")
            self.llm_with_tools = self.llm.bind_tools(all_tools_for_llm)
        else:
            logger.warning("❌ Gemini LLM not available; fallback mode will be used.")


    # =====================================================
    # 🧩 Parse Tool Results into Documents
    # =====================================================
    def _parse_results_to_documents(self, tool_name: str, tool_args: dict, tool_result_string: str) -> List[Document]:
        documents = []
        try:
            if tool_name == "semantic_scholar_search":
                papers = json.loads(tool_result_string)
                for paper in papers:
                    title = paper.get("title", "")
                    authors = [a.get("name", "") for a in paper.get("authors", [])]
                    summary = paper.get("abstract", "")
                    url = paper.get("url", "")
                    content = f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {summary}"
                    metadata = {
                        "source": url,
                        "title": title,
                        "authors": authors,
                        "year": paper.get("year", ""),
                        "data_source": "SemanticScholar",
                        "query": tool_args.get("query", "")
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            elif tool_name == "tavily_search":
                results = json.loads(tool_result_string)
                for res in results:
                    content = res.get('content', '')
                    metadata = {
                        "source": res.get('url', ''),
                        "title": res.get('title', ''),
                        "data_source": "Tavily",
                        "query": tool_args.get("query", "")
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            elif tool_name == "scrape_website":
                url = tool_args.get("url", "")
                if not tool_result_string.startswith("Failed to scrape"):
                    content = tool_result_string
                    metadata = {
                        "source": url,
                        "title": f"Scraped content from {url}",
                        "data_source": "WebScraper"
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

        except Exception as e:
            logger.warning(f"Failed to parse tool result for {tool_name}: {e}")

        return documents


    # =====================================================
    # 🚀 Run Paper Mining Agent
    # =====================================================
    def run(self, task: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        research_task_description = (
            task.get("description")
            or task.get("topic")
            or state.get("intent", {}).get("idea", "latest AI research")
        )
        logger.info(f"🔬 Mining for papers and research on: {research_task_description}")

        try:
            if not self.llm_with_tools:
                logger.warning("Using fallback paper data (LLM unavailable).")
                summary, docs = self._fallback_papers(research_task_description)
                return {
                    "success": True,
                    "output_summary": summary,
                    "output_raw_docs": docs,
                    "output_type": "PaperReport",
                    "meta": {"mode": "Fallback"},
                }

            query = f"""
You are an AI research assistant. Your goal is to find key technical papers, articles, and libraries related to:
"{research_task_description}"

Follow this plan:
PHASE 1: BROAD DISCOVERY
→ Use tavily_search to find high-level concepts, blogs, or libraries.
PHASE 2: ACADEMIC SEARCH
→ Use semantic_scholar_search to find relevant academic papers.
PHASE 3: SCRAPE DETAILS
→ Use scrape_website on promising links from Tavily.
PHASE 4: SUMMARIZE
→ Use PaperList to summarize 3–5 top papers and insights.
"""

            messages = [
                SystemMessage("You are TechPaperMiner. Follow all 4 phases and call PaperList at the end."),
                HumanMessage(content=query)
            ]

            collected_documents = []
            final_summary_list = []

            for _ in range(7):
                ai_response = self.llm_with_tools.invoke(messages)
                messages.append(ai_response)

                if not ai_response.tool_calls:
                    messages.append(HumanMessage("Please call a tool or 'PaperList' to finish."))
                    continue

                tool_messages = []
                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call["name"]

                    if tool_name == "PaperList":
                        final_response_data = tool_call["args"]
                        final_summary_list = [PaperItem(**p).model_dump() for p in final_response_data.get("papers", [])]
                        break

                    tool_to_call = self.tools_map.get(tool_name)
                    if not tool_to_call:
                        tool_result = f"Error: Unknown tool '{tool_name}'."
                    else:
                        try:
                            tool_result = tool_to_call.invoke(tool_call["args"])
                        except Exception as e:
                            tool_result = f"Error running tool {tool_name}: {e}"

                    logger.info(f"🛠 TOOL CALL: {tool_name}({tool_call['args']})")
                    tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

                    new_docs = self._parse_results_to_documents(tool_name, tool_call["args"], tool_result)
                    collected_documents.extend(new_docs)

                messages.extend(tool_messages)
                if final_summary_list:
                    break

            if not final_summary_list:
                logger.error("LLM finished loop without providing a 'PaperList' summary.")
                try:
                    llm_structured = self.llm.with_structured_output(PaperList)
                    final_response = llm_structured.invoke(messages + [HumanMessage("Now call PaperList to summarize results.")])
                    final_summary_list = [paper.model_dump() for paper in final_response.papers]
                except Exception as e:
                    logger.error(f"Failed to generate final summary: {e}")
                    return {"success": False, "error": str(e)}

            return {
                "success": True,
                "output_summary": final_summary_list,
                "output_raw_docs": collected_documents,
                "output_type": "PaperReport",
                "meta": {"source": "LLM+Tools", "agent": "TechPaperMiner"},
            }

        except Exception as e:
            logger.exception("TechPaperMinerAgent failed.")
            return {"success": False, "error": str(e)}


    # =====================================================
    # 🧰 Fallback Mode
    # =====================================================
    def _fallback_papers(self, topic: str):
        summary = [
            {
                "title": f"Example paper on {topic}",
                "authors": ["Researcher One", "Researcher Two"],
                "summary": f"A fallback summary related to {topic}.",
                "source_url": "https://example.com/paper.pdf",
                "key_findings": ["Finding A", "Finding B"]
            }
        ]
        docs = [
            Document(
                page_content=f"Title: Example paper on {topic}\nSummary: A fallback summary related to {topic}.",
                metadata={"source": "https://example.com/paper.pdf", "data_source": "Fallback"}
            )
        ]
        return summary, docs


# =====================================================
# 🧪 Local Test
# =====================================================
if __name__ == "__main__":
    agent = TechPaperMinerAgent(use_llm=True)

    dummy_task = {
        "id": "T2",
        "title": "AI-Driven Static Code Analysis",
        "description": "Explore latest papers and methods on static analysis, AST parsing, dataflow, and vulnerability detection in code.",
        "priority": "High",
        "depends_on": [],
        "assigned_agent": "TechPaperMiner"
    }

    dummy_state = {"intent": {"idea": "AI-powered code security research"}}

    result = agent.run(dummy_task, dummy_state)
    print(json.dumps(result.get("output_summary", {}), indent=2))
