"""
tech_paper_miner.py
-------------------------------------
(Hybrid "Smart Collector") Agent that finds and returns technical papers.
Returns BOTH a final JSON summary AND the raw documents for RAG.

- Fetches data from:
    • arXiv API (for pre-prints and papers)
    • Tavily Search (for blogs, news, and other papers)
    • Web Scraper (to get text from non-arXiv links)
- Uses Gemini LLM to plan collection AND summarize.
"""

import os
import json
import requests
import arxiv
from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime

from app.config import config

# === LangChain (v1.x+) imports ===
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.documents import Document
from bs4 import BeautifulSoup

# ----------------------------------------------------------
# Define the Structured Output Shape
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Utility Tools
# ----------------------------------------------------------

def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search the arXiv pre-print server for technical papers."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "pdf_url": result.pdf_url,
                "published_date": str(result.published.date())
            })
        
        return json.dumps(papers, indent=2)
    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return f"arXiv search failed: {e}"

def tavily_search(query: str, num_results: int = 5) -> str:
    """Search the web for research blogs, news, and non-arXiv papers."""
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
    """Smarter HTML scraper that extracts clean text from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        body_text = soup.body.get_text(separator=" ", strip=True)
        clean_text = body_text[:8000]
        return clean_text
    except Exception as e:
        return f"Failed to scrape {url}: {e}"

# ----------------------------------------------------------
# Tech Paper Miner Agent
# ----------------------------------------------------------
class TechPaperMinerAgent:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        self.llm_with_tools = None

        # ---- Define Tools ----
        arxiv_tool = StructuredTool.from_function(
            func=arxiv_search,
            name="arxiv_search",
            description="Search arXiv for scientific pre-prints and papers."
        )
        tavily_tool = StructuredTool.from_function(
            func=tavily_search,
            name="tavily_search",
            description="Search the web for research blogs, news, and non-arXiv papers."
        )
        scrape_tool = StructuredTool.from_function(
            func=scrape_website,
            name="scrape_website",
            description="Scrapes the clean text content from a URL (e.g., a blog or news article)."
        )

        # ---- Tool Maps ----
        self.tools_map = {
            arxiv_tool.name: arxiv_tool,
            tavily_tool.name: tavily_tool,
            scrape_tool.name: scrape_tool
        }
        all_tools_for_llm = [arxiv_tool, tavily_tool, scrape_tool, PaperList]

        # ---- Initialize Gemini ----
        if self.use_llm and getattr(config, "GEMINI_API_KEY8", None):
            os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY8
            self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
            logger.info("✅ Gemini LLM initialized for TechPaperMiner.")
            
            self.llm_with_tools = self.llm.bind_tools(all_tools_for_llm)
        else:
            logger.warning("❌ Gemini LLM not available; fallback mode will be used.")

    # ----------------------------------------------------------
    # Helper for parsing tool results to Documents
    # ----------------------------------------------------------
    def _parse_results_to_documents(self, tool_name: str, tool_args: dict, tool_result_string: str) -> List[Document]:
        """Converts the raw JSON/text output from tools into a list of Document objects."""
        documents = []
        try:
            if tool_name == "arxiv_search":
                papers = json.loads(tool_result_string)
                for paper in papers:
                    content = f"Title: {paper.get('title', '')}\nAuthors: {', '.join(paper.get('authors', []))}\nSummary: {paper.get('summary', '')}"
                    metadata = {
                        "source": paper.get('pdf_url', ''),
                        "title": paper.get('title', ''),
                        "authors": paper.get('authors', []),
                        "published_date": paper.get('published_date', ''),
                        "data_source": "arXiv",
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

    # ----------------------------------------------------------
    # Run Method (Manual Loop)
    # ----------------------------------------------------------
    def run(self, task: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes paper research and returns BOTH summary and raw docs."""
        
        # 🔻🔻🔻 [CHANGE 1: Updated Input Logic] 🔻🔻🔻
        # Use the 'description' field from the new task
        research_task_description = task.get("description") or task.get("topic") or state.get("intent", {}).get("idea", "latest AI research")
        logger.info(f"🔬 Mining for papers and research on: {research_task_description}")
        # 🔺🔺🔺 [END CHANGE 1] 🔺🔺🔺

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

            # 🔻🔻🔻 [CHANGE 2: Updated Prompt] 🔻🔻🔻
            # The prompt is now tailored to the new, complex task description
            query = f"""
You are an AI research assistant. Your goal is to find key technical papers, articles, and libraries related to the research task:
"{research_task_description}"

Follow this adaptive research plan:

PHASE 1: BROAD DISCOVERY
* Use `tavily_search` to find key concepts, libraries, and high-level articles on given research task

PHASE 2: ACADEMIC SEARCH
* Based on the concepts from Phase 1, use `arxiv_search` to find specific academic papers on given research task

PHASE 3: SCRAPE DETAILS
* If Phase 1 returned any *highly relevant* blog posts or articles (not PDFs), call `scrape_website` on their URLs to get the full text.

PHASE 4: SUMMARIZE
* After you have all the data, analyze the abstracts, snippets, and scraped text.
* Call `PaperList` to format your final summary of the most important 3-5 papers/articles/libraries.
            """
            # 🔺🔺🔺 [END CHANGE 2] 🔺🔺🔺
            
            messages = [
                SystemMessage(
                    "You are TechPaperMiner, an AI research assistant. "
                    "You must follow the user's multi-phase plan: 1. Tavily Search, 2. arXiv Search, 3. Scrape, 4. Format."
                    "You MUST call `PaperList` as your final action to finish."
                ),
                HumanMessage(content=query)
            ]
            
            collected_documents = [] # To store raw Document objects
            final_summary_list = []
            
            for _ in range(7): 
                ai_response = self.llm_with_tools.invoke(messages)
                messages.append(ai_response)
                
                if not ai_response.tool_calls:
                    logger.warning("LLM gave text response, re-prompting to call a tool...")
                    messages.append(HumanMessage("That was not a valid tool. You must call a tool, or call 'PaperList' to finish."))
                    continue 

                tool_messages = []
                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call["name"]
                    
                    if tool_name == "PaperList":
                        logger.info("LLM is providing the final summary.")
                        final_response_data = tool_call["args"]
                        final_summary_list = [PaperItem(**paper).model_dump() for paper in final_response_data.get('papers', [])]
                        break 
                    
                    tool_to_call = self.tools_map.get(tool_name)
                    
                    if not tool_to_call:
                        tool_result = f"Error: Unknown tool '{tool_name}'."
                    else:
                        try:
                            tool_result = tool_to_call.invoke(tool_call["args"])
                        except Exception as e:
                            tool_result = f"Error running tool {tool_name}: {e}"
                    
                    logger.info(f"🛠️ TOOL CALL: {tool_name}({tool_call['args']})")
                    logger.info(f"✨ TOOL RESULT: {tool_result[:200]}...\n")
                    
                    tool_messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                    )
                    
                    new_docs = self._parse_results_to_documents(tool_name, tool_call["args"], tool_result)
                    collected_documents.extend(new_docs)
                
                messages.extend(tool_messages)

                if final_summary_list:
                    break
            
            if not final_summary_list:
                logger.error("LLM finished loop without providing a 'PaperList' summary.")
                # Fallback: Try one last time to force a summary
                try:
                    logger.info("Attempting to force-generate summary from collected context...")
                    final_force_prompt = messages + [HumanMessage("You have all the data. Call 'PaperList' now.")]
                    # We need the LLM that is *forced* to output the structure
                    llm_with_final_structure = self.llm.with_structured_output(PaperList)
                    final_response = llm_with_final_structure.invoke(final_force_prompt)
                    final_summary_list = [paper.model_dump() for paper in final_response.papers]
                except Exception as e:
                    logger.error(f"Failed to force-generate summary: {e}")
                    return {
                        "success": False, "output_summary": [], "output_raw_docs": collected_documents,
                        "error": "LLM failed to provide a final summary."
                    }

            
            logger.info(f"🎁 FINAL RAW OUTPUT (Pydantic): {final_summary_list}\n")

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

    # ----------------------------------------------------------
    # Fallback
    # ----------------------------------------------------------
    def _fallback_papers(self, topic: str):
        """Returns a tuple of (summary_list, document_list)"""
        summary = [
            {
                "title": "A relevant paper for the topic",
                "authors": ["Researcher One", "Researcher Two"],
                "summary": f"A fallback paper summary related to {topic}.",
                "source_url": "https://example.com/paper.pdf",
                "key_findings": ["Finding A", "Finding B"]
            }
        ]
        
        docs = [
            Document(
                page_content=f"Title: A relevant paper for the topic\nSummary: A fallback paper summary related to {topic}.",
                metadata={
                    "source": "https://example.com/paper.pdf",
                    "title": "A relevant paper for the topic",
                    "authors": ["Researcher One", "Researcher Two"],
                    "data_source": "Fallback"
                }
            )
        ]
        return summary, docs

# ----------------------------------------------------------
# Local Test
# ----------------------------------------------------------
if __name__ == "__main__":
    agent = TechPaperMinerAgent(use_llm=True)
    
    # 🔻🔻🔻 [CHANGE 3: Updated Dummy Task] 🔻🔻🔻
    dummy_task = {
        "id": "T2",
        "title": "Advanced Static Analysis & Security Tech Research",
        "description": "Research state-of-the-art techniques and open-source/commercial libraries for static code analysis (e.g., AST parsing, data flow analysis, control flow analysis) and vulnerability scanning (SAST). Identify common metrics for code quality (e.g., cyclomatic complexity, test coverage, maintainability index) and prioritize types of vulnerabilities (e.g., OWASP Top 10). Investigate feasibility and considerations for multi-language support.",
        "priority": "High",
        "depends_on": [],
        "assigned_agent": "TechPaperMiner"
    }
    # 🔺🔺🔺 [END CHANGE 3] 🔺🔺🔺
    
    dummy_state = {"intent": {"idea": "AI-powered GitHub repository analyzer"}}

    result = agent.run(dummy_task, dummy_state)
    
    # --- Prepare Content for Console and File ---
    report_content = []
    report_content.append("--- 🏁 FINAL PROCESSED RESULT ---")
    
    if result['success']:
        report_content.append(f"Success: {result['success']}")
        report_content.append(f"Output Type: {result.get('output_type', 'N/A')}")
        
        # --- Part 1: Summary ---
        report_content.append("\n" + "--- 1. SUMMARY OUTPUT (JSON) ---")
        summary_json = json.dumps(result.get('output_summary', {}), indent=2)
        report_content.append(summary_json)
        
        # --- Part 2: Raw Docs ---
        report_content.append("\n" + "--- 2. RAW DOCUMENTS COLLECTED ---")
        raw_docs = result.get('output_raw_docs', [])
        report_content.append(f"Total Documents: {len(raw_docs)}")
        
        for i, doc in enumerate(raw_docs):
            report_content.append("\n" + f"--- Document {i+1} ---")
            report_content.append(f"Content: {doc.page_content}") # Full content
            report_content.append(f"Metadata: {doc.metadata}")
            report_content.append("-" * 20)
            
    else:
        report_content.append(f"Error: {result.get('error', 'Unknown error')}")

    # --- Print summary to console ---
    print("\n".join(report_content[:7])) # Print first few lines
    if result.get('success', False):
        print(f"\nTotal Documents Collected: {len(result.get('output_raw_docs', []))}")

    # --- Save full report to file ---
    output_filename = "tech_paper_miner_report.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        print(f"\n✅ Report successfully saved to {output_filename}")
    except Exception as e:
        print(f"\n❌ Failed to save report file: {e}")