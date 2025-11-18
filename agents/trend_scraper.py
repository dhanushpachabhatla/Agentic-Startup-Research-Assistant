"""
trends_scraper.py
-------------------------------------
(Hybrid "Smart Collector") Agent that discovers trends and returns BOTH a summary and raw documents.

- Fetches data from:
    • News API (general headlines)
    • Reddit (developer/startup discussions)
    • Tavily Search (fallback)
- Uses Gemini LLM to adaptively plan collection AND summarize.

Compatible with LangChain v1.x+
"""

import os
import json
import requests
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

# Utility: News API Fetcher
def fetch_trending_news(topic: str = "technology", num_results: int = 10) -> str:
    """Fetch latest news articles using NewsAPI."""
    api_key = getattr(config, "NEWS_API_KEY", None)
    if not api_key:
        return "News API key not configured."

    try:
        url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize={num_results}&language=en&apiKey={api_key}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        return json.dumps(data.get("articles", []), indent=2)
    except Exception as e:
        logger.warning(f"News API failed: {e}")
        return f"News API failed: {e}"


# Utility: Reddit API (public JSON endpoint)
def fetch_reddit_trends(subreddit: str, limit: int = 5) -> str:
    """Fetch top Reddit posts from a specific, relevant subreddit (e.g., 'programming', 'startups', 'MachineLearning')."""
    try:
        logger.info(f"Attempting to fetch from subreddit: r/{subreddit}")
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.reddit.com/r/{subreddit}/top/.json?limit={limit}&t=day"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        posts = [
            {
                "title": post["data"]["title"],
                "score": post["data"]["score"],
                "url": f"https://reddit.com{post['data']['permalink']}"
            }
            for post in data["data"]["children"]
        ]
        return json.dumps(posts, indent=2)
    except Exception as e:
        logger.warning(f"Reddit fetch failed for r/{subreddit}: {e}")
        return f"Reddit fetch failed for r/{subreddit}: {e}"


# Utility: Tavily Search (fallback)
def tavily_trend_search(query: str, num_results: int = 5) -> str:
    """Web search to discover broad trends or 'drill down' on specific new topics."""
    api_key = getattr(config, "TAVILY_API_KEY", None)
    if not api_key:
        return "Tavily API key not configured."

    try:
        url = "https://api.tavily.com/search"
        payload = {"api_key": api_key, "query": query, "num_results": num_results}
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        
        return json.dumps(data.get("results", []), indent=2)
    except Exception as e:
        logger.warning(f"Tavily trend search failed: {e}")
        return f"Tavily trend search failed: {e}"


# Output Structure for Trends
class TrendItem(BaseModel):
    trend_name: str = Field(..., description="Name of the emerging trend, pain point, or developer need.")
    short_summary: str = Field(..., description="Brief summary of the trend, including who it affects (e.g., developers, tech leads).")
    relevance_score: int = Field(..., description="Relevance score to the user's task (0–100)")
    supporting_sources: List[str] = Field(..., description="List of supporting source URLs")

class TrendList(BaseModel):
    """A list of current trends. This is the REQUIRED format for the final summary."""
    trends: List[TrendItem]


# Trends Scraper Agent
class TrendsScraperAgent:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        self.llm_with_tools = None
        self.llm_with_final_structure = None

        # ---- Define Tools with updated descriptions ----
        news_tool = StructuredTool.from_function(
            func=fetch_trending_news,
            name="NewsAPITool",
            description="Fetch latest general news headlines related to a broad topic."
        )
        reddit_tool = StructuredTool.from_function(
            func=fetch_reddit_trends,
            name="RedditTrendTool",
            description="Fetch trending discussions from a *specific* subreddit. You must choose a relevant subreddit name (e.g., 'devops', 'programming', 'ExperiencedDevs')."
        )
        tavily_tool = StructuredTool.from_function(
            func=tavily_trend_search,
            name="TavilyTrendSearch",
            description="Perform a web search. Good for finding 'developer surveys', 'state of developer' reports, or specific pain points."
        )

        # Tool maps
        self.tools_map = {news_tool.name: news_tool, reddit_tool.name: reddit_tool, tavily_tool.name: tavily_tool}
        all_tools_for_llm = [news_tool, reddit_tool, tavily_tool, TrendList]

        # ---- Initialize Gemini ----
        if self.use_llm and getattr(config, "GEMINI_API_KEY22", None):
            os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY22
            
            self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
            
            logger.info("✅ Gemini LLM initialized for TrendsScraper.")

            # Tool binding
            self.llm_with_tools = self.llm.bind_tools(all_tools_for_llm)
            self.llm_with_final_structure = self.llm.with_structured_output(TrendList)
        else:
            logger.warning("❌ Gemini LLM not available; fallback mode will be used.")
            self.llm_with_tools = None

    # Helper for parsing tool results to Documents
    def _parse_results_to_documents(self, tool_name: str, tool_args: dict, tool_result_string: str) -> List[Document]:
        """Converts the raw JSON output from tools into a list of Document objects."""
        documents = []
        try:
            data = json.loads(tool_result_string)
            if not isinstance(data, list):
                return []

            if tool_name == "NewsAPITool":
                for article in data:
                    content = f"Title: {article.get('title', '')}\nDescription: {article.get('description', '')}"
                    metadata = {
                        "source": article.get('url', ''),
                        "title": article.get('title', ''),
                        "published_at": article.get('publishedAt', ''),
                        "data_source": "NewsAPI",
                        "query": tool_args.get("topic", "")
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            
            elif tool_name == "RedditTrendTool":
                for post in data:
                    content = post.get('title', '')
                    metadata = {
                        "source": post.get('url', ''),
                        "title": post.get('title', ''),
                        "score": post.get('score', 0),
                        "data_source": "Reddit",
                        "subreddit": tool_args.get("subreddit", "")
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            elif tool_name == "TavilyTrendSearch":
                for result in data:
                    content = result.get('content', '')
                    metadata = {
                        "source": result.get('url', ''),
                        "title": result.get('title', ''),
                        "score": result.get('score', 0),
                        "data_source": "Tavily",
                        "query": tool_args.get("query", "")
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
        
        except Exception as e:
            logger.warning(f"Failed to parse JSON for {tool_name}: {e}")
        
        return documents

    # Run Method (Rebuilt with Manual Loop)
    def run(self, task: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes trend analysis and returns BOTH summary and raw docs."""
        
        research_task_description = task.get("description") or task.get("topic") or state.get("intent", {}).get("idea", "technology")
        logger.info(f"🌍 Gathering trends for task: {research_task_description}")

        try:
            if not self.llm_with_tools:
                logger.warning("Using fallback trend set (no LLM active).")
                summary, docs = self._fallback_trends(research_task_description)
                return {
                    "success": True,
                    "output_summary": summary,
                    "output_raw_docs": docs,
                    "output_type": "TrendReport",
                    "meta": {"mode": "Fallback"},
                }

            query = f"""
You are an AI research analyst. Your goal is to find trends on given research task:
"{research_task_description}"

Follow this adaptive research plan:

PHASE 1: BROAD SEARCH & SURVEYS
* Use `TavilyTrendSearch` to find 2-3 recent "developer surveys" or other reports on given research task description. These are crucial for finding pain points.
* Use `NewsAPITool` to find general news on given research task description.

PHASE 2: DEEP DIVE
* Based on Phase 1, call `TavilyTrendSearch` again to "drill down" on a specific key pain point or tool you found.

PHASE 3: COMMUNITY PULSE
* Based on the topic, choose the *single best* technical/related subreddit to search (related to given research task description).
* Call `RedditTrendTool` with your chosen subreddit to find developer pain points.

PHASE 4: SUMMARIZE
* Once you have all the data, analyze it.
* Call `TrendList` to format your final summary of 3-5 key trends, needs, or pain points.
            """
            
            messages = [
                SystemMessage(
                    "You are TrendsScraper, an adaptive research agent. "
                    "You must follow the user's multi-phase plan. "
                    "Analyze tool results before deciding your next step. "
                    "Focus on given task description and AI tool adoption."
                    "Do not call `TrendList` until you have gathered all data."
                ),
                HumanMessage(content=query)
            ]
            
            collected_documents = [] # To store raw Document objects
            
            for _ in range(7): 
                ai_response = self.llm_with_tools.invoke(messages)
                messages.append(ai_response)
                
                if not ai_response.tool_calls:
                    logger.warning("LLM provided a text response instead of calling a tool.")
                    break

                if ai_response.tool_calls[0]["name"] == "TrendList":
                    logger.info("LLM is ready to provide final summary.")
                    break
                
                tool_messages = []
                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call["name"]
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
                    
                    # Convert and collect raw documents
                    new_docs = self._parse_results_to_documents(tool_name, tool_call["args"], tool_result)
                    collected_documents.extend(new_docs)
                
                messages.extend(tool_messages)

            # 3. The loop is finished. Force the LLM to give the final structured answer.
            messages.append(HumanMessage(
                "You now have all the information from the tools. "
                "Provide your final summary of the key trends by formatting everything as a 'TrendList'."
            ))
            
            final_response = self.llm_with_final_structure.invoke(messages)
            logger.info(f"🎁 FINAL RAW OUTPUT (Pydantic): {final_response}\n")
            
            summary_list = [trend.model_dump() for trend in final_response.trends]

            return {
                "success": True,
                "output_summary": summary_list,
                "output_raw_docs": collected_documents,
                "output_type": "TrendReport",
                "meta": {"source": "LLM+Tools", "agent": "TrendsScraper"},
            }

        except Exception as e:
            logger.exception("TrendsScraperAgent failed.")
            return {"success": False, "error": str(e)}

    # Fallback Trends
    def _fallback_trends(self, topic: str):
        """Returns a tuple of (summary_list, document_list)"""
        summary = [
            {
                "trend_name": "AI in Code Review",
                "short_summary": "Increasing use of AI to automate parts of code review, finding bugs, and suggesting improvements.",
                "relevance_score": 90,
                "supporting_sources": ["https://example.com/ai-code-review"]
            }
        ]
        
        docs = [
            Document(
                page_content="AI is being used to automate code reviews, which is a major pain point for developers.",
                metadata={"source": "https://example.com/ai-code-review", "title": "AI in Code Review", "data_source": "Fallback"}
            )
        ]
        return summary, docs

# Local Tes
if __name__ == "__main__":
    agent = TrendsScraperAgent(use_llm=True)
    
    dummy_task = {
        "id": "T5",
        "title": "Developer Productivity & AI Trends Analysis",
        "description": "Scrape recent articles, reports, and developer surveys to identify current trends in developer tools, AI/LLM adoption in development workflows (e.g., code generation, review, debugging), and evolving needs in code security and quality. Understand the primary users within a development team (e.g., individual developers, tech leads, security teams) and their pain points regarding code understanding and review.",
        "priority": "Medium",
        "depends_on": [],
        "assigned_agent": "TrendScraper"
    }
    
    dummy_state = {"intent": {"idea": "emerging developer trends"}}

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

    output_filename = "trends_scraper_report3.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        print(f"\n✅ Report successfully saved to {output_filename}")
    except Exception as e:
        print(f"\n❌ Failed to save report file: {e}")