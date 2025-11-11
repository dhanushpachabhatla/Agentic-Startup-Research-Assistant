"""
trends_scraper.py
-------------------------------------
(Hybrid) Agent that discovers trends and returns BOTH a summary and raw documents.

- Fetches data from:
    • News API (general headlines)
    • Reddit (developer/startup discussions)
    • Tavily Search (fallback)
- Uses Gemini LLM to plan collection AND summarize.

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

# 🔻🔻🔻 [NEW IMPORTS] 🔻🔻🔻
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.documents import Document
# 🔺🔺🔺 [END NEW IMPORTS] 🔺🔺🔺


# ----------------------------------------------------------
# Utility: News API Fetcher
# ----------------------------------------------------------
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
        
        # Return the raw articles list as a JSON string
        return json.dumps(data.get("articles", []), indent=2)
    except Exception as e:
        logger.warning(f"News API failed: {e}")
        return f"News API failed: {e}"


# ----------------------------------------------------------
# Utility: Reddit API (public JSON endpoint)
# ----------------------------------------------------------
def fetch_reddit_trends(subreddit: str = "technology", limit: int = 5) -> str:
    """Fetch top Reddit posts from a given subreddit."""
    try:
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
        logger.warning(f"Reddit fetch failed: {e}")
        return f"Reddit fetch failed: {e}"


# ----------------------------------------------------------
# Utility: Tavily Search (fallback)
# ----------------------------------------------------------
def tavily_trend_search(query: str, num_results: int = 5) -> str:
    """Fallback trend search using Tavily API."""
    api_key = getattr(config, "TAVILY_API_KEY", None)
    if not api_key:
        return "Tavily API key not configured."

    try:
        url = "https://api.tavily.com/search"
        payload = {"api_key": api_key, "query": query, "num_results": num_results}
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        
        # Return the raw results list as a JSON string
        return json.dumps(data.get("results", []), indent=2)
    except Exception as e:
        logger.warning(f"Tavily trend search failed: {e}")
        return f"Tavily trend search failed: {e}"


# ----------------------------------------------------------
# Output Structure for Trends
# ----------------------------------------------------------
class TrendItem(BaseModel):
    trend_name: str = Field(..., description="Name of the emerging trend")
    short_summary: str = Field(..., description="Brief summary of the trend")
    relevance_score: int = Field(..., description="Relevance score between 0–100")
    supporting_sources: List[str] = Field(..., description="List of supporting source URLs")

class TrendList(BaseModel):
    """A list of current trends. This is the REQUIRED format for the final summary."""
    trends: List[TrendItem]


# ----------------------------------------------------------
# Trends Scraper Agent
# ----------------------------------------------------------
class TrendsScraperAgent:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        self.llm_with_tools = None
        self.llm_with_final_structure = None

        # Define tools
        news_tool = StructuredTool.from_function(
            func=fetch_trending_news,
            name="NewsAPITool",
            description="Fetch latest news headlines related to a topic using NewsAPI."
        )
        reddit_tool = StructuredTool.from_function(
            func=fetch_reddit_trends,
            name="RedditTrendTool",
            description="Fetch trending Reddit posts from a topic-related subreddit."
        )
        tavily_tool = StructuredTool.from_function(
            func=tavily_trend_search,
            name="TavilyTrendSearch",
            description="Fallback web search to identify trending discussions or topics."
        )

        # Tool maps
        self.tools_map = {news_tool.name: news_tool, reddit_tool.name: reddit_tool, tavily_tool.name: tavily_tool}
        all_tools_for_llm = [news_tool, reddit_tool, tavily_tool, TrendList]

        # ---- Initialize Gemini ----
        if self.use_llm and getattr(config, "GEMINI_API_KEY2", None):
            os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY2
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

    # ----------------------------------------------------------
    # NEW: Helper for parsing tool results to Documents
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Run Method (Rebuilt with Manual Loop)
    # ----------------------------------------------------------
    def run(self, task: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes trend analysis and returns BOTH summary and raw docs."""
        topic = task.get("topic") or state.get("intent", {}).get("idea", "technology")
        logger.info(f"🌍 Gathering trends for topic: {topic}")

        try:
            if not self.llm_with_tools:
                logger.warning("Using fallback trend set (no LLM active).")
                summary, docs = self._fallback_trends(topic)
                return {
                    "success": True,
                    "output_summary": summary,
                    "output_raw_docs": docs,
                    "output_type": "TrendReport",
                    "meta": {"mode": "Fallback"},
                }

            query = f"""
You are an AI research analyst discovering emerging trends in the tech/startup world.
Your task is to find trends related to: "{topic}"

Here is the plan you must follow:
1.  Use `NewsAPITool` to find recent news articles.
2.  Use `RedditTrendTool` to find community discussions on a relevant subreddit (e.g., 'technology', 'startups', 'AI').
3.  Use `TavilyTrendSearch` to get a broader web perspective.
4.  After you have collected data from at least two tools, analyze all the results.
5.  Finally, call `TrendList` to format your final summary of 3-5 key trends.
            """
            
            messages = [
                SystemMessage(
                    "You are TrendsScraper, a research agent. "
                    "You must follow the user's plan: 1. Call tools to gather data. 2. Call `TrendList` to summarize."
                ),
                HumanMessage(content=query)
            ]
            
            collected_documents = [] # To store raw Document objects
            
            for _ in range(5): # Limit to 5 steps
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

    # ----------------------------------------------------------
    # Fallback Trends
    # ----------------------------------------------------------
    def _fallback_trends(self, topic: str):
        """Returns a tuple of (summary_list, document_list)"""
        summary = [
            {
                "trend_name": "AI Agents in Productivity Tools",
                "short_summary": "Rapid adoption of AI copilots and assistants.",
                "relevance_score": 90,
                "supporting_sources": ["https://www.theverge.com"]
            },
            {
                "trend_name": "Edge AI Deployment",
                "short_summary": "Running models locally for privacy.",
                "relevance_score": 85,
                "supporting_sources": ["https://venturebeat.com"]
            },
        ]
        
        docs = [
            Document(
                page_content="Rapid adoption of AI copilots and assistants in daily workflows like Notion, Google Workspace, and Figma.",
                metadata={"source": "https://www.theverge.com", "title": "AI Agents in Productivity Tools", "data_source": "Fallback"}
            ),
            Document(
                page_content="Increasing focus on running lightweight AI models locally for privacy and latency gains.",
                metadata={"source": "https://venturebeat.com", "title": "Edge AI Deployment", "data_source": "Fallback"}
            )
        ]
        return summary, docs


# ----------------------------------------------------------
# Local Test
# ----------------------------------------------------------
if __name__ == "__main__":
    agent = TrendsScraperAgent(use_llm=True)
    dummy_task = {
        "id": "T3",
        "title": "Tech Trend Discovery",
        "topic": "AI-powered startups",
    }
    dummy_state = {"intent": {"idea": "emerging AI startups"}}

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
        report_content.append(f"Error: {result['error']}")

    # --- Print summary to console ---
    print("\n".join(report_content[:7])) # Print first few lines
    if result.get('success', False):
        print(f"\nTotal Documents Collected: {len(result.get('output_raw_docs', []))}")


    # --- Save full report to file ---
    output_filename = "trends_scraper_report.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        print(f"\n✅ Report successfully saved to {output_filename}")
    except Exception as e:
        print(f"\n❌ Failed to save report file: {e}")