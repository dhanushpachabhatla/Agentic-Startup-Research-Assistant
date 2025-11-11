"""
competitor_scout.py
-------------------------------------
(Hybrid) LangChain 1.x+ CompetitorScout
Returns BOTH a final JSON summary AND the raw documents for RAG.
- Gemini (LLM reasoning)
- Tavily Search
- Web Scraping
- Manual Agent Loop
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional
from loguru import logger
from app.config import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Optional

# 🔻🔻🔻 [NEW IMPORT] 🔻🔻🔻
from langchain_core.documents import Document
# 🔺🔺🔺 [NEW IMPORT] 🔺🔺🔺

from bs4 import BeautifulSoup # <-- Make sure this is imported

# ----------------------------------------------------------
# Define the Structured Output Shape
# ----------------------------------------------------------

class Competitor(BaseModel):
    """A single competitor's details."""
    name: str = Field(..., description="The name of the competitor company.")
    domain: str = Field(..., description="The market or domain they operate in (e.g., 'Code Security').")
    summary: str = Field(..., description="A brief summary of what the competitor does.")
    website: Optional[str] = Field(None, description="The competitor's main website URL.")
    reason_for_similarity: str = Field(..., description="Why this company is a competitor to the user's idea.")
    estimated_similarity_score: int = Field(..., description="A 0-100 score of how similar they are.", ge=0, le=100)
    
    key_features: Optional[List[str]] = Field(
        None, description="A list of key features or product offerings."
    )
    pricing_model: Optional[str] = Field(
        None, description="The competitor's pricing model (e.g., 'Freemium', 'Enterprise', 'Open Source')."
    )
    target_audience: Optional[str] = Field(
        None, description="The primary target audience (e.g., 'Hobbyist Developers', 'Enterprise DevOps Teams')."
    )

class CompetitorList(BaseModel):
    """A list of competitors. This is the REQUIRED format for the final answer."""
    competitors: List[Competitor]

# ----------------------------------------------------------
# Utility Tools
# ----------------------------------------------------------

def tavily_search(query: str, num_results: int = 5) -> str:
    """Search for top competitor companies using Tavily API."""
    if not getattr(config, "TAVILY_API_KEY", None):
        return "Tavily API key not configured."

    try:
        url = "https://api.tavily.com/search"
        payload = {"api_key": config.TAVILY_API_KEY, "query": query, "num_results": num_results}
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return json.dumps(results[:num_results], indent=2)
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return f"Tavily search failed: {e}"

def scrape_website(url: str) -> str:
    """Smarter HTML scraper that extracts clean text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        
        soup = BeautifulSoup(resp.text, "html.parser")
        body_text = soup.body.get_text(separator=" ", strip=True)
        clean_text = body_text[:8000] 
        
        # Return only the clean text, not the f-string
        return clean_text
    
    except Exception as e:
        return f"Failed to scrape {url}: {e}"


# ----------------------------------------------------------
# Competitor Scout (Simplified Chain)
# ----------------------------------------------------------
class CompetitorScoutAgent:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        self.llm_with_tools = None
        self.llm_with_final_structure = None
        
        tavily_tool = StructuredTool.from_function(
            func=tavily_search,
            name="tavily_search",
            description="Search for top competitor companies using Tavily API."
        )
        scrape_tool = StructuredTool.from_function(
            func=scrape_website,
            name="scrape_website",
            description="Scrapes the clean text content from a competitor's website URL."
        )
        
        self.tools_map = {tavily_tool.name: tavily_tool, scrape_tool.name: scrape_tool}
        all_tools_for_llm = [tavily_tool, scrape_tool, CompetitorList]

        if self.use_llm and getattr(config, "GEMINI_API_KEY4", None):
            os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY4
            
            self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
            logger.info("✅ Gemini LLM initialized for CompetitorScout.")
            
            self.llm_with_tools = self.llm.bind_tools(all_tools_for_llm)
            self.llm_with_final_structure = self.llm.with_structured_output(CompetitorList)
        else:
            logger.warning("❌ Gemini LLM not available; fallback mode will be used.")

    # 🔻🔻🔻 [NEW HELPER FUNCTION] 🔻🔻🔻
    def _parse_results_to_documents(self, tool_name: str, tool_args: dict, tool_result_string: str) -> List[Document]:
        """Converts the raw JSON/text output from tools into a list of Document objects."""
        documents = []
        try:
            if tool_name == "tavily_search":
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
    # 🔺🔺🔺 [END NEW HELPER FUNCTION] 🔺🔺🔺

    # ----------------------------------------------------------
    # Main Run Function
    # ----------------------------------------------------------
    def run(self, task: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes competitor analysis and returns BOTH summary and raw docs."""
        try:
            description = task.get("description") or state.get("intent", {}).get("idea", "")
            if not description:
                return {"success": False, "error": "No valid task description provided."}

            logger.info(f"🧠 Analyzing competitors for: {description}")

            if not self.llm_with_tools:
                logger.warning("Using fallback competitor data (LLM unavailable).")
                # (You may want to update this to return Document objects too)
                return {
                    "success": True,
                    "output_summary": self._fallback_competitors(description),
                    "output_raw_docs": [],
                    "meta": {"mode": "Fallback"},
                }
            
            query = f"""
Analyze competitors for the following startup research task:

{description}

Here is the exact plan you must follow:
1.  First, call `tavily_search` to find a list of potential competitors and their URLs.
2.  Second, for EACH competitor, you MUST call `scrape_website` on their URL to get detailed content. You will need this content to find their features, pricing, and audience.
3.  After you have scraped the websites, analyze all the collected text.
4.  Finally, call the `CompetitorList` tool to provide your complete analysis.
            """
            
            messages = [
                SystemMessage(
                    "You are CompetitorScout, a market intelligence agent. "
                    "You must follow the user's plan exactly: 1. Search, 2. Scrape, 3. Analyze, 4. Format."
                    "Do NOT call CompetitorList until you have scraped the websites."
                ),
                HumanMessage(content=query)
            ]

            # 🔻🔻🔻 [NEW] Initialize empty list for raw docs 🔻🔻🔻
            collected_documents = []
            
            for _ in range(10): 
                ai_response = self.llm_with_tools.invoke(messages)
                messages.append(ai_response)
                
                if not ai_response.tool_calls:
                    logger.warning("LLM provided a text response instead of calling a tool.")
                    break

                if ai_response.tool_calls[0]["name"] == "CompetitorList":
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
                    logger.info(f"✨ TOOL RESULT: {tool_result[:200]}...\n") # Log snippet
                    
                    tool_messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                    )

                    # 🔻🔻🔻 [NEW] Collect raw docs 🔻🔻🔻
                    new_docs = self._parse_results_to_documents(tool_name, tool_call["args"], tool_result)
                    collected_documents.extend(new_docs)
                    # 🔺🔺🔺 [END NEW] 🔺🔺🔺
                
                messages.extend(tool_messages)

            messages.append(HumanMessage(
                "You now have all the information from searching and scraping. "
                "Provide your final analysis by formatting everything as a 'CompetitorList'."
            ))
            
            final_response = self.llm_with_final_structure.invoke(messages)
            logger.info(f"🎁 FINAL RAW OUTPUT (Pydantic): {final_response}\n")

            competitors_list = [comp.model_dump() for comp in final_response.competitors]

            # 🔻🔻🔻 [NEW] Updated return statement 🔻🔻🔻
            return {
                "success": True,
                "output_summary": competitors_list,
                "output_raw_docs": collected_documents,
                "output_type": "CompetitorAnalysisReport",
                "meta": {"source": "LLM+Tools", "agent": "CompetitorScout (Manual)"},
            }
            # 🔺🔺🔺 [END NEW] 🔺🔺🔺

        except Exception as e:
            logger.exception("CompetitorScoutAgent failed.")
            return {"success": False, "error": str(e)}


    # ----------------------------------------------------------
    # Fallback
    # ----------------------------------------------------------
    def _fallback_competitors(self, idea: str) -> List[Dict[str, Any]]:
        # This only returns the summary part
        return [
            {
                "name": "GitGuardian", "domain": "Code Security", "summary": "Focuses on secrets detection...",
                "reason_for_similarity": f"Focuses on code security like '{idea}'.", "website": "https://www.gitguardian.com",
                "estimated_similarity_score": 85, "key_features": ["Secrets detection"], "pricing_model": "Freemium",
                "target_audience": "Developers"
            },
            {
                "name": "SonarQube", "domain": "Code Quality Analysis", "summary": "Static code analysis for quality...",
                "reason_for_similarity": f"Performs static code analysis like '{idea}'.", "website": "https://www.sonarsource.com",
                "estimated_similarity_score": 90, "key_features": ["Static code analysis"], "pricing_model": "Open Source",
                "target_audience": "Enterprise DevOps Teams"
            }
        ]


# ----------------------------------------------------------
# Local Test
# ----------------------------------------------------------
if __name__ == "__main__":
    agent = CompetitorScoutAgent(use_llm=True)
    # dummy_task = {
    #     "id": "T1",
    #     "title": "Competitor Landscape Analysis",
    #     "description": (
    #         "Identify and analyze existing tools for static code analysis, SAST, code quality, and "
    #         "design pattern detection. Focus on features, pricing models, target audience, "
    #         "integration capabilities (e.g., GitHub), and interactive interfaces."
    #     ),
    # }
    dummy_task =  { 
      "id": "T1", 
      "title": "Competitor Landscape Analysis", 
      "description": "Identify and analyze existing tools for static code analysis, SAST (Static Application Security Testing), code quality, and design pattern detection. Focus on their features, pricing models, target audience, integration capabilities (e.g., GitHub), and how they provide insights. Specifically look for tools offering interactive or conversational interfaces, and identify their strengths, weaknesses, and market gaps.", 
      "priority": "High", 
      "depends_on": [], 
      "assigned_agent": "CompetitorScout" 
    }
    dummy_state = {"intent": {"idea": "AI-powered GitHub repository analyzer"}}

    result = agent.run(dummy_task, dummy_state)
    
    # 🔻🔻🔻 [NEW] Print to console AND save to file 🔻🔻🔻
    
    # --- 1. Prepare Content for Console and File ---
    report_content = []
    report_content.append("--- 🏁 FINAL PROCESSED RESULT ---")
    
    if result['success']:
        report_content.append(f"Success: {result['success']}")
        report_content.append(f"Output Type: {result['output_type']}")
        
        # --- Part 1: Summary ---
        report_content.append("\n" + "--- 1. SUMMARY OUTPUT (JSON) ---")
        summary_json = json.dumps(result['output_summary'], indent=2)
        report_content.append(summary_json)
        
        # --- Part 2: Raw Docs ---
        report_content.append("\n" + "--- 2. RAW DOCUMENTS COLLECTED ---")
        raw_docs = result['output_raw_docs']
        report_content.append(f"Total Documents: {len(raw_docs)}")
        
        for i, doc in enumerate(raw_docs):
            report_content.append("\n" + f"--- Document {i+1} ---")
            report_content.append(f"Content: {doc.page_content[:]}...") # Append a snippet
            report_content.append(f"Metadata: {doc.metadata}")
            report_content.append("-" * 20)
            
    else:
        report_content.append(f"Error: {result['error']}")

    # --- 2. Print all content to console ---
    print("\n".join(report_content))

    # --- 3. Save all content to a text file ---
    output_filename = "competitor_scout_report2.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        print(f"\n✅ Report successfully saved to {output_filename}")
    except Exception as e:
        print(f"\n❌ Failed to save report file: {e}")
    # 🔺🔺🔺 [END NEW] 🔺🔺🔺