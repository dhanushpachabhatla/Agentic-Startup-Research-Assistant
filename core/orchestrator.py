import json
from loguru import logger
from typing import TypedDict, List, Dict, Any, Set, Optional, Annotated
import operator 

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

# --- 1. Import Your Agents ---
from agents.competitor_scout import CompetitorScoutAgent
from agents.tech_paper_miner import TechPaperMinerAgent
from agents.trend_scraper import TrendsScraperAgent
from langchain_google_genai import ChatGoogleGenerativeAI 

# --- 2. Define the Graph's State ---
# 🔻🔻🔻 [FIX 1: Use Annotated to merge state correctly] 🔻🔻🔻
class GraphState(TypedDict):
    plan: Dict[str, Any]
    # This tells LangGraph to *add* to the set, not overwrite it
    completed_tasks: Annotated[Set[int], operator.ior] 
    # This tells LangGraph to *extend* the list, not overwrite it
    raw_documents: Annotated[List[Document], operator.add]
    agent_summaries: Annotated[List[Dict[str, Any]], operator.add]
    final_report: str 
# 🔺🔺🔺 [END FIX 1] 🔺🔺🔺

# --- 3. Helper Function to Find Tasks ---
def find_runnable_tasks(plan: Dict[str, Any], completed: Set[int]) -> List[Dict[str, Any]]:
    """Finds all tasks that can be run (dependencies are met)."""
    runnable_tasks = []
    all_tasks = plan.get("tasks", [])
    for task in all_tasks:
        task_id = task.get("id")
        if task_id in completed:
            continue 
        
        dependencies = set(task.get("depends_on", []))
        if dependencies.issubset(completed):
            runnable_tasks.append(task)
    
    return runnable_tasks

# --- 4. Define Agent Nodes ---

def _run_synthesis_task(state: GraphState, task: Dict[str, Any]) -> dict:
    """
    A helper function to run a final synthesis task (like T6 or T7).
    """
    logger.info(f"--- 🧠 Invoking Synthesizer for task: {task.get('id')} ---")
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
    
    prompt = f"""
You are a research analyst. Your task is to synthesize all the research collected so far to answer the objective in the task:
{json.dumps(task, indent=2)}

Here are the summaries from the previous collection tasks:
{json.dumps(state['agent_summaries'], indent=2)}

Here are snippets from the raw documents:
{json.dumps([doc.page_content[:200] for doc in state['raw_documents']], indent=2)}

Please generate the final report for this task now.
    """
    
    final_report = llm.invoke(prompt).content
    logger.info(f"Synthesis for {task.get('id')} complete: {final_report[:200]}...")
    
    return {
        "agent_summaries": [
            {"agent": task.get("assigned_agent"), "task_id": task.get("id"), "summary": final_report}
        ],
        "completed_tasks": {task.get("id")}
    }


def competitor_scout_node(state: GraphState):
    """
    Dynamically runs tasks for CompetitorScout.
    """
    logger.info("--- 🔬 Invoking Competitor Scout ---")
    plan = state['plan']
    completed = state['completed_tasks']
    
    task = None
    for t in plan.get("tasks", []):
        if t.get("id") not in completed and t.get("assigned_agent") == "CompetitorScout":
            dependencies = set(t.get("depends_on", []))
            if dependencies.issubset(completed):
                task = t
                break 
            
    if not task:
        logger.warning("CompetitorScout called, but no pending task found.")
        return {"completed_tasks": set()}

    logger.info(f"Running task: {task.get('id')} - {task.get('title')}")
    
    if not task.get("depends_on"):
        agent = CompetitorScoutAgent()
        result = agent.run(task, state)
        return {
            "raw_documents": result.get("output_raw_docs", []),
            "agent_summaries": [
                {"agent": "CompetitorScout", "task_id": task.get("id"), "summary": result.get("output_summary", {})}
            ],
            "completed_tasks": {task.get("id")}
        }
    else:
        return _run_synthesis_task(state, task)

def tech_paper_miner_node(state: GraphState):
    """
    Dynamically runs tasks for TechPaperMiner.
    """
    logger.info("--- 📚 Invoking Tech Paper Miner ---")
    plan = state['plan']
    completed = state['completed_tasks']
    
    task = None
    for t in plan.get("tasks", []):
        if t.get("id") not in completed and t.get("assigned_agent") == "TechPaperMiner":
            dependencies = set(t.get("depends_on", []))
            if dependencies.issubset(completed):
                task = t
                break 
            
    if not task:
        logger.warning("TechPaperMiner called, but no pending task found.")
        return {"completed_tasks": set()} 

    logger.info(f"Running task: {task.get('id')} - {task.get('title')}")

    if not task.get("depends_on"):
        agent = TechPaperMinerAgent()
        result = agent.run(task, state)
        return {
            "raw_documents": result.get("output_raw_docs", []),
            "agent_summaries": [
                {"agent": "TechPaperMiner", "task_id": task.get("id"), "summary": result.get("output_summary", {})}
            ],
            "completed_tasks": {task.get("id")}
        }
    else:
        return _run_synthesis_task(state, task)

def trend_scraper_node(state: GraphState):
    """
    Dynamically runs tasks for TrendsScraper.
    """
    logger.info("--- 📈 Invoking Trend Scraper ---")
    plan = state['plan']
    completed = state['completed_tasks']
    
    task = None
    for t in plan.get("tasks", []):
        if t.get("id") not in completed and t.get("assigned_agent") == "TrendScraper":
            dependencies = set(t.get("depends_on", []))
            if dependencies.issubset(completed):
                task = t
                break

    if not task:
        logger.warning("TrendScraper called, but no pending task found.")
        return {"completed_tasks": set()}
        
    logger.info(f"Running task: {task.get('id')} - {task.get('title')}")
    
    if not task.get("depends_on"):
        agent = TrendsScraperAgent()
        result = agent.run(task, state)
        
        return {
            "raw_documents": result.get("output_raw_docs", []),
            "agent_summaries": [
                {"agent": "TrendsScraper", "task_id": task.get("id"), "summary": result.get("output_summary", {})}
            ],
            "completed_tasks": {task.get("id")}
        }
    else:
        return _run_synthesis_task(state, task)

# --- 5. Define the Router (The Graph's "Brain") ---

def get_next_node(state: GraphState):
    """
    This is the main router. It checks the state and decides
    which *single* node to run next.
    """
    plan = state['plan']
    completed = state['completed_tasks']
    
    runnable_tasks = find_runnable_tasks(plan, completed)
    
    if not runnable_tasks:
        logger.info("All tasks complete. Ending graph.")
        return END
    
    # Get the *first* runnable task
    task_to_run = runnable_tasks[0]
    agent_name = task_to_run.get("assigned_agent")
    
    logger.info(f"Routing to: {agent_name} for task {task_to_run.get('id')}")
    
    if agent_name == "CompetitorScout":
        return "competitor_scout"
    elif agent_name == "TechPaperMiner":
        return "tech_paper_miner"
    elif agent_name == "TrendScraper":
        return "trend_scraper"
    else:
        logger.error(f"Unknown agent: {agent_name}")
        return END

def router_node(state: GraphState):
    """
    This node is just a "gate" or "junction".
    """
    logger.info("--- Routing tasks... ---")
    return {}


# --- 6. Build the Graph ---

logger.info("Building the orchestrator graph...")

# 🔻🔻🔻 [FIX 2: Remove the manual merge_state function] 🔻🔻🔻
# def merge_state(old_state: GraphState, new_partial_state: dict) -> GraphState:
#     ...
# 🔺🔺🔺 [END FIX 2] 🔺🔺🔺

# 🔻🔻🔻 [FIX 3: Initialize graph without merge_state] 🔻🔻🔻
# The Annotated types in GraphState will handle merging automatically.
workflow = StateGraph(GraphState)
# 🔺🔺🔺 [END FIX 3] 🔺🔺🔺

# Add the nodes
workflow.add_node("competitor_scout", competitor_scout_node)
workflow.add_node("tech_paper_miner", tech_paper_miner_node)
workflow.add_node("trend_scraper", trend_scraper_node)
workflow.add_node("router_node", router_node)

# The entry point is the simple router_node
workflow.set_entry_point("router_node")

workflow.add_conditional_edges(
    "router_node",
    get_next_node, 
    {
        "competitor_scout": "competitor_scout",
        "tech_paper_miner": "tech_paper_miner",
        "trend_scraper": "trend_scraper",
        END: END
    }
)

# Add the loops: After each agent runs, it goes back to the router_node
workflow.add_edge("competitor_scout", "router_node")
workflow.add_edge("tech_paper_miner", "router_node")
workflow.add_edge("trend_scraper", "router_node")

# Compile the graph
app = workflow.compile()

logger.info("✅ Graph compiled successfully.")

# --- 7. Run the Orchestrator ---
if __name__ == "__main__":
    
    # This is your new plan
    planner_output = {
      "research_goal": "Define a comprehensive product concept and initial feature set for an AI-powered GitHub repository analysis tool...",
      "suggested_agents": [
        "CompetitorScout",
        "TechPaperMiner",
        "TrendScraper"
      ],
      "tasks": [
        {
          "id": 1,
          "title": "Competitor Feature & Gap Analysis",
          "description": "Analyze specified competitors (SonarQube, Snyk, CodeClimate, DeepSource, GitHub Copilot, various SAST tools)...",
          "priority": "High",
          "depends_on": [],
          "assigned_agent": "CompetitorScout"
        },
        {
          "id": 2,
          "title": "Emerging Trends in Developer Tools & Cybersecurity",
          "description": "Research current and future trends in software development, developer tools, AI/ML integration in dev workflows...",
          "priority": "High",
          "depends_on": [],
          "assigned_agent": "TrendScraper"
        },
        {
          "id": 3,
          "title": "State-of-the-Art in AI Code Analysis & NLP/LLMs",
          "description": "Investigate the latest advancements and research papers in AI/ML for static code analysis, bug detection...",
          "priority": "High",
          "depends_on": [],
          "assigned_agent": "TechPaperMiner"
        },
        # --- This is just a partial plan for testing ---
        # {
        #   "id": 4,
        #   "title": "Design Pattern Recognition Techniques",
        #   "description": "Conduct a deep dive into existing and emerging techniques, algorithms, and tools for automatically recognizing...",
        #   "priority": "High",
        #   "depends_on": [],
        #   "assigned_agent": "TechPaperMiner"
        # }
      ],
    }
    
    # This is the initial state we pass to the graph
    inputs = {
        "plan": planner_output,
        "completed_tasks": set(),
        "raw_documents": [],
        "agent_summaries": [],
        "final_report": ""
    }
    
    logger.info("--- 🚀 Invoking the Orchestrator with Dynamic Plan ---")
    
    # We use app.invoke() ONE time.
    final_state = app.invoke(inputs, {"recursion_limit": 15})
    
    logger.info("--- 🏁 Orchestrator Finished ---")
    
    # --- Save the final aggregated output to files ---
    
    if final_state:
        # --- 1. Save Summaries ---
        agent_summaries = final_state.get('agent_summaries', [])
        summary_filename = "data/memory_store/agent_summaries.json"
        try:
            with open(summary_filename, "w", encoding="utf-8") as f:
                json.dump(agent_summaries, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Summaries saved to {summary_filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save summaries file: {e}")

        # --- 2. Save Raw Documents ---
        raw_docs_list = final_state.get('raw_documents', [])
        
        # Convert Document objects to a JSON-serializable list of dicts
        serializable_docs = []
        for doc in raw_docs_list:
            serializable_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        docs_filename = "data/raw_docs/raw_docs.json"
        try:
            with open(docs_filename, "w", encoding="utf-8") as f:
                json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Raw documents saved to {docs_filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save raw documents file: {e}")

        # --- 3. Print a summary to console ---
        logger.info(f"Total Raw Documents Collected: {len(raw_docs_list)}")
        logger.info(f"Total Summaries Collected: {len(agent_summaries)}")

    else:
        logger.error("Graph execution failed to return a final state.")

# 🔺🔺🔺 [END MODIFIED BLOCK] 🔺🔺🔺