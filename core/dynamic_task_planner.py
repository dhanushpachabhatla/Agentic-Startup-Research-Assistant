import os
import json
import logging
from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class DynamicTaskPlanner:
    """
    Dynamic Task Planner:
    - Converts parsed intent → adaptive subtask plan
    - Selects agents dynamically based on business domain and data needs
    - Outputs structured task plan for the Orchestrator
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm and config.GEMINI_API_KEY6:
            os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY6
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.4,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            logger.info("✅ Gemini LLM initialized for DynamicTaskPlanner.")
        else:
            self.llm = None
            logger.warning("⚠️ Using fallback deterministic plan (no LLM active).")

    # -----------------------------------------------------------------
    def _build_prompt(self, intent: Dict[str, Any]) -> str:
        return f"""
You are an intelligent **Dynamic Task Planner** for an Agentic Startup Research Assistant.

Given this parsed startup intent:
{json.dumps(intent, indent=2)}

Generate a **pure JSON** plan with these keys:
- research_goal: short summary of what needs to be researched
- suggested_agents: list of agents to invoke next (choose from: ['CompetitorScout', 'TechPaperMiner', 'TrendScraper'])
- tasks: detailed list of subtasks (each with id, title, description, priority, depends_on, and assigned_agent)
- expected_outputs: what type of insights each subtask should produce (e.g., reports, summaries, datasets)
- reasoning_notes: why these tasks and agents were chosen
- max 4 tasks

⚠️ Strict rules:
- Return **only JSON**, no explanations, code blocks, or markdown.
- Keep field names exactly as above.
"""

    # -----------------------------------------------------------------
    def _generate_plan_with_llm(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(intent)
        try:
            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            clean_json = self._extract_json(text)

            plan = json.loads(clean_json)
            logger.info("✅ Dynamic plan generated successfully via Gemini.")
            return plan

        except Exception as e:
            logger.error(f"⚠️ LLM planning failed: {e}")
            return self._fallback_plan(intent)

    # -----------------------------------------------------------------
    def _extract_json(self, text: str) -> str:
        """
        Safely extract JSON from LLM output that may contain text, code fences, etc.
        """
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        # Attempt to find JSON braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        return text

    # -----------------------------------------------------------------
    def _fallback_plan(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        tasks, agents = [], []

        tech_keywords = intent.get("tech_keywords", [])
        if isinstance(tech_keywords, str):
            tech_keywords = [x.strip() for x in tech_keywords.split(",")]

        if any(k in tech_keywords for k in ["AI", "Machine Learning", "Blockchain", "Computer Vision"]):
            agents.append("TechPaperMiner")
        if "competitor" in intent.get("raw_query", "").lower() or "market" in intent.get("raw_query", "").lower():
            agents.append("CompetitorScout")
        agents.append("TrendScraper")

        for i, ag in enumerate(agents, 1):
            tasks.append({
                "id": i,
                "title": f"{ag} Task",
                "description": f"Run {ag} to collect insights related to startup idea.",
                "priority": i,
                "depends_on": [],
                "assigned_agent": ag
            })

        return {
            "research_goal": "Generate initial market, competitor, and tech insights.",
            "suggested_agents": agents,
            "tasks": tasks,
            "expected_outputs": ["raw_docs", "summaries", "trend_data"],
            "reasoning_notes": "Fallback deterministic plan."
        }

    # -----------------------------------------------------------------
    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent = state.get("intent", {})
        if not intent:
            logger.error("❌ No intent found in state.")
            return state

        logger.info("🧠 Generating dynamic task plan...")
        plan = self._generate_plan_with_llm(intent) if self.llm else self._fallback_plan(intent)
        state["task_plan"] = plan
        return state


# -----------------------------------------------------------------
# Manual test
# -----------------------------------------------------------------
if __name__ == "__main__":

    intent_example = {'industry': 'Software Development, Developer Tools, AI/ML, Cybersecurity, Software Quality Assurance', 'business_model': 'Freemium, Subscription (SaaS), Enterprise Licensing', 'target_audience': 'Software Developers, Engineering Teams, Tech Leads, CTOs, Security Teams', 'tech_keywords': 'Code Analysis, Static Analysis, AI, Machine Learning, Natural Language Processing (NLP), Large Language Models (LLMs), GitHub API, Cybersecurity, Software Quality Assurance, Design Pattern Recognition', 'competitor_names': 'SonarQube, Snyk, CodeClimate, DeepSource, GitHub Copilot (for Q&A aspects), various SAST tools (e.g., Checkmarx, Fortify)', 'intent_type': 'Product Concept Definition, Startup Idea Generation, Feature Specification', 'problem_statement': "Developers lack an easy, interactive, and comprehensive way to get immediate, detailed insights into their codebase's quality, security, structure, and design patterns, often relying on disparate tools or manual reviews.", 'solution_summary': 'An AI-powered GitHub repository analysis tool with a chatbot interface that provides on-demand, interactive insights into code quality, potential bugs, security vulnerabilities, architectural structure, and identified design patterns by simply providing a repository link.', 'data_needs': 'Access to GitHub repository source code (via API), historical commit data, extensive datasets of code examples for training AI models on quality, bugs, vulnerabilities, and design patterns.', 'agent_triggers': ['github repository analysis tool', 'developers', 'repo links', 'chatbot type q/a', 'insights', 'code quality', 'bugs', 'vulnerabilities', 'code structure', 'design patterns used'], 'complexity_level': 'High', 'raw_query': 'github repository analysis tool for developers where user can give repo links and the tool will will have a chatbot type q/a and give insights about code quality, bugs, vulnerabilities, code structure, design patterns used, etc.'}


    planner = DynamicTaskPlanner(use_llm=True)
    result = planner.plan({"intent": intent_example})
    print(json.dumps(result["task_plan"], indent=2))
