"""
intent_parser.py - python -m core.intent_parser
----------------
Extracts structured intent and contextual metadata from user startup ideas or queries.
Acts as the first step in the agentic pipeline before Dynamic Task Planner.
"""
import os
import re
from typing import Dict, Any
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import config


class IntentParser:
    """
    Intent Parser Agent
    -------------------
    Extracts:
      - domain / industry
      - target audience
      - business model
      - tech keywords
      - competitor mentions
      - intent type (idea, compare, trend, tech, etc.)
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        if use_llm and config.GEMINI_API_KEY1:
            try:
                os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY1
                self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
            except Exception as e:
                logger.warning(f" LLM initialization failed: {e}. Falling back to regex parser.")
                self.use_llm = False

        # Define the fallback keywords for rule-based extraction
        self.domains = [
            "health", "education", "finance", "travel", "food", "fitness", "pet", "real estate",
            "transportation", "ai", "mental health", "agriculture", "gaming", "retail", "sustainability"
        ]
        self.tech_terms = [
            "AI", "machine learning", "blockchain", "NLP", "data analytics", "AR", "VR",
            "IoT", "chatbot", "LLM", "neural network"
        ]
        self.intent_patterns = {
            "compare": r"(compare|vs|difference|better than)",
            "trend": r"(trend|market|growth|statistics)",
            "tech": r"(technology|innovation|research|paper)",
            "idea": r"(idea|startup|launch|build)",
        }

    # ------------------------
    # Public Method
    # ------------------------
    def parse(self, user_input: str) -> Dict[str, Any]:
        logger.info(f" Parsing intent for: '{user_input}'")

        if self.use_llm and self.llm:
            return self._parse_with_llm(user_input)
        else:
            return self._parse_with_rules(user_input)

    # ------------------------
    # LLM-based Semantic Parsing
    # ------------------------
    def _parse_with_llm(self, text: str) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["query"],
            # template=(
            #     "You are an intelligent intent parser for a Startup Research Assistant.\n"
            #     "Extract key structured data from the user query below.\n\n"
            #     "User query: {query}\n\n"
            #     "Return a JSON with the following fields:\n"
            #     "industry, business_model, target_audience, tech_keywords, competitor_names, intent_type."
            # ),
            template = (
                    "You are an intelligent intent parser for a Startup Research Assistant.\n"
                    "Extract structured insights from the user query below.\n\n"
                    "User query: {query}\n\n"
                    "Return a JSON with the following fields:\n"
                    "industry, business_model, target_audience, tech_keywords, competitor_names, intent_type,\n"
                    "problem_statement, solution_summary, data_needs, agent_triggers, complexity_level."
                )

        )
        try:
            response = self.llm.invoke(prompt.format_prompt(query=text).to_string())
            parsed = self._safe_extract_json(response.content)
            parsed["raw_query"] = text
            logger.success(" LLM-based intent parsed successfully.")
            return parsed
        except Exception as e:
            logger.warning(f" LLM parsing failed ({e}), falling back to rule-based.")
            return self._parse_with_rules(text)

    # ------------------------
    # Rule-based Fallback Parser
    # ------------------------
    def _parse_with_rules(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        extracted_domain = next((d for d in self.domains if d in text_lower), "general")
        extracted_tech = [t for t in self.tech_terms if t.lower() in text_lower]
        extracted_intent = next(
            (intent for intent, pattern in self.intent_patterns.items() if re.search(pattern, text_lower)),
            "idea",
        )

        competitors = re.findall(r"[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?", text)
        if len(competitors) <= 1:
            competitors = []

        result = {
            "industry": extracted_domain,
            "business_model": self._infer_business_model(text_lower),
            "target_audience": self._infer_audience(text_lower),
            "tech_keywords": extracted_tech,
            "competitor_names": competitors,
            "intent_type": extracted_intent,
            "raw_query": text,
        }

        logger.success(" Rule-based intent parsed successfully.")
        return result

    # ------------------------
    # Helper Methods
    # ------------------------
    def _infer_business_model(self, text: str) -> str:
        if "platform" in text:
            return "Platform"
        if "app" in text:
            return "Mobile App"
        if "service" in text:
            return "Service"
        if "tool" in text or "software" in text:
            return "SaaS"
        if "marketplace" in text:
            return "Marketplace"
        return "General"

    def _infer_audience(self, text: str) -> str:
        if "student" in text:
            return "Students"
        if "business" in text or "startup" in text:
            return "Businesses"
        if "pet" in text:
            return "Pet Owners"
        if "doctor" in text or "patient" in text:
            return "Healthcare Users"
        return "General Audience"

    def _safe_extract_json(self, text: str) -> Dict[str, Any]:
        import json, re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"industry": None, "business_model": None, "intent_type": "idea"}


# For standalone testing
if __name__ == "__main__":
    parser = IntentParser()
    sample_queries = "github repository analysis tool for developers where user can give repo links and the tool will will have a chatbot type q/a and give insights about code quality, bugs, vulnerabilities, code structure, design patterns used, etc."
    print("\nInput:", sample_queries)
    print(parser.parse(sample_queries))
