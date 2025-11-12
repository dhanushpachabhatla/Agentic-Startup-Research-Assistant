"""
config.py
---------
Central configuration manager for the Agentic Startup Research Assistant.

Loads environment variables, manages API keys, LLM defaults,
storage paths, and logging.
"""

import os
from dotenv import load_dotenv
from loguru import logger


class Config:
    """Central configuration manager for the entire system."""

    def __init__(self, env_path: str = None):
        # === Load .env ===
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.env_path = env_path or os.path.join(base_dir, "..", ".env")
        load_dotenv(self.env_path)

        # === API Keys ===
        self.GEMINI_API_KEY1 = os.getenv("GEMINI_API_KEY1")
        self.GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2")
        self.GEMINI_API_KEY3 = os.getenv("GEMINI_API_KEY3")
        self.GEMINI_API_KEY4 = os.getenv("GEMINI_API_KEY4")
        self.GEMINI_API_KEY5 = os.getenv("GEMINI_API_KEY5")
        self.GEMINI_API_KEY6 = os.getenv("GEMINI_API_KEY6")
        self.GEMINI_API_KEY7 = os.getenv("GEMINI_API_KEY7")
        self.GEMINI_API_KEY8 = os.getenv("GEMINI_API_KEY8")
        self.GEMINI_API_KEY9 = os.getenv("GEMINI_API_KEY9")
        self.GEMINI_API_KEY10 = os.getenv("GEMINI_API_KEY10")
        self.GROK_API_KEY = os.getenv("GROK_API_KEY")
        self.SERPAPI_KEY = os.getenv("SERPAPI_KEY")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        # === LangSmith Tracing ===
        self.LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
        self.LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
        self.LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agentic-startup-assistant")

        # === Models & Parameters ===
        self.MODEL_CONFIG = {
            "primary_llm": "gemini-2.5-flash",
            "fallback_llm": "gemini-1.5-flash",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        # === Paths ===
        self.BASE_DIR = os.path.abspath(os.path.join(base_dir, ".."))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.RAW_DOCS = os.path.join(self.DATA_DIR, "raw_docs")
        self.EMBEDDINGS = os.path.join(self.DATA_DIR, "embeddings")
        self.MEMORY_STORE = os.path.join(self.DATA_DIR, "memory_store")

        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.RAW_DOCS, exist_ok=True)
        os.makedirs(self.MEMORY_STORE, exist_ok=True)

        # === Logging ===
        logger.add(
            os.path.join(self.LOG_DIR, "system.log"),
            rotation="2 MB",
            retention="7 days",
            level="INFO",
        )
        logger.info("✅ Configuration initialized successfully.")

    # === Utility Methods ===
    def get_api_key(self, provider: str) -> str:
        """Retrieve an API key dynamically by provider name (case-insensitive)."""
        return getattr(self, f"{provider.upper()}_API_KEY", None)

    def summary(self) -> dict:
        """Return a safe summary for debugging."""
        active_keys = [
            k for k in ["OPENAI_API_KEY", "GEMINI_API_KEY1", "GROK_API_KEY", "SERPAPI_KEY"]
            if getattr(self, k)
        ]
        return {
            "Active Keys": active_keys,
            "Primary Model": self.MODEL_CONFIG["primary_llm"],
            "Tracing Enabled": self.LANGSMITH_TRACING,
            "Log Directory": self.LOG_DIR,
        }

    def reload(self):
        """Reload environment variables (e.g., if .env updated)."""
        load_dotenv(self.env_path, override=True)
        logger.info("🔁 Environment variables reloaded.")


# === Singleton instance for global use ===
config = Config()

# Optional: print a short summary when run standalone
if __name__ == "__main__":
    from pprint import pprint
    pprint(config.summary())
