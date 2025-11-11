# core/index_strategic_knowledge.py
"""
Indexes high-level strategic knowledge (agent summaries, strategy report, final report)
into the existing Chroma vector store — without clearing prior data.
"""

import json
from pathlib import Path
from loguru import logger
from langchain_core.documents import Document
from core.rag_manager import VectorStoreManager

# ===============================================================
# 📁 File paths
# ===============================================================
DATA_DIR = Path("data/memory_store")

FILES_TO_INDEX = [
    DATA_DIR / "agent_summaries.json",
    DATA_DIR / "strategy_report.json",
    DATA_DIR / "final_report.json",
]

# ===============================================================
# 🧩 Helper Functions
# ===============================================================
def _normalize_to_text(item) -> str:
    """Flatten any Python object to clean string content for Document embedding."""
    if isinstance(item, str):
        return item.strip()
    elif isinstance(item, dict):
        # Prefer specific text fields if available
        for key in ["summary", "text", "content", "output", "description", "body"]:
            if key in item and isinstance(item[key], str):
                return item[key]
        # Otherwise flatten the whole dict
        return json.dumps(item, ensure_ascii=False)
    elif isinstance(item, list):
        return "\n".join(_normalize_to_text(x) for x in item)
    else:
        return str(item)


def load_texts():
    """Load and normalize multiple JSON/text files into LangChain Documents."""
    docs = []
    for path in FILES_TO_INDEX:
        if not path.exists():
            logger.warning(f"⚠️ {path} missing — skipping.")
            continue

        try:
            logger.info(f"📄 Loading {path.name}...")
            raw = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.suffix == ".json"
                else path.read_text(encoding="utf-8")
            )

            if isinstance(raw, list):
                for entry in raw:
                    text = _normalize_to_text(entry)
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": path.name, "category": "strategy"}))
            elif isinstance(raw, dict):
                text = _normalize_to_text(raw)
                docs.append(Document(page_content=text, metadata={"source": path.name, "category": "strategy"}))
            elif isinstance(raw, str):
                docs.append(Document(page_content=raw, metadata={"source": path.name, "category": "strategy"}))
            else:
                docs.append(Document(page_content=json.dumps(raw, ensure_ascii=False),
                                     metadata={"source": path.name, "category": "strategy"}))

        except Exception as e:
            logger.error(f"❌ Failed to parse {path}: {e}")

    logger.info(f"✅ Loaded {len(docs)} strategic documents for indexing.")
    return docs


# ===============================================================
# 🚀 Indexing Entry Point
# ===============================================================
if __name__ == "__main__":
    logger.info("📚 Indexing strategic knowledge (agent summaries + strategy reports)...")

    docs = load_texts()
    if not docs:
        logger.warning("No strategic documents found. Exiting.")
        exit()

    # Initialize vector store manager (DO NOT clear store)
    manager = VectorStoreManager()
    db = manager._get_db()

    # Deduplication logic based on metadata (source)
    existing_sources = set()
    try:
        # if retriever supports list collection metadata
        if hasattr(db, "get") and callable(getattr(db, "get", None)):
            collection = db.get(include=["metadatas"])
            if "metadatas" in collection:
                for meta in collection["metadatas"]:
                    if isinstance(meta, dict) and meta.get("source"):
                        existing_sources.add(meta["source"])
    except Exception as e:
        logger.warning(f"Could not check existing sources: {e}")

    new_docs = [d for d in docs if d.metadata.get("source") not in existing_sources]
    logger.info(f"🧩 {len(new_docs)} new strategic docs to add (skipped {len(docs) - len(new_docs)} already indexed).")

    if not new_docs:
        logger.info("✅ No new documents to add. Index already up to date.")
    else:
        manager.add_documents(new_docs)
        logger.info("✅ Strategic knowledge successfully merged into existing vector store (no data loss).")
