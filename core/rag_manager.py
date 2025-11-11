import os
import shutil
import json
import time
import asyncio
from loguru import logger
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from app.config import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from google import genai 


load_dotenv()
if getattr(config, "GEMINI_API_KEY3", None):
    os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY3


class Embedder:
    def __init__(self):
        self.model_name = "text-embedding-004"
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_4"))

    async def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[List[List[float]]]:
        if not texts:
            return []
        retries = 3
        for attempt in range(retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=genai.types.EmbedContentConfig(
                        output_dimensionality=3072,
                        task_type=task_type
                    )
                )
                return [e.values for e in result.embeddings]
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < retries - 1:
                    logger.warning(f"[Gemini Embedder] Rate limited. Retrying in 10s (attempt {attempt+1})...")
                    await asyncio.sleep(10)
                    continue
                logger.error(f"[Gemini Embedder] Embedding error: {e}")
                return None


class GeminiEmbeddingWrapper:
    """Wraps the async Embedder into a LangChain-compatible embedding interface."""

    def __init__(self, embedder: "Embedder"):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds multiple documents."""
        return asyncio.run(self.embedder.embed_texts(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query."""
        result = asyncio.run(self.embedder.embed_texts([text]))
        return result[0] if result else []


class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedder = Embedder()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        self.db = None

    def _get_db(self):
        """Initialize the Chroma client."""
        if self.db is None:
            logger.info(f"Initializing vector store client at: {self.persist_directory}")
            embedding_wrapper = GeminiEmbeddingWrapper(self.embedder)
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding_wrapper
            )
        return self.db


    def _embedding_function(self, texts: List[str]) -> List[List[float]]:
        """Sync wrapper around async embedder call."""
        try:
            return asyncio.run(self.embedder.embed_texts(texts))
        except Exception as e:
            logger.error(f"Embedding function failed: {e}")
            return [[] for _ in texts]

    def add_documents(self, documents: List[Document]):
        """Chunk and embed documents, store in Chroma."""
        if not documents:
            logger.warning("No documents provided to add.")
            return

        logger.info(f"Received {len(documents)} raw documents. Starting chunking...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        try:
            db = self._get_db()
            logger.info(f"Adding {len(chunks)} chunks to vector store one-by-one...")

            # for i, chunk in enumerate(chunks):
            #     chunk.metadata = self._sanitize_metadata(chunk.metadata)
            #     db.add_documents([chunk])
            #     logger.info(f"  > Added chunk {i+1}/{len(chunks)}")
            #     # time.sleep(1.2)  # respect rate limits

            BATCH_SIZE = 10
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                for c in batch:
                    c.metadata = self._sanitize_metadata(c.metadata)
                db.add_documents(batch)
                logger.info(f"  > Added chunks {i+1}–{i+len(batch)}/{len(chunks)}")
                time.sleep(1.2)  # respect rate limits



            logger.info(f"✅ Successfully added all {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"❌ Failed to add documents to Chroma: {e}")
            raise e

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant chunks."""
        logger.info(f"Searching for: '{query}'...")
        db = self._get_db()
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        results = retriever.invoke(query)
        logger.info(f"Found {len(results)} relevant chunks.")
        return results

    def clear_store(self):
        """Wipes the vector store."""
        logger.warning(f"Clearing vector store at: {self.persist_directory}")
        self.db = None
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            logger.info("Vector store directory deleted.")
        logger.info("Vector store cleared.")


    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all metadata values are flat (str, int, float, bool, None)."""
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean_meta[k] = v
            elif isinstance(v, list):
                # Convert list to comma-separated string
                clean_meta[k] = ", ".join(map(str, v))
            elif isinstance(v, dict):
                # Convert dict to a compact string
                clean_meta[k] = json.dumps(v, ensure_ascii=False)
            else:
                # Drop or stringify unknown types
                clean_meta[k] = str(v)
        return clean_meta


if __name__ == "__main__":
    json_input_file = "data/raw_docs/raw_docs.json"
    documents_to_add = []

    try:
        with open(json_input_file, "r", encoding="utf-8") as f:
            docs_from_json = json.load(f)
        logger.info(f"Loaded {len(docs_from_json)} documents from {json_input_file}")

        for doc_dict in docs_from_json:
            documents_to_add.append(
                Document(
                    page_content=doc_dict.get("page_content", ""),
                    metadata=doc_dict.get("metadata", {})
                )
            )
    except FileNotFoundError:
        logger.error(f"'{json_input_file}' not found. Run the orchestrator first.")
        exit()
    except Exception as e:
        logger.error(f"Error loading/parsing {json_input_file}: {e}")
        exit()

    manager = VectorStoreManager()
    manager.clear_store()
    manager.add_documents(documents_to_add)

    print("\n--- 1. Search for 'SonarQube features' ---")
    results1 = manager.search("What are the key features and pricing for SonarQube?")
    for doc in results1:
        print(f"  > CHUNK: {doc.page_content[:]}...")
        print(f"  > SOURCE: {doc.metadata.get('source', 'N/A')}\n")

    print("--- 2. Search for 'developer pain points' ---")
    results2 = manager.search("What pain points do developers have?")
    for doc in results2:
        print(f"  > CHUNK: {doc.page_content[:]}...")
        print(f"  > SOURCE: {doc.metadata.get('source', 'N/A')}\n")
