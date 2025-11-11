"""
retriever_selector.py
---------------------
Selects and combines retrievers (dense + sparse) for hybrid search.
"""

import os
from loguru import logger
from typing import List

from core.rag_manager import VectorStoreManager
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class RetrieverSelector:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.vector_manager = VectorStoreManager(persist_directory=persist_directory)
        self.db = self.vector_manager._get_db()

        logger.info("Initializing dense retriever from Chroma...")
        self.dense_retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # Optional sparse retriever (BM25)
        try:
            docs = []
            chroma_docs = self.db.get(include=["documents", "metadatas"])
            for text, meta in zip(chroma_docs["documents"], chroma_docs["metadatas"]):
                docs.append(Document(page_content=text, metadata=meta))
            self.sparse_retriever = BM25Retriever.from_documents(docs)
            logger.info("BM25 sparse retriever initialized successfully.")
        except Exception as e:
            logger.warning(f"Sparse retriever init failed: {e}")
            self.sparse_retriever = None

        logger.info("Hybrid retrieval will combine dense and sparse results.")

    def retrieve(self, query: str) -> List[Document]:
        logger.info(f"🔍 Retrieving for query: {query}")
        results = []
        if self.sparse_retriever:
            dense_results = self.dense_retriever.invoke(query)
            sparse_results = self.sparse_retriever.invoke(query)
            # Combine results: you can do smart merging or just concatenate
            results = list({doc.page_content: doc for doc in dense_results + sparse_results}.values()) # deduplicate by content
            logger.info(f"Found {len(results)} candidate chunks (hybrid).")
        else:
            results = self.dense_retriever.invoke(query)
            logger.info(f"Found {len(results)} candidate chunks (dense only).")
        return results



if __name__ == "__main__":
    retriever = RetrieverSelector()
    results = retriever.retrieve("what are the competitors for my tool which github repository analysis tool for developers where user can give repo links and the tool will will have a chatbot type q/a and give insights about code quality, bugs, vulnerabilities, code structure, design patterns used, etc. ")
    for r in results:
        print(f"- {r.page_content[:150]}...")
