"""Lightweight multimodal RAG utilities for MiniMind-V."""

from .prompt_builder import build_rag_prompt, format_retrieval_context
from .retriever import RAGDocument, TextRAGRetriever

__all__ = [
    "RAGDocument",
    "TextRAGRetriever",
    "build_rag_prompt",
    "format_retrieval_context",
]
