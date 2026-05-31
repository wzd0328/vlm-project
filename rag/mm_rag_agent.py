from typing import Dict, List, Optional

from .prompt_builder import build_rag_prompt
from .retriever import TextRAGRetriever


class MiniMindRAGAgent:
    """Rule-light RAG orchestrator for MiniMind-V inference scripts."""

    def __init__(self, retriever: TextRAGRetriever, top_k: int = 4, language: str = "zh", min_score: float = 1e-6):
        self.retriever = retriever
        self.top_k = top_k
        self.language = language
        self.min_score = min_score

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict]:
        return self.retriever.search(question, top_k=top_k or self.top_k, min_score=self.min_score)

    def build_prompt(self, question: str, image_token: str = "<image>", top_k: Optional[int] = None):
        results = self.retrieve(question, top_k=top_k)
        prompt = build_rag_prompt(
            question=question,
            retrieval_results=results,
            image_token=image_token,
            language=self.language,
        )
        return prompt, results
