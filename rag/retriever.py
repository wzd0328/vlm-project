import json
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List


@dataclass
class RAGDocument:
    """A compact knowledge item for multimodal RAG."""

    doc_id: str
    caption: str = ""
    ocr: str = ""
    source: str = ""
    image_path: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = [self.caption, self.ocr]
        for key in ("tags", "objects", "summary", "content"):
            value = self.metadata.get(key)
            if value:
                parts.append(str(value))
        return "\n".join(part for part in parts if part).strip()


class TextRAGRetriever:
    """Dependency-free TF-IDF retriever for caption/OCR/document chunks.

    It intentionally avoids FAISS/sklearn so the RAG demo and smoke tests can
    run in a minimal environment. The public API is small, making it easy to
    replace with a dense SigLIP/CLIP retriever later.
    """

    INDEX_FILE = "text_rag_index.pkl"
    DOCS_FILE = "documents.jsonl"

    def __init__(self, ngram_range=(1, 2), max_features: int = 50000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.documents: List[RAGDocument] = []
        self.vocab: List[str] = []
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []

    @staticmethod
    def _tokens(text: str) -> List[str]:
        # English words/numbers + individual CJK characters for Chinese queries.
        return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", (text or "").lower())

    def _terms(self, text: str) -> List[str]:
        tokens = self._tokens(text)
        terms: List[str] = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            if n <= 0:
                continue
            for i in range(0, max(len(tokens) - n + 1, 0)):
                terms.append(" ".join(tokens[i : i + n]))
        return terms

    @staticmethod
    def _norm(vector: Dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def _vectorize(self, text: str) -> Dict[str, float]:
        counts = Counter(term for term in self._terms(text) if term in self.idf)
        if not counts:
            return {}
        total = sum(counts.values())
        return {term: (count / total) * self.idf[term] for term, count in counts.items()}

    def fit(self, documents: Iterable[RAGDocument]):
        self.documents = [doc for doc in documents if doc.text]
        if not self.documents:
            raise ValueError("No non-empty RAG documents were provided.")

        doc_term_sets = []
        term_frequency = Counter()
        for doc in self.documents:
            terms = self._terms(doc.text)
            term_frequency.update(terms)
            doc_term_sets.append(set(terms))

        self.vocab = [term for term, _ in term_frequency.most_common(self.max_features)]
        vocab_set = set(self.vocab)
        doc_freq = Counter()
        for term_set in doc_term_sets:
            doc_freq.update(term for term in term_set if term in vocab_set)

        num_docs = len(self.documents)
        self.idf = {term: math.log((1 + num_docs) / (1 + doc_freq[term])) + 1 for term in self.vocab}
        self.doc_vectors = [self._vectorize(doc.text) for doc in self.documents]
        self.doc_norms = [self._norm(vector) for vector in self.doc_vectors]
        return self

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> List[Dict]:
        if not self.documents or not self.idf:
            raise RuntimeError("Retriever is not fitted. Call fit() or load() first.")
        query_vector = self._vectorize(query)
        query_norm = self._norm(query_vector)
        if not query_vector or query_norm == 0:
            return []

        scored = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            doc_norm = self.doc_norms[idx]
            if doc_norm == 0:
                continue
            dot = sum(value * doc_vector.get(term, 0.0) for term, value in query_vector.items())
            score = dot / (query_norm * doc_norm)
            if score >= min_score:
                scored.append((score, idx))

        results = []
        for score, idx in sorted(scored, reverse=True)[: max(top_k, 1)]:
            doc = self.documents[idx]
            payload = asdict(doc)
            payload["score"] = float(score)
            results.append(payload)
        return results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, self.INDEX_FILE), "wb") as f:
            pickle.dump(
                {
                    "ngram_range": self.ngram_range,
                    "max_features": self.max_features,
                    "vocab": self.vocab,
                    "idf": self.idf,
                    "doc_vectors": self.doc_vectors,
                    "doc_norms": self.doc_norms,
                },
                f,
            )
        with open(os.path.join(index_dir, self.DOCS_FILE), "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: str):
        with open(os.path.join(index_dir, cls.INDEX_FILE), "rb") as f:
            payload = pickle.load(f)
        retriever = cls(
            ngram_range=payload.get("ngram_range", (1, 2)),
            max_features=payload.get("max_features", 50000),
        )
        retriever.vocab = payload["vocab"]
        retriever.idf = payload["idf"]
        retriever.doc_vectors = payload["doc_vectors"]
        retriever.doc_norms = payload["doc_norms"]
        docs_path = os.path.join(index_dir, cls.DOCS_FILE)
        with open(docs_path, "r", encoding="utf-8") as f:
            retriever.documents = [RAGDocument(**json.loads(line)) for line in f if line.strip()]
        return retriever


def load_jsonl_documents(path: str) -> List[RAGDocument]:
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or row.get("id") or f"row_{line_no}")
            metadata = row.get("metadata") or {}
            for key in ("tags", "objects", "summary", "content"):
                if key in row and key not in metadata:
                    metadata[key] = row[key]
            documents.append(
                RAGDocument(
                    doc_id=doc_id,
                    caption=str(row.get("caption", "")),
                    ocr=str(row.get("ocr", "")),
                    source=str(row.get("source", "")),
                    image_path=str(row.get("image_path", "")),
                    metadata=metadata,
                )
            )
    return documents


def dump_retrieval_results(results: List[Dict]) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)
