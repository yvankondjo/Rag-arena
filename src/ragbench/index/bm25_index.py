from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bm25s
from langdetect import detect
from stop_words import get_stop_words
import tiktoken


def json_safe(obj):
    """Convertit récursivement en JSON-safe avec clés string."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, set):
        return [json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    # numpy scalars éventuels
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


class BM25Index:
    """
    - Corpus = list[dict] JSON-serializable: {"text": "...", "metadata": {...}}
      => bm25s can save/load it with retriever.save/load(load_corpus=True).
    - Tokenizer = tiktoken (splitter returns list[str] tokens)
    - Persistence:
        - BM25 index saved in save_dir
        - corpus (including metadata) saved in save_dir by bm25s
    """
    def __init__(self, bm25_index_save_dir: str, encoding_name: str = "o200k_base"):
        self.save_dir = Path(bm25_index_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.encoding = tiktoken.get_encoding(encoding_name)

        self.retriever: Optional[bm25s.BM25] = None
        self._metadata_cache: Optional[List[Dict[str, Any]]] = None

        self._try_load_retriever()


    def add_corpus(self, corpus: List[str], corpus_json: List[Dict[str, Any]]) -> bool:
        """
        Build/rebuild the index (persistent).
        """
        if len(corpus) != len(corpus_json):
            raise ValueError("corpus and corpus_json must have the same length")

        corpus_clean = [self._clean_text(doc) for doc in corpus]

        corpus_tokens = [self._tiktoken_splitter(doc) for doc in corpus_clean]

        try:

            safe_corpus_json = [json_safe(doc) for doc in corpus_json]
            print(f"[BM25Index] Sanitized {len(safe_corpus_json)} documents")

           
            retriever = bm25s.BM25(corpus=safe_corpus_json)
            retriever.index(corpus_tokens)

            retriever.save(str(self.save_dir))

            self.retriever = retriever
            self._metadata_cache = safe_corpus_json

            print(f"[BM25Index] Successfully indexed {len(corpus)} documents")
            return True

        except Exception as e:
            import traceback
            print(f"[BM25Index] Error creating BM25 index: {e}")
            print(f"[BM25Index] Traceback: {traceback.format_exc()}")
            return False


    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if top_k <= 0:
            return {"context": "", "sources": [], "ids": [], "documents": [], "metadatas": []}

        if self.retriever is None:
            self._try_load_retriever()
        if self.retriever is None:
            return {"context": "", "sources": [], "ids": [], "documents": [], "metadatas": []}

        q_clean = self._clean_text(query)  
        q_tokens = [self._tiktoken_splitter(q_clean)]

        
        docs, scores = self.retriever.retrieve(q_tokens, k=top_k)

        context_parts: List[str] = []
        sources: List[str] = []
        documents: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []

       
        for rank, (doc, score) in enumerate(zip(docs[0], scores[0]), start=1):
            chunk_text = doc.get("text", "")
            md = doc.get("metadata", {}) or {}

            context_parts.append(f"[Source {rank}] (score={float(score):.4f})\n{chunk_text}\n")
            sources.append(str(md.get("document_id", md.get("source", "unknown"))))
            ids.append(str(md.get("chunk_id", md.get("id", "unknown"))))
            documents.append(chunk_text)
            metadatas.append(md)

        # Extract scores for each document
        scores_list = [float(score) for score in scores[0]]

        return {
            "context": "\n".join(context_parts) if context_parts else "",
            "sources": sources,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "scores": scores_list,  # Add actual BM25 scores
        }

    def _tiktoken_splitter(self, text: str) -> List[str]:
        text = text or ""
        ids = self.encoding.encode(text, disallowed_special=())
        return [self.encoding.decode([tid]) for tid in ids]

    def _try_load_retriever(self) -> None:
        try:
            self.retriever = bm25s.BM25.load(str(self.save_dir), load_corpus=True)
        except Exception:
            self.retriever = None

    def _clean_text(self, text: str) -> str:
        """
        Must be IDENTICAL between corpus and query.
        """
        text = text or ""
        text = self._remove_multilingual_stopwords(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _remove_multilingual_stopwords(self, text: str) -> str:
        try:
            lang = detect(text)
        except Exception:
            return text

        try:
            stop_words = set(get_stop_words(lang))
        except Exception:
            return text

        words = text.split()
        filtered = [w for w in words if w.lower() not in stop_words]
        return " ".join(filtered)
