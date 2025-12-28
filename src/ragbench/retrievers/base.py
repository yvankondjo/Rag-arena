"""Retrieval data structures."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""

    chunk_id: str
    text: str
    score: float
    document_id: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "document_id": self.document_id,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Result from retrieval."""

    chunks: list[RetrievedChunk]
    query: str
    retriever_type: str
    latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0  # Separate reranking latency for fair comparison

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalResult":
        """Create RetrievalResult from dictionary.
        
        Args:
            data: Dictionary with retrieval result data
            
        Returns:
            RetrievalResult instance
        """

        chunks = []
        documents = data.get("documents", [])
        ids = data.get("ids", [])
        scores = data.get("scores", [])
        metadatas = data.get("metadatas", [])
        
        for i, (doc, chunk_id) in enumerate(zip(documents, ids)):
            score = scores[i] if i < len(scores) else 0.0
            metadata = metadatas[i] if i < len(metadatas) else {}
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=doc,
                score=score,
                document_id=metadata.get("document_id", chunk_id),
                metadata=metadata,
            )
            chunks.append(chunk)
        
        return cls(
            chunks=chunks,
            query=data.get("query", ""),
            retriever_type=data.get("retriever_type", "unknown"),
            latency_ms=data.get("latency_ms", 0.0),
        )

    @property
    def context(self) -> str:
        """Concatenated chunk texts."""
        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            parts.append(f"[Chunk {i}] (score={chunk.score:.4f})\n{chunk.text}")
        return "\n\n".join(parts)

    @property
    def sources(self) -> list[str]:
        """Unique document sources."""
        return list(dict.fromkeys(chunk.document_id for chunk in self.chunks))

    @property
    def ids(self) -> list[str]:
        """Chunk IDs."""
        return [chunk.chunk_id for chunk in self.chunks]

    @property
    def documents(self) -> list[str]:
        """Chunk texts."""
        return [chunk.text for chunk in self.chunks]

    @property
    def scores(self) -> list[float]:
        """Chunk scores."""
        return [chunk.score for chunk in self.chunks]

    @property
    def metadatas(self) -> list[dict]:
        """Chunk metadatas."""
        return [chunk.metadata for chunk in self.chunks]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "context": self.context,
            "sources": self.sources,
            "ids": self.ids,
            "documents": self.documents,
            "scores": self.scores,
            "metadatas": self.metadatas,
            "retriever_type": self.retriever_type,
            "latency_ms": self.latency_ms,
        }
