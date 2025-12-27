"""Client abstractions for LLM and embedding providers."""

from ragbench.clients.embedding_client import EmbeddingClient, create_embedding_client

__all__ = [
    "EmbeddingClient",
    "create_embedding_client",
]
