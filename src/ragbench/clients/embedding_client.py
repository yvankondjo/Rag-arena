"""Embedding client for query embeddings only."""

from typing import Optional, List
from chromadb.utils import embedding_functions


class EmbeddingClient:
    """OpenAI embedding client for query embedding only.

    Note: ChromaDB handles document embedding in batch during indexing.
    This client is only used for embedding user queries.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        """Initialize embedding client.

        Args:
            api_key: OpenAI API key
            model: Embedding model identifier
        """
        self.model = model
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model,
        )

    def embed_single(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embedding_fn([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embedding_fn(texts)


def create_embedding_client(
    api_key: str,
    model: str = "text-embedding-3-small",
) -> EmbeddingClient:
    """Factory function to create embedding client.

    Args:
        api_key: OpenAI API key
        model: Embedding model identifier

    Returns:
        EmbeddingClient instance
    """
    return EmbeddingClient(api_key=api_key, model=model)
