"""Build indexes (Chroma + BM25) from BEIR dataset (doc-level)."""

import json
import logging
from pathlib import Path
from typing import List, Dict

from ragbench.beir_download import BEIRDownloader, load_beir_dataset
from ragbench.index.chromadb_index import ChromaDBIndex
from ragbench.index.bm25_index import BM25Index
from ragbench.config import AppConfig
import tiktoken

logger = logging.getLogger(__name__)


def truncate_text_to_tokens(text: str, max_tokens: int = 8000, encoding_name: str = "cl100k_base") -> str:
    """Truncate text to maximum token count using tiktoken.

    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        encoding_name: tiktoken encoding name

    Returns:
        Truncated text
    """
    if not text:
        return text

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text


    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)


    return truncated_text


def build_indexes(
    dataset_name: str = "scifact",
):
    """Build Chroma + BM25 indexes for a dataset (doc-level).

    Args:
        dataset_name: BEIR dataset (e.g., 'scifact')
    """
    config = AppConfig()

    logger.info(f"🔽 Checking/downloading {dataset_name}...")
    downloader = BEIRDownloader(config.raw_data_dir)
    dataset_path = downloader.download_dataset(dataset_name)

    logger.info("📖 Loading corpus, queries, qrels...")
    corpus, queries, qrels = load_beir_dataset(dataset_path)
    
    logger.info("📄 Preparing documents for indexing...")
    all_docs: List[Dict] = []

    for doc_id, doc in corpus.items():
        text = f"{doc.get('title', '')}\n\n{doc.get('text', '')}".strip()

        original_text = text
        text = truncate_text_to_tokens(text, max_tokens=8000, encoding_name="cl100k_base")

        if len(text) < len(original_text):
            logger.debug(f"Truncated document {doc_id} from {len(original_text)} to {len(text)} chars")

        all_docs.append({
            "id": doc_id,
            "text": text,
            "metadata": {
                "document_id": doc_id,
                "title": doc.get("title", ""),
            }
        })

    logger.info(f"Prepared {len(all_docs)} documents (truncated to max 8000 tokens each)")
    
    logger.info("🔍 Building Chroma index...")
    collection_name = f"{dataset_name}_doclevel"

    chroma_index = ChromaDBIndex(
        collection_name=collection_name,
        persist_directory=str(config.persist_directory),
        embedding_model=config.embedding_model,
    )

    doc_texts = [doc["text"] for doc in all_docs]
    doc_ids = [doc["id"] for doc in all_docs]
    doc_metas = [doc["metadata"] for doc in all_docs]


    batch_size = 50  # Small batches for OpenAI limits
    total_added = 0

    logger.info(f"Adding {len(doc_texts)} documents to Chroma in batches of {batch_size}...")
    try:
        for i in range(0, len(doc_texts), batch_size):
            batch_end = min(i + batch_size, len(doc_texts))
            batch_ids = doc_ids[i:batch_end]
            batch_texts = doc_texts[i:batch_end]
            batch_metas = doc_metas[i:batch_end]

            chroma_index.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            total_added += len(batch_ids)
            logger.info(f"  Added {total_added}/{len(all_docs)} documents")

        logger.info(f"✓ Chroma index created with {len(all_docs)} documents")
    except Exception as e:
        logger.error(f"Failed to create Chroma index: {e}")
        return None
    
    logger.info("📑 Building BM25 index...")
    bm25_dir = config.bm25_index_dir / f"{dataset_name}_doclevel"
    bm25_index = BM25Index(str(bm25_dir))
    
    corpus_for_bm25 = [
        {"text": doc["text"], "metadata": doc["metadata"]}
        for doc in all_docs
    ]
    
    if bm25_index.add_corpus(doc_texts, corpus_for_bm25):
        logger.info(f"✓ BM25 index created at {bm25_dir}")
    else:
        logger.error("Failed to create BM25 index")
    
    metadata_path = config.processed_data_dir / f"{dataset_name}_doclevel_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "indexing": "doclevel",
            "num_documents": len(corpus),
            "num_queries": len(queries),
            "qrels": qrels,
        }, f, indent=2)
    
    logger.info(f"✓ Metadata saved to {metadata_path}")
    
    return {
        "chroma_index": chroma_index,
        "bm25_index": bm25_index,
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "documents": all_docs,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dataset = sys.argv[1] if len(sys.argv) > 1 else "scifact"

    build_indexes(dataset_name=dataset)
