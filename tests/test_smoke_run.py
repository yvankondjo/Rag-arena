"""Quick test: download + chunk + index + run 1 config"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_pipeline():
    """Smoke test: end-to-end pipeline."""
    from ragbench.build_indexes import build_indexes
    from ragbench.run_experiment import RAGConfig, run_single_config
    from ragbench.beir_download import load_beir_queries, load_beir_qrels
    import asyncio
    
    logger.info("="*60)
    logger.info("🧪 SMOKE TEST: Full RAG Pipeline")
    logger.info("="*60)
    
    # Step 1: Build indexes (doc-level)
    logger.info("\n1️⃣ Building indexes...")
    result = build_indexes(
        dataset_name="scifact",
        limit_documents=10,  # Tiny corpus for test
        limit_queries=10,
    )

    logger.info(f"✓ Indexed {len(result['documents'])} documents")
    
    # Step 2: Run single config
    logger.info("\n2️⃣ Running single config: simple + dense...")
    
    config = RAGConfig(
        orchestration_mode="simple",
        retrieval_mode="dense",
    )
    
    from ragbench.config import AppConfig
    app_config = AppConfig()
    
    queries = load_beir_queries(
        app_config.raw_data_dir / "scifact" / "queries.jsonl"
    )
    queries = dict(list(queries.items())[:5])  # 5 queries
    
    qrels = load_beir_qrels(
        app_config.raw_data_dir / "scifact" / "qrels" / "test.tsv"
    )
    
    results_dir = app_config.results_dir
    
    result = asyncio.run(
        run_single_config(
            config,
            queries,
            qrels,
            asyncio.Semaphore(1),
            results_dir,
        )
    )
    
    if result["status"] == "success":
        logger.info(f"✓ Config completed: {result['num_queries']} queries in {result['elapsed']:.1f}s")
        print("\n" + "="*60)
        print("🎉 SMOKE TEST PASSED!")
        print("="*60 + "\n")
    else:
        logger.error(f"❌ Config failed: {result.get('error')}")
        return False
    
    return True


if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        exit(1)
