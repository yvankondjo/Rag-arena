"""Retrieval evaluation metrics using BEIR framework."""

from typing import Dict, List
from dataclasses import dataclass
from beir.retrieval.evaluation import EvaluateRetrieval


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    ndcg_at_10: float
    recall_at_5: float
    mrr_at_10: float
    num_queries: int


def evaluate_retrieval_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
) -> RetrievalMetrics:
    """Evaluate retrieval metrics using BEIR framework.

    Args:
        results: Retrieval results {query_id: {doc_id: score}}
        qrels: Query relevance judgments {query_id: {doc_id: relevance_score}}

    Returns:
        RetrievalMetrics with aggregated scores
    """

    evaluator = EvaluateRetrieval()

    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[5, 10])

    ndcg_at_10 = ndcg["NDCG@10"] if "NDCG@10" in ndcg else 0.0
    recall_at_5 = recall["Recall@5"] if "Recall@5" in recall else 0.0

    mrr_scores = []
    for qid, result in results.items():
        if qid not in qrels:
            continue

        sorted_docs = sorted(result.items(), key=lambda x: x[1], reverse=True)


        query_qrels = qrels[qid]
        for rank, (doc_id, _) in enumerate(sorted_docs, 1):
            if query_qrels.get(doc_id, 0) > 0:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    mrr_at_10 = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    return RetrievalMetrics(
        ndcg_at_10=ndcg_at_10,
        recall_at_5=recall_at_5,
        mrr_at_10=mrr_at_10,
        num_queries=len(results),
    )


def evaluate_single_query_retrieval(
    results: Dict[str, float],
    qrels: Dict[str, int],
) -> Dict[str, float]:
    """Evaluate retrieval metrics for a single query using BEIR.

    Args:
        results: Retrieval results {doc_id: score}
        qrels: Relevance judgments {doc_id: relevance_score}

    Returns:
        Dict with individual metric scores
    """

    single_results = {"query": results}
    single_qrels = {"query": qrels}

    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(single_qrels, single_results, k_values=[5, 10])

    sorted_docs = sorted(results.items(), key=lambda x: x[1], reverse=True)
    mrr_10 = 0.0
    for rank, (doc_id, _) in enumerate(sorted_docs[:10], 1):
        if qrels.get(doc_id, 0) > 0:
            mrr_10 = 1.0 / rank
            break

    return {
        "ndcg_at_10": ndcg.get("NDCG@10", 0.0),
        "recall_at_5": recall.get("Recall@5", 0.0),
        "mrr_at_10": mrr_10,
    }
