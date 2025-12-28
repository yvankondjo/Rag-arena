"""RAG evaluation metrics using Ragas library."""

import asyncio
import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Suppress verbose Ragas warnings about generation count
# OpenRouter doesn't support n>1, so Ragas always gets 1 generation instead of 3
# This is expected behavior, not an error
logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(logging.ERROR)

try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# LangChain embeddings for RAGAS (must have embed_query method)
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class RagasResult:
    """Result of Ragas evaluation.
    
    Only Faithfulness and AnswerRelevancy are computed.
    For retrieval quality, use BEIR metrics (NDCG, MRR, Recall)
    which are based on document ID relevance judgments (qrels).
    """
    faithfulness: float
    answer_relevancy: float
    num_samples: int = 0


def _get_llm_and_embeddings(model: str = None):
    """Configure LLM and embeddings for RAGAS evaluation.
    
    FIX #7: Use the same model as generation for consistency.
    
    Args:
        model: Model name to use (defaults to config_benchmark setting)
    
    Returns:
        Tuple of (llm, embeddings) or (None, None) if not available.
    """
    llm = None
    embeddings = None
    
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available for RAGAS")
        return None, None
    
    if model is None:
        try:
            from ragbench.config_benchmark import get_benchmark_config, RAGAS_TEMPERATURE
            config = get_benchmark_config()
            model = config.model if config.model else "gpt-4o-mini"
        except ImportError:
            model = "gpt-4o-mini"
    
    try:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            llm = ChatOpenAI(
                model=model,
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0,
                timeout=120,
                max_retries=3,
            )
            logger.info(f"RAGAS using OpenRouter LLM: {model}")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_api_key,
            )
            logger.info("RAGAS using OpenAI embeddings")
    except Exception as e:
        logger.warning(f"Failed to configure RAGAS LLM/embeddings: {e}")
    
    return llm, embeddings


def _run_ragas_evaluation_sync(dataset, metrics, llm, embeddings):
    """Run RAGAS evaluation synchronously in a dedicated event loop.
    
    This avoids conflicts with existing event loops by creating a fresh one.
    
    Args:
        dataset: HuggingFace Dataset
        metrics: List of RAGAS metrics
        llm: LangChain LLM
        embeddings: LangChain embeddings
        
    Returns:
        RAGAS evaluation result
    """

    try:
        loop = asyncio.get_running_loop()

        in_event_loop = True
    except RuntimeError:
        in_event_loop = False
    
    if in_event_loop:

        import concurrent.futures
        
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                if llm and embeddings:
                    return evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
                elif llm:
                    return evaluate(dataset, metrics=metrics, llm=llm)
                else:
                    return evaluate(dataset, metrics=metrics)
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result(timeout=600)
    else:

        if llm and embeddings:
            return evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        elif llm:
            return evaluate(dataset, metrics=metrics, llm=llm)
        else:
            return evaluate(dataset, metrics=metrics)


def evaluate_rag_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
) -> RagasResult:
    """Evaluate RAG system using Ragas metrics.

    Args:
        questions: List of questions
        answers: List of generated answers
        contexts: List of retrieved contexts (MUST be list of chunks, not joined text)
        ground_truths: Optional ground truth answers

    Returns:
        RagasResult with computed metrics
    """
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "Ragas not installed. Install with: pip install ragas"
        )

    if len(questions) != len(answers) or len(questions) != len(contexts):
        raise ValueError("Questions, answers, and contexts must have same length")


    llm, embeddings = _get_llm_and_embeddings()
    
    if not llm:
        logger.warning("No LLM configured for RAGAS - returning zero scores")
        return RagasResult(
            faithfulness=0.0,
            answer_relevancy=0.0,
            num_samples=len(questions),
        )


    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    dataset = Dataset.from_dict(data)

    # Only Faithfulness and AnswerRelevancy (no ground_truth needed)
    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings) if embeddings else AnswerRelevancy(llm=llm)
    metrics = [faithfulness_metric, answer_relevancy_metric]


    try:
        logger.info(f"Running RAGAS evaluation on {len(questions)} samples...")
        result = _run_ragas_evaluation_sync(dataset, metrics, llm, embeddings)
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise RuntimeError(f"RAGAS evaluation failed: {e}")


    df = result.to_pandas()
    
    faithfulness = df["faithfulness"].mean() if "faithfulness" in df.columns else 0.0
    answer_relevancy = df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else 0.0

    logger.info(f"RAGAS evaluation completed: faithfulness={faithfulness:.3f}, "
                f"answer_relevancy={answer_relevancy:.3f}")

    return RagasResult(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        num_samples=len(questions),
    )


def evaluate_single_query(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a single query-answer pair with Ragas.

    Args:
        question: The question
        answer: The generated answer
        contexts: Retrieved contexts
        ground_truth: Optional ground truth answer

    Returns:
        Dict with metric scores
    """
    result = evaluate_rag_with_ragas(
        questions=[question],
        answers=[answer],
        contexts=[contexts],
        ground_truths=[ground_truth] if ground_truth else None,
    )

    return {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
    }


def compute_ragas_aggregates(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute aggregate statistics from multiple Ragas evaluations.

    Args:
        results: List of individual evaluation results

    Returns:
        Dict with mean scores for each metric
    """
    if not results:
        return {
            "faithfulness_mean": 0.0,
            "answer_relevancy_mean": 0.0,
        }

    df = pd.DataFrame(results)

    return {
        "faithfulness_mean": df["faithfulness"].mean(),
        "answer_relevancy_mean": df["answer_relevancy"].mean(),
    }
