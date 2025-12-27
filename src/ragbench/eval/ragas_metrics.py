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
        ContextPrecision,
        ContextRecall,
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
    """Result of Ragas evaluation."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    num_samples: int


def _get_llm_and_embeddings():
    """Configure LLM and embeddings for RAGAS evaluation.
    
    Returns:
        Tuple of (llm, embeddings) or (None, None) if not available.
    """
    llm = None
    embeddings = None
    
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available for RAGAS")
        return None, None
    
    try:

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0,
                timeout=120,
                max_retries=3,
            )
            logger.info("RAGAS using OpenRouter LLM")
        

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
            context_precision=0.0,
            context_recall=0.0,
            num_samples=len(questions),
        )


    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)



    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings) if embeddings else AnswerRelevancy(llm=llm)

    metrics = [faithfulness_metric, answer_relevancy_metric]


    if ground_truths:
        context_precision_metric = ContextPrecision(llm=llm)
        context_recall_metric = ContextRecall(llm=llm)
        metrics.extend([context_precision_metric, context_recall_metric])


    try:
        logger.info(f"Running RAGAS evaluation on {len(questions)} samples...")
        result = _run_ragas_evaluation_sync(dataset, metrics, llm, embeddings)
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise RuntimeError(f"RAGAS evaluation failed: {e}")


    df = result.to_pandas()
    

    available_columns = df.columns.tolist()
    scores = {}
    
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in available_columns:
            scores[col] = df[col].mean()
        else:
            scores[col] = 0.0

    logger.info(f"RAGAS evaluation completed: faithfulness={scores.get('faithfulness', 0):.3f}, "
                f"answer_relevancy={scores.get('answer_relevancy', 0):.3f}")

    return RagasResult(
        faithfulness=scores.get("faithfulness", 0.0),
        answer_relevancy=scores.get("answer_relevancy", 0.0),
        context_precision=scores.get("context_precision", 0.0),
        context_recall=scores.get("context_recall", 0.0),
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
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
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
            "context_precision_mean": 0.0,
            "context_recall_mean": 0.0,
        }

    df = pd.DataFrame(results)

    return {
        "faithfulness_mean": df["faithfulness"].mean(),
        "answer_relevancy_mean": df["answer_relevancy"].mean(),
        "context_precision_mean": df["context_precision"].mean(),
        "context_recall_mean": df["context_recall"].mean(),
    }
