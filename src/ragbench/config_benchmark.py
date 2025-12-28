"""Centralized benchmark configuration - loads from configs/base.yaml.

This module ensures ALL components use the same parameters for fair comparison.
Configuration is loaded from YAML files, with env var overrides for parallelization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import os
import yaml


def _find_config_file() -> Path:
    """Find base.yaml config file."""
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / "base.yaml",
        Path("configs/base.yaml"),
        Path("../configs/base.yaml"),
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            return config_path
    
    raise FileNotFoundError("Could not find configs/base.yaml")


def _load_yaml_config() -> Dict[str, Any]:
    """Load configuration from base.yaml."""
    config_path = _find_config_file()
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Load config from YAML
# =============================================================================
_yaml_config = _load_yaml_config()

# Generation config
_gen_config = _yaml_config.get("generation", {})
GENERATION_TEMPERATURE = _gen_config.get("temperature", 0.0)
MAX_GENERATION_TOKENS = _gen_config.get("max_tokens", 2048)
SYSTEM_PROMPT = _gen_config.get("system_prompt", "You are a helpful assistant.").strip()
USER_PROMPT_TEMPLATE = _gen_config.get("user_prompt_template", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:").strip()

# Reranker config
_reranker_config = _yaml_config.get("reranker", {})
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", _reranker_config.get("backend", "cohere"))
RERANKER_MODEL = _reranker_config.get("local_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COHERE_RERANK_MODEL = _reranker_config.get("cohere_model", "rerank-v3.5")
COHERE_TOP_N = _reranker_config.get("cohere_top_n", 10)
USE_LOCAL_RERANKER_ONLY = RERANKER_BACKEND == "local"
# Over-fetch multiplier: retrieve 2x candidates for reranker to reorder
RERANKER_OVER_FETCH = _reranker_config.get("over_fetch_multiplier", 2)

# Evaluation config
_eval_config = _yaml_config.get("evaluation", {})
RAGAS_TEMPERATURE = _eval_config.get("ragas_temperature", 0.0)
RAGAS_MODEL = None  # Set from model at runtime

# Sampling config
_sampling_config = _yaml_config.get("sampling", {})
QUERY_SAMPLING_STRATEGY = _sampling_config.get("strategy", "random")
RANDOM_SEED = _sampling_config.get("seed", 42)

# Parallelization config (env vars override YAML)
_parallel_config = _yaml_config.get("parallelization", {})
MAX_CONCURRENT_QUERIES = int(os.getenv(
    "MAX_CONCURRENT_QUERIES", 
    _parallel_config.get("max_concurrent_queries", 5)
))
MAX_CONCURRENT_EMBEDDINGS = int(os.getenv(
    "MAX_CONCURRENT_EMBEDDINGS",
    _parallel_config.get("max_concurrent_embeddings", 10)
))
MAX_CONCURRENT_LLM_CALLS = int(os.getenv(
    "MAX_CONCURRENT_LLM_CALLS",
    _parallel_config.get("max_concurrent_llm_calls", 5)
))
MAX_CONCURRENT_RERANK_CALLS = int(os.getenv(
    "MAX_CONCURRENT_RERANK_CALLS",
    _parallel_config.get("max_concurrent_rerank_calls", 10)
))


@dataclass
class BenchmarkConfig:
    """Unified configuration for the benchmark.
    
    All parameters that MUST be identical across configurations are here.
    This ensures fair comparison between Simple and Agentic RAG.
    """
    
    # Generation (identical for all pipelines)
    generation_temperature: float = GENERATION_TEMPERATURE
    max_generation_tokens: int = MAX_GENERATION_TOKENS
    system_prompt: str = SYSTEM_PROMPT
    user_prompt_template: str = USER_PROMPT_TEMPLATE
    
    # Model (from config file, same for generation AND evaluation)
    model: str = field(default_factory=lambda: _yaml_config.get("model", "gpt-4o-mini"))
    
    # Evaluation
    ragas_temperature: float = RAGAS_TEMPERATURE
    
    # Reranker
    reranker_backend: str = RERANKER_BACKEND
    reranker_model: str = RERANKER_MODEL
    
    # Sampling
    query_sampling_strategy: str = QUERY_SAMPLING_STRATEGY
    random_seed: int = RANDOM_SEED
    
    # Retrieved context for RAGAS evaluation
    # For agentic: use ALL retrieved contexts, not just the last one
    use_cumulative_context_for_ragas: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.generation_temperature != 0.0:
            import warnings
            warnings.warn(
                f"Generation temperature {self.generation_temperature} != 0.0 - "
                "results may not be reproducible"
            )
        
        if self.query_sampling_strategy not in ("random", "stratified", "all"):
            raise ValueError(
                f"Invalid sampling strategy: {self.query_sampling_strategy}. "
                "Use 'random', 'stratified', or 'all'"
            )


@dataclass
class QueryLevelMetrics:
    """Metrics for a single query - enables proper statistical analysis.
    
    Storing per-query metrics allows:
    - Paired statistical tests (comparing same query across configs)
    - Bootstrap confidence intervals
    - Proper variance estimation
    """
    query_id: str
    query_text: str
    config_hash: str
    
    # Latency breakdown (in seconds)
    retrieval_latency: float = 0.0
    reranking_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    
    # Retrieval metrics (first step - for fair comparison)
    retrieved_doc_ids: list = field(default_factory=list)  # First step doc_ids
    relevant_doc_ids: list = field(default_factory=list)
    ndcg_at_10: float = 0.0  # First step NDCG
    recall_at_5: float = 0.0  # First step Recall
    mrr_at_10: float = 0.0  # First step MRR
    
    # Final step metrics (after query refinement for agentic)
    final_step_doc_ids: list = field(default_factory=list)
    final_step_ndcg_at_10: float = 0.0
    final_step_recall_at_5: float = 0.0
    final_step_mrr_at_10: float = 0.0
    
    # Orchestration gain (final - first, shows value of multi-step)
    orchestration_gain_ndcg: float = 0.0
    orchestration_gain_recall: float = 0.0
    
    # RAGAS metrics (per query) - only Faithfulness and AnswerRelevancy
    # For retrieval quality, use BEIR metrics above (ndcg, recall, mrr)
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    
    # Context used for generation AND evaluation
    context_chunks: list = field(default_factory=list)
    context_text: str = ""
    
    # Generated answer
    answer: str = ""
    
    # For agentic: number of retrieval steps
    num_retrieval_steps: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "config_hash": self.config_hash,
            "retrieval_latency": self.retrieval_latency,
            "reranking_latency": self.reranking_latency,
            "generation_latency": self.generation_latency,
            "total_latency": self.total_latency,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "relevant_doc_ids": self.relevant_doc_ids,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_5": self.recall_at_5,
            "mrr_at_10": self.mrr_at_10,
            "final_step_doc_ids": self.final_step_doc_ids,
            "final_step_ndcg_at_10": self.final_step_ndcg_at_10,
            "final_step_recall_at_5": self.final_step_recall_at_5,
            "final_step_mrr_at_10": self.final_step_mrr_at_10,
            "orchestration_gain_ndcg": self.orchestration_gain_ndcg,
            "orchestration_gain_recall": self.orchestration_gain_recall,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_chunks_count": len(self.context_chunks),
            "context_text_length": len(self.context_text),
            "answer_length": len(self.answer),
            "num_retrieval_steps": self.num_retrieval_steps,
        }


# Global benchmark configuration instance
_benchmark_config: Optional[BenchmarkConfig] = None


def load_model_from_config() -> str:
    """Load model name from base.yaml config file."""
    return _yaml_config.get("model", "gpt-4o-mini")


def get_benchmark_config() -> BenchmarkConfig:
    """Get the global benchmark configuration."""
    global _benchmark_config
    if _benchmark_config is None:
        _benchmark_config = BenchmarkConfig()
    return _benchmark_config


def set_benchmark_config(config: BenchmarkConfig):
    """Set the global benchmark configuration."""
    global _benchmark_config
    _benchmark_config = config
