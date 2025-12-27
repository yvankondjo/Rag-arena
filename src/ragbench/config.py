"""Configuration management for RAGBench-12x."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from ragbench.clients import  create_embedding_client

load_dotenv()

# Root directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Application configuration with environment-based defaults."""

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter"))
    llm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_API_KEY" if os.getenv("LLM_PROVIDER", "openrouter") == "openrouter" else "OPENAI_API_KEY",
            "sk_test_placeholder"
        )
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))


    embedding_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk_test_placeholder"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))


    dataset_name: str = "scifact"
    dataset_split: str = "test"
    num_queries: Optional[int] = None


    retrieval_mode: str = "dense"
    use_reranker: bool = False
    top_k: int = 10
    top_k_bm25: int = 20
    top_k_dense: int = 20
    top_k_rerank: int = 10


    orchestration_mode: str = "simple"
    max_agentic_steps: int = 3


    persist_directory: Path = field(default_factory=lambda: DATA_DIR / "indexes" / "chroma")
    bm25_index_dir: Path = field(default_factory=lambda: DATA_DIR / "indexes" / "bm25s")
    raw_data_dir: Path = field(default_factory=lambda: DATA_DIR / "raw")
    processed_data_dir: Path = field(default_factory=lambda: DATA_DIR / "processed")
    log_file: Path = field(default_factory=lambda: RESULTS_DIR / "logs" / "rag_logs.jsonl")

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._create_directories()

    def _validate(self):
        """Validate configuration values."""

        if not self.llm_api_key or self.llm_api_key == "sk_test_placeholder":
            import warnings
            warnings.warn("LLM API key not set or using placeholder. LLM calls will fail at runtime.")

        if not self.embedding_api_key or self.embedding_api_key == "sk_test_placeholder":
            import warnings
            warnings.warn("Embedding API key not set or using placeholder. Embedding calls will fail at runtime.")

        if self.retrieval_mode not in ("dense", "keyword", "hybrid"):
            raise ValueError(
                f"retrieval_mode must be 'dense', 'keyword', or 'hybrid', got {self.retrieval_mode}"
            )

        if self.orchestration_mode not in ("simple", "agentic"):
            raise ValueError(
                f"orchestration_mode must be 'simple' or 'agentic', got {self.orchestration_mode}"
            )

    def _create_directories(self):
        """Create necessary directories."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    # LLM client creation removed - handled directly in pipelines

    def create_embedding_client(self):
        """Create embedding client based on configuration."""
        return create_embedding_client(
            api_key=self.embedding_api_key,
            model=self.embedding_model,
        )





class Config:
    """Legacy configuration class (backward compatible)."""

    DATA_DIR = DATA_DIR
    PERSIST_DIRECTORY = DATA_DIR / "indexes" / "chroma"
    PERSIST_DATABASE = DATA_DIR / "database.db"
    SAVE_DOCUMENTS_DIR = DATA_DIR / "documents"
    LOG_FILE = RESULTS_DIR / "logs" / "rag_logs.jsonl"
    EMBEDDING_MODEL = "text-embedding-3-small"
    VECTOR_STORE_COLLECTION = "scifact"
    RERANKER_MODEL = "text-davinci-003"
    BM25_INDEX_DIR = DATA_DIR / "indexes" / "bm25s"


default_config = None
