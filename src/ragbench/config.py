"""Configuration management for RAGBench-12x."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

# Root directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file and ensure it contains a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}

    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(content).__name__}")

    return content


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Configuration for a single benchmark variant from the public 12x grid."""

    orchestration_mode: str
    retrieval_mode: str
    use_reranker: bool = False
    dataset: str = "scifact"
    model: str = "gpt-4o-mini"
    top_k: int = 10
    max_agentic_steps: int = 3

    def get_run_name(self) -> str:
        """Return a stable human-readable name for the run."""
        reranker = "rerank" if self.use_reranker else "no_rerank"
        return f"{self.orchestration_mode}_{self.retrieval_mode}_{reranker}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable representation."""
        return {
            "orchestration_mode": self.orchestration_mode,
            "retrieval_mode": self.retrieval_mode,
            "use_reranker": self.use_reranker,
            "dataset": self.dataset,
            "model": self.model,
            "top_k": self.top_k,
            "max_agentic_steps": self.max_agentic_steps,
        }


@dataclass
class AppConfig:
    """Application configuration with environment-based defaults."""

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter"))
    llm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_API_KEY"
            if os.getenv("LLM_PROVIDER", "openrouter") == "openrouter"
            else "OPENAI_API_KEY",
            "sk_test_placeholder",
        )
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk_test_placeholder")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
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
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    log_file: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.persist_directory = Path(self.persist_directory)
        self.bm25_index_dir = Path(self.bm25_index_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.results_dir = Path(self.results_dir)
        self.log_file = (
            Path(self.log_file) if self.log_file else self.results_dir / "logs" / "rag_logs.jsonl"
        )

        self._validate()
        self._create_directories()

    def _validate(self):
        """Validate configuration values."""
        if not self.llm_api_key or self.llm_api_key == "sk_test_placeholder":
            import warnings

            warnings.warn("LLM API key not set or using placeholder. LLM calls will fail at runtime.")

        if not self.embedding_api_key or self.embedding_api_key == "sk_test_placeholder":
            import warnings

            warnings.warn(
                "Embedding API key not set or using placeholder. Embedding calls will fail at runtime."
            )

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
        assert self.log_file is not None
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def create_embedding_client(self):
        """Create embedding client based on configuration."""
        from ragbench.clients import create_embedding_client

        return create_embedding_client(
            api_key=self.embedding_api_key,
            model=self.embedding_model,
        )


def load_config_from_yaml(
    base_yaml: Optional[Path] = None,
    axes_yaml: Optional[Path] = None,
) -> list[BenchmarkRunConfig]:
    """Load the public 12x benchmark grid from YAML files."""
    base_config_path = Path(base_yaml) if base_yaml is not None else CONFIG_DIR / "base.yaml"
    axes_config_path = Path(axes_yaml) if axes_yaml is not None else CONFIG_DIR / "axes.yaml"

    base_config = _load_yaml_config(base_config_path)
    axes_config = _load_yaml_config(axes_config_path)

    orchestration_modes = axes_config.get("orchestration", {}).get("modes", [])
    retrieval_modes = axes_config.get("retrieval", {}).get("modes", [])
    reranker_modes = axes_config.get("reranker", {}).get("modes", [])

    if not orchestration_modes or not retrieval_modes or not reranker_modes:
        raise ValueError("axes.yaml must define non-empty orchestration, retrieval, and reranker modes")

    normalized_reranker_modes: list[bool] = []
    for mode in reranker_modes:
        if mode in ("rerank", True):
            normalized_reranker_modes.append(True)
        elif mode in ("no_rerank", False):
            normalized_reranker_modes.append(False)
        else:
            raise ValueError(f"Unsupported reranker mode: {mode!r}")

    configs: list[BenchmarkRunConfig] = []
    for orchestration_mode in orchestration_modes:
        for retrieval_mode in retrieval_modes:
            for use_reranker in normalized_reranker_modes:
                configs.append(
                    BenchmarkRunConfig(
                        orchestration_mode=orchestration_mode,
                        retrieval_mode=retrieval_mode,
                        use_reranker=use_reranker,
                        dataset=base_config.get("dataset_name", "scifact"),
                        model=base_config.get("model", "gpt-4o-mini"),
                        top_k=base_config.get("top_k", 10),
                        max_agentic_steps=base_config.get("max_agentic_steps", 3),
                    )
                )

    return configs


class Config:
    """Legacy configuration class (backward compatible)."""

    DATA_DIR = DATA_DIR
    RESULTS_DIR = RESULTS_DIR
    PERSIST_DIRECTORY = DATA_DIR / "indexes" / "chroma"
    PERSIST_DATABASE = DATA_DIR / "database.db"
    SAVE_DOCUMENTS_DIR = DATA_DIR / "documents"
    LOG_FILE = RESULTS_DIR / "logs" / "rag_logs.jsonl"
    EMBEDDING_MODEL = "text-embedding-3-small"
    VECTOR_STORE_COLLECTION = "scifact"
    RERANKER_MODEL = "text-davinci-003"
    BM25_INDEX_DIR = DATA_DIR / "indexes" / "bm25s"


default_config = None
