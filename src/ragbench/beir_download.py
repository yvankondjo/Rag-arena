"""Download and prepare BEIR datasets using BEIR library."""

import logging
from pathlib import Path
from beir import util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


class BEIRDownloader:
    """Download BEIR datasets using BEIR library."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_name: str = "scifact") -> Path:
        """Download BEIR dataset using beir.util.

        Args:
            dataset_name: Name of the dataset to download

        Returns:
            Path to extracted dataset directory
        """
        dataset_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        extract_path = self.data_dir / dataset_name

        if extract_path.exists() and (extract_path / "corpus.jsonl").exists():
            logger.info(f"Dataset {dataset_name} already exists at {extract_path}")
            return extract_path

        logger.info(f"Downloading {dataset_name} from {dataset_url}...")
        data_path = util.download_and_unzip(dataset_url, str(self.data_dir))

        logger.info(f"Dataset ready at {data_path}")
        return Path(data_path)


def load_beir_corpus(dataset_path: Path) -> dict:
    """Load BEIR corpus using GenericDataLoader.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Dict[doc_id -> {title, text}]
    """
    corpus_path = dataset_path / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    corpus, _, _ = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
    logger.info(f"Loaded {len(corpus)} documents")
    return corpus


def load_beir_queries(dataset_path: Path) -> dict:
    """Load BEIR queries using GenericDataLoader.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Dict[query_id -> query_text]
    """
    queries_path = dataset_path / "queries.jsonl"
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries not found: {queries_path}")

    _, queries, _ = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_beir_qrels(dataset_path: Path) -> dict:
    """Load BEIR qrels using GenericDataLoader.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Dict[qid -> {doc_id -> relevance_score}]
    """
    qrels_path = dataset_path / "qrels" / "test.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels not found: {qrels_path}")

    _, _, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
    logger.info(f"Loaded qrels for {len(qrels)} queries")
    return qrels


def load_beir_dataset(dataset_path: Path) -> tuple:
    """Load full BEIR dataset (corpus, queries, qrels) using GenericDataLoader.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Tuple[corpus, queries, qrels]
    """
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split="test")
    logger.info(f"Loaded dataset: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    downloader = BEIRDownloader(data_dir)

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "scifact"

    dataset_path = downloader.download_dataset(dataset_name)

    corpus = load_beir_corpus(dataset_path)
    queries = load_beir_queries(dataset_path)
    qrels = load_beir_qrels(dataset_path)

    print(f"✓ Dataset ready: {dataset_name}")
