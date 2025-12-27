"""CLI for RAGBench-12x."""

import argparse
import logging
import sys
from pathlib import Path

from ragbench.beir_download import BEIRDownloader, load_beir_corpus, load_beir_queries, load_beir_qrels
from ragbench.build_indexes import build_indexes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_download(args):
    """Download BEIR dataset."""
    from ragbench.config import AppConfig
    config = AppConfig()

    downloader = BEIRDownloader(config.raw_data_dir)
    dataset_path = downloader.download_dataset(args.dataset)

    corpus = load_beir_corpus(dataset_path)
    queries = load_beir_queries(dataset_path)
    qrels = load_beir_qrels(dataset_path)

    print(f"✓ Dataset {args.dataset} downloaded")
    print(f"  - Corpus: {len(corpus)} documents")
    print(f"  - Queries: {len(queries)}")
    print(f"  - Qrels: {len(qrels)} queries with relevance")


def cmd_index(args):
    """Build indexes (Chroma + BM25) - doc-level."""
    result = build_indexes(dataset_name=args.dataset)

    print(f"✓ Indexes built for {args.dataset} (doc-level)")
    print(f"  - Documents: {len(result['documents'])}")
    print(f"  - Queries: {len(result['queries'])}")


def cmd_benchmark(args):
    """Run full 12-config benchmark."""
    from ragbench.run_experiment import run_benchmark

    results = run_benchmark(
        dataset=args.dataset,
        parallel=args.parallel,
        max_configs=args.max_configs,
        max_queries=args.max_queries,
        dry_run=args.dry_run,
        resume=args.resume,
    )

    if results:
        print(f"✓ Benchmark completed")
        print(f"  - Results: {len(results)} configurations")


def cmd_report(args):
    """Generate benchmark report."""
    from ragbench.report.aggregate import aggregate_results
    from ragbench.report.render_md import render_markdown_report

    # Aggregate results
    results_df, metrics_df = aggregate_results(args.results_dir)

    # Generate markdown report
    report_md = render_markdown_report(results_df, metrics_df, args.results_dir)

    # Write report
    with open(args.output, 'w') as f:
        f.write(report_md)

    print(f"✓ Report generated: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="RAGBench-12x: Benchmark 12 RAG configurations (doc-level)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # download
    parser_dl = subparsers.add_parser("download", help="Download BEIR dataset")
    parser_dl.add_argument("--dataset", default="scifact", help="Dataset name")
    parser_dl.set_defaults(func=cmd_download)

    # index
    parser_idx = subparsers.add_parser("index", help="Build indexes (doc-level)")
    parser_idx.add_argument("--dataset", default="scifact", help="Dataset name")
    parser_idx.set_defaults(func=cmd_index)

    # benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Run 12-config benchmark")
    parser_bench.add_argument("--dataset", default="scifact", help="Dataset name")
    parser_bench.add_argument("--parallel", action="store_true", help="Use AsyncIO parallelization (max 3 concurrent)")
    parser_bench.add_argument("--resume", action="store_true", 
                              help="Resume interrupted run - skip already completed configs")
    parser_bench.add_argument("--max-configs", type=int, default=None, help="Limit configs (for testing)")
    parser_bench.add_argument("--max-queries", type=int, default=None, help="Limit queries (for testing)")
    parser_bench.add_argument("--dry-run", action="store_true", help="Dry run: show configs only")
    parser_bench.set_defaults(func=cmd_benchmark)

    # report
    parser_report = subparsers.add_parser("report", help="Generate benchmark report")
    parser_report.add_argument("--results-dir", type=Path, default=Path("results"), help="Results directory")
    parser_report.add_argument("--output", type=Path, default=Path("results/report.md"), help="Output report file")
    parser_report.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
