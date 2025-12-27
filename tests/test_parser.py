import argparse
import logging
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def cmd_report(args):
    """Generate report from benchmark results."""
    from ragbench.report.aggregate import generate_report

    generate_report(results_dir=Path(args.results_dir), output_path=Path(args.output))

    print(f"✓ Report generated at {args.output}")

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark CLI")
    subparser = parser.add_subparsers(dest="command")
    report_parser = subparser.add_parser("report", help="Generate benchmark report")
    report_parser.add_argument( 
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing benchmark results",
    )