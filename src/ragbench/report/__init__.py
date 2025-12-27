"""Benchmark result aggregation and reporting."""

from .aggregate import aggregate_results, save_aggregated_results
from .render_md import render_markdown_report

__all__ = [
    "aggregate_results",
    "save_aggregated_results",
    "render_markdown_report",
]
