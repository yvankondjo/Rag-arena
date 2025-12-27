#!/usr/bin/env python3
"""Generate comprehensive benchmark report with graphs."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from datetime import datetime

def load_run_results(run_dir: Path) -> dict:
    """Load results from a single run directory."""
    config_file = run_dir / "config.yaml"
    metrics_file = run_dir / "metrics.json"

    # Load config
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Load metrics
    with open(metrics_file) as f:
        metrics = json.load(f)

    return {
        "config_hash": run_dir.name,
        "orchestration_mode": config.get("orchestration_mode", "unknown"),
        "retrieval_mode": config.get("retrieval_mode", "unknown"),
        "use_reranker": config.get("use_reranker", False),
        "dataset": config.get("dataset", "unknown"),
        "top_k": config.get("top_k", 10),
        "max_agentic_steps": config.get("max_agentic_steps", 3),
        "elapsed_seconds": metrics.get("elapsed_seconds", 0),
        "avg_time_per_query": metrics.get("avg_time_per_query", 0),
        "num_queries": metrics.get("num_queries", 0),
        "success_count": metrics.get("successful_queries", 0),
        "error_count": metrics.get("error_count", 0),
        # Retrieval metrics
        "ndcg_at_10": metrics.get("ndcg_at_10"),
        "recall_at_5": metrics.get("recall_at_5"),
        "mrr_at_10": metrics.get("mrr_at_10"),
        # RAG quality metrics
        "faithfulness": metrics.get("faithfulness"),
        "answer_relevancy": metrics.get("answer_relevancy"),
        "context_precision": metrics.get("context_precision"),
        "context_recall": metrics.get("context_recall"),
    }

def aggregate_results(results_dir: Path) -> pd.DataFrame:
    """Aggregate results from all runs."""
    runs_dir = results_dir / "runs"
    run_dirs = [Path(d) for d in runs_dir.glob("*") if Path(d).is_dir()]

    results = []
    for run_dir in run_dirs:
        try:
            result = load_run_results(run_dir)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {run_dir}: {e}")
            continue

    return pd.DataFrame(results)

def create_comparison_charts(df: pd.DataFrame, output_dir: Path):
    """Create matplotlib comparison charts."""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create output directory for images
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 1. Performance by Orchestration Mode
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RAGBench-12x: Performance Comparison by Orchestration Mode', fontsize=16, fontweight='bold')

    # Time comparison
    orch_time = df.groupby('orchestration_mode')['elapsed_seconds'].mean()
    axes[0, 0].bar(orch_time.index, orch_time.values, color=['#2E86AB', '#F24236'])
    axes[0, 0].set_title('Average Execution Time by Orchestration Mode')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].grid(True, alpha=0.3)

    # NDCG comparison
    orch_ndcg = df.groupby('orchestration_mode')['ndcg_at_10'].mean()
    axes[0, 1].bar(orch_ndcg.index, orch_ndcg.values, color=['#2E86AB', '#F24236'])
    axes[0, 1].set_title('NDCG@10 by Orchestration Mode')
    axes[0, 1].set_ylabel('NDCG@10')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    # Faithfulness comparison
    orch_faith = df.groupby('orchestration_mode')['faithfulness'].mean()
    axes[1, 0].bar(orch_faith.index, orch_faith.values, color=['#2E86AB', '#F24236'])
    axes[1, 0].set_title('Faithfulness by Orchestration Mode')
    axes[1, 0].set_ylabel('Faithfulness')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Answer Relevancy comparison
    orch_rel = df.groupby('orchestration_mode')['answer_relevancy'].mean()
    axes[1, 1].bar(orch_rel.index, orch_rel.values, color=['#2E86AB', '#F24236'])
    axes[1, 1].set_title('Answer Relevancy by Orchestration Mode')
    axes[1, 1].set_ylabel('Answer Relevancy')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(images_dir / 'orchestration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance by Retrieval Mode
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RAGBench-12x: Performance Comparison by Retrieval Mode', fontsize=16, fontweight='bold')

    # Time comparison
    retr_time = df.groupby('retrieval_mode')['elapsed_seconds'].mean()
    colors = ['#2E86AB', '#F24236', '#F5B700']
    axes[0, 0].bar(retr_time.index, retr_time.values, color=colors)
    axes[0, 0].set_title('Average Execution Time by Retrieval Mode')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].grid(True, alpha=0.3)

    # NDCG comparison
    retr_ndcg = df.groupby('retrieval_mode')['ndcg_at_10'].mean()
    axes[0, 1].bar(retr_ndcg.index, retr_ndcg.values, color=colors)
    axes[0, 1].set_title('NDCG@10 by Retrieval Mode')
    axes[0, 1].set_ylabel('NDCG@10')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    # Faithfulness comparison
    retr_faith = df.groupby('retrieval_mode')['faithfulness'].mean()
    axes[1, 0].bar(retr_faith.index, retr_faith.values, color=colors)
    axes[1, 0].set_title('Faithfulness by Retrieval Mode')
    axes[1, 0].set_ylabel('Faithfulness')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Answer Relevancy comparison
    retr_rel = df.groupby('retrieval_mode')['answer_relevancy'].mean()
    axes[1, 1].bar(retr_rel.index, retr_rel.values, color=colors)
    axes[1, 1].set_title('Answer Relevancy by Retrieval Mode')
    axes[1, 1].set_ylabel('Answer Relevancy')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(images_dir / 'retrieval_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Performance by Reranker Mode
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RAGBench-12x: Performance Comparison by Reranker Mode', fontsize=16, fontweight='bold')

    # Time comparison
    rer_time = df.groupby('use_reranker')['elapsed_seconds'].mean()
    axes[0, 0].bar(['No Rerank', 'With Rerank'], rer_time.values, color=['#2E86AB', '#F24236'])
    axes[0, 0].set_title('Average Execution Time by Reranker Mode')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].grid(True, alpha=0.3)

    # NDCG comparison
    rer_ndcg = df.groupby('use_reranker')['ndcg_at_10'].mean()
    axes[0, 1].bar(['No Rerank', 'With Rerank'], rer_ndcg.values, color=['#2E86AB', '#F24236'])
    axes[0, 1].set_title('NDCG@10 by Reranker Mode')
    axes[0, 1].set_ylabel('NDCG@10')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)

    # Faithfulness comparison
    rer_faith = df.groupby('use_reranker')['faithfulness'].mean()
    axes[1, 0].bar(['No Rerank', 'With Rerank'], rer_faith.values, color=['#2E86AB', '#F24236'])
    axes[1, 0].set_title('Faithfulness by Reranker Mode')
    axes[1, 0].set_ylabel('Faithfulness')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Answer Relevancy comparison
    rer_rel = df.groupby('use_reranker')['answer_relevancy'].mean()
    axes[1, 1].bar(['No Rerank', 'With Rerank'], rer_rel.values, color=['#2E86AB', '#F24236'])
    axes[1, 1].set_title('Answer Relevancy by Reranker Mode')
    axes[1, 1].set_ylabel('Answer Relevancy')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(images_dir / 'reranker_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. All 12 Configurations Performance
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('RAGBench-12x: All Configurations Performance Overview', fontsize=16, fontweight='bold')

    # Create config labels
    df['config_label'] = df.apply(lambda x: f"{x['orchestration_mode'][:4]}_{x['retrieval_mode'][:4]}_{'rer' if x['use_reranker'] else 'nor'}", axis=1)

    # Sort by time for consistent ordering
    df_sorted = df.sort_values('elapsed_seconds')

    # Time comparison
    axes[0, 0].bar(range(len(df_sorted)), df_sorted['elapsed_seconds'], color=plt.cm.tab20.colors[:12])
    axes[0, 0].set_title('Execution Time by Configuration')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_xticks(range(len(df_sorted)))
    axes[0, 0].set_xticklabels(df_sorted['config_label'], rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)

    # NDCG comparison
    axes[0, 1].bar(range(len(df_sorted)), df_sorted['ndcg_at_10'], color=plt.cm.tab20.colors[:12])
    axes[0, 1].set_title('NDCG@10 by Configuration')
    axes[0, 1].set_ylabel('NDCG@10')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(range(len(df_sorted)))
    axes[0, 1].set_xticklabels(df_sorted['config_label'], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)

    # Faithfulness comparison
    axes[1, 0].bar(range(len(df_sorted)), df_sorted['faithfulness'], color=plt.cm.tab20.colors[:12])
    axes[1, 0].set_title('Faithfulness by Configuration')
    axes[1, 0].set_ylabel('Faithfulness')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticks(range(len(df_sorted)))
    axes[1, 0].set_xticklabels(df_sorted['config_label'], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)

    # Answer Relevancy comparison
    axes[1, 1].bar(range(len(df_sorted)), df_sorted['answer_relevancy'], color=plt.cm.tab20.colors[:12])
    axes[1, 1].set_title('Answer Relevancy by Configuration')
    axes[1, 1].set_ylabel('Answer Relevancy')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(range(len(df_sorted)))
    axes[1, 1].set_xticklabels(df_sorted['config_label'], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(images_dir / 'all_configurations.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated comparison charts in {images_dir}")

def update_readme_with_graphs(readme_path: Path, images_dir: Path):
    """Update README.md with graphs."""
    # Read current README
    with open(readme_path, 'r') as f:
        content = f.read()

    # Create graphs section
    graphs_section = f"""

## Performance Analysis

### Orchestration Mode Comparison
![Orchestration Comparison]({images_dir}/orchestration_comparison.png)

### Retrieval Mode Comparison
![Retrieval Comparison]({images_dir}/retrieval_comparison.png)

### Reranker Mode Comparison
![Reranker Comparison]({images_dir}/reranker_comparison.png)

### All 12 Configurations Overview
![All Configurations]({images_dir}/all_configurations.png)

*Benchmark results from RAGBench-12x with SciFact dataset (50 queries per configuration)*
"""

    # Find the end of the main content (before any existing graphs or appendices)
    # If there's already a graphs section, replace it
    if "## Performance Analysis" in content:
        # Replace existing section
        start = content.find("## Performance Analysis")
        end = content.find("\n## ", start + 1)
        if end == -1:
            end = len(content)
        content = content[:start] + graphs_section
    else:
        # Add after main content, before any appendices
        if "## Contributing" in content:
            insert_pos = content.find("## Contributing")
            content = content[:insert_pos] + graphs_section + "\n" + content[insert_pos:]
        else:
            content += graphs_section

    # Write back
    with open(readme_path, 'w') as f:
        f.write(content)

    print(f"Updated {readme_path} with performance graphs")

def main():
    """Main function."""
    results_dir = Path("results")
    output_dir = Path("results")

    # Aggregate results
    print("Aggregating results...")
    df = aggregate_results(results_dir)

    # Save aggregated results
    df.to_csv(output_dir / "results.csv", index=False)
    df.to_json(output_dir / "results.jsonl", orient='records', lines=True)

    print(f"Aggregated {len(df)} configurations")

    # Create comparison charts
    print("Creating comparison charts...")
    create_comparison_charts(df, output_dir)

    # Update README with graphs
    readme_path = Path("README.md")
    if readme_path.exists():
        images_dir = output_dir / "images"
        update_readme_with_graphs(readme_path, images_dir)

    print("Report generation complete!")

if __name__ == "__main__":
    main()
