"""Rigorous statistical analysis for RAGBench.

FIX #3: Proper statistical tests using query-level metrics.
- Bootstrap confidence intervals
- Paired tests (Wilcoxon signed-rank for non-parametric)
- Multiple comparison corrections (Bonferroni, Holm)

The previous approach was INVALID because:
1. T-tests were done on 6 config-level aggregates
2. These 6 points share the same queries → not independent
3. Correct: compare at query-level where we have 50+ paired observations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from scipy import stats
import warnings


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_samples: int


@dataclass
class PairedTestResult:
    """Result of a paired statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    significant_at_05: bool
    significant_at_01: bool
    mean_diff: float
    n_pairs: int


@dataclass
class ComparisonResult:
    """Full comparison between two groups."""
    group_a: str
    group_b: str
    metric: str
    mean_a: float
    mean_b: float
    bootstrap_ci_a: BootstrapCI
    bootstrap_ci_b: BootstrapCI
    paired_test: PairedTestResult
    # Multiple comparison corrected p-value
    corrected_p_value: Optional[float] = None
    corrected_significant: Optional[bool] = None


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for the mean.
    
    Args:
        data: Array of observations
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        BootstrapCI with mean, CI bounds, and standard error
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaNs
    
    if len(data) == 0:
        return BootstrapCI(
            mean=np.nan, ci_lower=np.nan, ci_upper=np.nan,
            std_error=np.nan, n_samples=0
        )
    
    rng = np.random.RandomState(seed)
    
    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return BootstrapCI(
        mean=np.mean(data),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=np.std(bootstrap_means),
        n_samples=len(data),
    )


def paired_wilcoxon_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
) -> PairedTestResult:
    """Wilcoxon signed-rank test for paired samples (non-parametric).
    
    This is the correct test for comparing two RAG systems on the SAME queries.
    Does not assume normal distribution.
    
    Args:
        data_a: Metric values for system A
        data_b: Metric values for system B (same queries, same order)
        
    Returns:
        PairedTestResult
    """
    data_a = np.array(data_a)
    data_b = np.array(data_b)
    
    # Remove pairs with NaN
    valid_mask = ~(np.isnan(data_a) | np.isnan(data_b))
    data_a = data_a[valid_mask]
    data_b = data_b[valid_mask]
    
    if len(data_a) < 5:
        return PairedTestResult(
            test_name="wilcoxon_signed_rank",
            statistic=np.nan,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="rank_biserial",
            significant_at_05=False,
            significant_at_01=False,
            mean_diff=np.nan,
            n_pairs=len(data_a),
        )
    
    # Wilcoxon signed-rank test
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = stats.wilcoxon(data_a, data_b, alternative='two-sided')
    except ValueError:
        # All differences are zero
        return PairedTestResult(
            test_name="wilcoxon_signed_rank",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="rank_biserial",
            significant_at_05=False,
            significant_at_01=False,
            mean_diff=0.0,
            n_pairs=len(data_a),
        )
    
    # Rank-biserial correlation as effect size
    # r = 1 - (2*W) / (n*(n+1)/2) where W is the smaller of W+ and W-
    n = len(data_a)
    effect_size = 1 - (2 * statistic) / (n * (n + 1) / 2)
    
    return PairedTestResult(
        test_name="wilcoxon_signed_rank",
        statistic=statistic,
        p_value=p_value,
        effect_size=abs(effect_size),
        effect_size_name="rank_biserial",
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        mean_diff=np.mean(data_a) - np.mean(data_b),
        n_pairs=n,
    )


def paired_ttest(
    data_a: np.ndarray,
    data_b: np.ndarray,
) -> PairedTestResult:
    """Paired t-test for comparing two systems on same queries.
    
    Use this only if data is approximately normal.
    For RAG metrics, Wilcoxon is generally safer.
    
    Args:
        data_a: Metric values for system A
        data_b: Metric values for system B (same queries, same order)
        
    Returns:
        PairedTestResult
    """
    data_a = np.array(data_a)
    data_b = np.array(data_b)
    
    # Remove pairs with NaN
    valid_mask = ~(np.isnan(data_a) | np.isnan(data_b))
    data_a = data_a[valid_mask]
    data_b = data_b[valid_mask]
    
    if len(data_a) < 3:
        return PairedTestResult(
            test_name="paired_ttest",
            statistic=np.nan,
            p_value=1.0,
            effect_size=0.0,
            effect_size_name="cohens_d",
            significant_at_05=False,
            significant_at_01=False,
            mean_diff=np.nan,
            n_pairs=len(data_a),
        )
    
    # Paired t-test
    statistic, p_value = stats.ttest_rel(data_a, data_b)
    
    # Cohen's d for paired samples
    diff = data_a - data_b
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0
    
    return PairedTestResult(
        test_name="paired_ttest",
        statistic=statistic,
        p_value=p_value,
        effect_size=abs(cohens_d),
        effect_size_name="cohens_d",
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        mean_diff=np.mean(data_a) - np.mean(data_b),
        n_pairs=len(data_a),
    )


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level (default 0.05)
        
    Returns:
        List of (corrected_p_value, is_significant) tuples
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    return [(p * n_tests, p < corrected_alpha) for p in p_values]


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down correction (less conservative).
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level (default 0.05)
        
    Returns:
        List of (corrected_p_value, is_significant) tuples (same order as input)
    """
    n = len(p_values)
    
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = [p_values[i] for i in sorted_indices]
    
    # Apply Holm correction
    corrected = [None] * n
    significant = [False] * n
    
    for rank, (orig_idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted_p = min(p * (n - rank), 1.0)
        corrected[orig_idx] = adjusted_p
        
        # Holm: reject if p < alpha / (n - rank + 1)
        threshold = alpha / (n - rank)
        significant[orig_idx] = p < threshold
    
    return list(zip(corrected, significant))


def load_query_level_metrics(results_dir: Path) -> pd.DataFrame:
    """Load query-level metrics from all runs.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        DataFrame with query-level metrics, indexed by (config_hash, query_id)
    """
    runs_dir = results_dir / "runs"
    all_metrics = []
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        query_metrics_file = run_dir / "query_metrics.jsonl"
        config_file = run_dir / "config.yaml"
        
        if not query_metrics_file.exists():
            continue
        
        # Load config
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Load query metrics
        with open(query_metrics_file) as f:
            for line in f:
                metric = json.loads(line)
                metric["orchestration_mode"] = config.get("orchestration_mode", "unknown")
                metric["retrieval_mode"] = config.get("retrieval_mode", "unknown")
                metric["use_reranker"] = config.get("use_reranker", False)
                all_metrics.append(metric)
    
    if not all_metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(all_metrics)


def compare_orchestration_modes(
    df: pd.DataFrame,
    metric: str,
) -> ComparisonResult:
    """Compare simple vs agentic using paired tests on shared queries.
    
    Args:
        df: DataFrame with query-level metrics
        metric: Metric name to compare
        
    Returns:
        ComparisonResult with bootstrap CIs and paired test
    """
    simple = df[df["orchestration_mode"] == "simple"]
    agentic = df[df["orchestration_mode"] == "agentic"]
    
    # Find common queries
    simple_queries = set(simple["query_id"].unique())
    agentic_queries = set(agentic["query_id"].unique())
    common_queries = simple_queries & agentic_queries
    
    # Get paired values (average across retrieval modes for each query)
    simple_by_query = simple.groupby("query_id")[metric].mean()
    agentic_by_query = agentic.groupby("query_id")[metric].mean()
    
    # Align
    common_queries = sorted(common_queries)
    simple_values = np.array([simple_by_query.get(q, np.nan) for q in common_queries])
    agentic_values = np.array([agentic_by_query.get(q, np.nan) for q in common_queries])
    
    # Bootstrap CIs
    bootstrap_simple = bootstrap_confidence_interval(simple_values)
    bootstrap_agentic = bootstrap_confidence_interval(agentic_values)
    
    # Paired Wilcoxon test (non-parametric, appropriate for bounded metrics)
    paired_test = paired_wilcoxon_test(simple_values, agentic_values)
    
    return ComparisonResult(
        group_a="simple",
        group_b="agentic",
        metric=metric,
        mean_a=np.nanmean(simple_values),
        mean_b=np.nanmean(agentic_values),
        bootstrap_ci_a=bootstrap_simple,
        bootstrap_ci_b=bootstrap_agentic,
        paired_test=paired_test,
    )


def run_full_analysis(results_dir: Path) -> Dict[str, Any]:
    """Run complete statistical analysis with proper methodology.
    
    Returns dict with:
    - comparisons: List of ComparisonResult
    - methodology: Description of statistical approach
    - warnings: Any methodological warnings
    """
    df = load_query_level_metrics(results_dir)
    
    if df.empty:
        return {
            "error": "No query-level metrics found. Run benchmark first.",
            "comparisons": [],
            "methodology": "",
            "warnings": [],
        }
    
    metrics = ["ndcg_at_10", "recall_at_5", "mrr_at_10"]
    
    # Add faithfulness/relevancy if available
    if "faithfulness" in df.columns:
        metrics.append("faithfulness")
    if "answer_relevancy" in df.columns:
        metrics.append("answer_relevancy")
    
    comparisons = []
    p_values = []
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        result = compare_orchestration_modes(df, metric)
        comparisons.append(result)
        p_values.append(result.paired_test.p_value)
    
    # Apply Holm correction for multiple comparisons
    if p_values:
        corrections = holm_correction(p_values)
        for comp, (corrected_p, significant) in zip(comparisons, corrections):
            comp.corrected_p_value = corrected_p
            comp.corrected_significant = significant
    
    n_queries = len(df["query_id"].unique())
    n_configs = len(df["config_hash"].unique())
    
    methodology = f"""
Statistical Methodology (Corrected):
=====================================
- Unit of analysis: Query (n={n_queries}), NOT configuration
- Paired tests: Each query evaluated by both Simple and Agentic RAG
- Test used: Wilcoxon signed-rank (non-parametric, no normality assumption)
- Effect size: Rank-biserial correlation
- Multiple comparison correction: Holm-Bonferroni (step-down)
- Confidence intervals: Bootstrap (10,000 resamples)

Previous approach was INVALID because:
- T-tests were done on {n_configs} config-level aggregates
- These points share the same queries → NOT independent observations
- Degrees of freedom were incorrectly inflated

Current approach is VALID because:
- We have {n_queries} paired observations (same query, different systems)
- Wilcoxon does not assume normality (appropriate for bounded [0,1] metrics)
- Multiple comparison correction controls family-wise error rate
"""
    
    warnings = []
    if n_queries < 30:
        warnings.append(f"⚠️ Small sample size (n={n_queries}). Results may have low power.")
    
    return {
        "comparisons": comparisons,
        "methodology": methodology,
        "warnings": warnings,
        "n_queries": n_queries,
        "n_configs": n_configs,
    }


def format_analysis_report(analysis: Dict[str, Any]) -> str:
    """Format analysis results as a readable report."""
    if "error" in analysis:
        return f"Error: {analysis['error']}"
    
    lines = []
    lines.append("=" * 80)
    lines.append("RIGOROUS STATISTICAL ANALYSIS")
    lines.append("=" * 80)
    lines.append(analysis["methodology"])
    
    if analysis["warnings"]:
        lines.append("\nWarnings:")
        for w in analysis["warnings"]:
            lines.append(f"  {w}")
    
    lines.append("\n" + "=" * 80)
    lines.append("COMPARISON: Simple RAG vs Agentic RAG")
    lines.append("=" * 80)
    
    for comp in analysis["comparisons"]:
        lines.append(f"\n📊 {comp.metric.upper()}")
        lines.append(f"   Simple:  {comp.mean_a:.4f} [{comp.bootstrap_ci_a.ci_lower:.4f}, {comp.bootstrap_ci_a.ci_upper:.4f}]")
        lines.append(f"   Agentic: {comp.mean_b:.4f} [{comp.bootstrap_ci_b.ci_lower:.4f}, {comp.bootstrap_ci_b.ci_upper:.4f}]")
        lines.append(f"   Diff:    {comp.paired_test.mean_diff:+.4f}")
        lines.append(f"   Wilcoxon p-value: {comp.paired_test.p_value:.4f}")
        lines.append(f"   Corrected p-value (Holm): {comp.corrected_p_value:.4f}")
        lines.append(f"   Effect size ({comp.paired_test.effect_size_name}): {comp.paired_test.effect_size:.3f}")
        
        if comp.corrected_significant:
            lines.append(f"   ✓ Significant after multiple comparison correction")
        else:
            lines.append(f"   ✗ NOT significant after correction")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Run analysis from command line
    results_dir = Path("results")
    analysis = run_full_analysis(results_dir)
    print(format_analysis_report(analysis))
