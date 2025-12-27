"""Latency and cost evaluation metrics."""

from typing import Dict, List, Optional, Any, Tuple
import time
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class LatencyMetrics:
    """Latency evaluation metrics."""
    latency_p50: float
    latency_p95: float
    latency_mean: float
    latency_std: float
    num_queries: int


@dataclass
class TokenUsage:
    """Token usage metrics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class CostMetrics:
    """Cost evaluation metrics."""
    total_cost_usd: float
    cost_per_query_usd: float
    total_tokens: int
    tokens_per_query: float


@contextmanager
def timer():
    """Context manager to measure execution time."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
       
        timer.elapsed = elapsed


def measure_latency(latencies: List[float]) -> LatencyMetrics:
    """Compute latency statistics from a list of latency measurements.

    Args:
        latencies: List of latency measurements in seconds

    Returns:
        LatencyMetrics with p50, p95, mean, std
    """
    if not latencies:
        return LatencyMetrics(0.0, 0.0, 0.0, 0.0, 0)

    latencies_array = np.array(latencies)

    return LatencyMetrics(
        latency_p50=np.percentile(latencies_array, 50),
        latency_p95=np.percentile(latencies_array, 95),
        latency_mean=np.mean(latencies_array),
        latency_std=np.std(latencies_array),
        num_queries=len(latencies),
    )


def estimate_token_usage(
    prompt_text: str,
    completion_text: str,
    model_name: str = "gpt-4o-mini",
) -> TokenUsage:
    """Estimate token usage for a single API call.

    This is a rough estimation. For accurate counts, use the actual
    token counts returned by the LLM API.

    Args:
        prompt_text: Input prompt text
        completion_text: Generated completion text
        model_name: Model name for token estimation

    Returns:
        TokenUsage with estimated counts
    """
    # Rough token estimation: ~4 characters per token
    # This is very approximate and should be replaced with actual API counts
    prompt_tokens = len(prompt_text) // 4
    completion_tokens = len(completion_text) // 4
    total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def estimate_cost(
    token_usages: List[TokenUsage],
    model_name: str = "gpt-4o-mini",
) -> CostMetrics:
    """Estimate cost based on token usage.

    Args:
        token_usages: List of TokenUsage objects
        model_name: Model name for pricing

    Returns:
        CostMetrics with estimated costs
    """
    if not token_usages:
        return CostMetrics(0.0, 0.0, 0, 0.0)

    # Pricing per 1K tokens (approximate, update based on current pricing)
    pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    }

    model_pricing = pricing.get(model_name, pricing["gpt-4o-mini"])

    total_prompt_tokens = sum(usage.prompt_tokens for usage in token_usages)
    total_completion_tokens = sum(usage.completion_tokens for usage in token_usages)
    total_tokens = sum(usage.total_tokens for usage in token_usages)

    prompt_cost = (total_prompt_tokens / 1000) * model_pricing["input"]
    completion_cost = (total_completion_tokens / 1000) * model_pricing["output"]
    total_cost = prompt_cost + completion_cost

    num_queries = len(token_usages)

    return CostMetrics(
        total_cost_usd=total_cost,
        cost_per_query_usd=total_cost / num_queries if num_queries > 0 else 0.0,
        total_tokens=total_tokens,
        tokens_per_query=total_tokens / num_queries if num_queries > 0 else 0.0,
    )


class LatencyTracker:
    """Utility class to track latencies during experiments."""

    def __init__(self):
        self.latencies: List[float] = []

    def add_measurement(self, latency: float):
        """Add a latency measurement."""
        self.latencies.append(latency)

    def get_metrics(self) -> LatencyMetrics:
        """Get aggregated latency metrics."""
        return measure_latency(self.latencies)

    def reset(self):
        """Reset all measurements."""
        self.latencies.clear()


class TokenTracker:
    """Utility class to track token usage during experiments."""

    def __init__(self):
        self.token_usages: List[TokenUsage] = []

    def add_usage(self, usage: TokenUsage):
        """Add token usage measurement."""
        self.token_usages.append(usage)

    def get_cost_metrics(self, model_name: str = "gpt-4o-mini") -> CostMetrics:
        """Get aggregated cost metrics."""
        return estimate_cost(self.token_usages, model_name)

    def reset(self):
        """Reset all measurements."""
        self.token_usages.clear()


def benchmark_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Benchmark a function call and return result with latency.

    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (function_result, latency_seconds)
    """
    with timer():
        result = func(*args, **kwargs)

    return result, timer.elapsed
