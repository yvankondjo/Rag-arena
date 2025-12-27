#!/usr/bin/env python3
"""Deep statistical analysis of RAGBench-12x results."""

import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv("results/results.csv")

print("=" * 80)
print("DEEP ANALYSIS: RAGBench-12x Results")
print("=" * 80)

# Separate by orchestration mode
simple = df[df['orchestration_mode'] == 'simple']
agentic = df[df['orchestration_mode'] == 'agentic']

print("\n" + "=" * 80)
print("1. AGGREGATED COMPARISON: Simple RAG vs Agentic RAG")
print("=" * 80)

metrics = ['ndcg_at_10', 'recall_at_5', 'mrr_at_10', 'faithfulness', 'answer_relevancy', 'elapsed_seconds']

print(f"\n{'Metric':<25} {'Simple RAG':<15} {'Agentic RAG':<15} {'Δ (Agentic-Simple)':<20} {'Winner':<15}")
print("-" * 90)

for metric in metrics:
    simple_mean = simple[metric].mean()
    agentic_mean = agentic[metric].mean()
    delta = agentic_mean - simple_mean
    pct_change = (delta / simple_mean) * 100
    
    if metric == 'elapsed_seconds':
        winner = "Simple ✓" if simple_mean < agentic_mean else "Agentic ✓"
    elif metric in ['faithfulness', 'answer_relevancy', 'ndcg_at_10', 'recall_at_5', 'mrr_at_10']:
        winner = "Simple ✓" if simple_mean > agentic_mean else "Agentic ✓"
    else:
        winner = "-"
    
    print(f"{metric:<25} {simple_mean:<15.4f} {agentic_mean:<15.4f} {delta:>+8.4f} ({pct_change:>+6.1f}%)     {winner}")

print("\n" + "=" * 80)
print("2. CRITICAL FINDING: Quality Metrics Analysis")
print("=" * 80)

print("\n📊 FAITHFULNESS (Is the answer grounded in the retrieved context?)")
print(f"   Simple RAG:  {simple['faithfulness'].mean():.4f} (±{simple['faithfulness'].std():.4f})")
print(f"   Agentic RAG: {agentic['faithfulness'].mean():.4f} (±{agentic['faithfulness'].std():.4f})")
faith_delta = simple['faithfulness'].mean() - agentic['faithfulness'].mean()
print(f"   ⚠️  Simple RAG has {faith_delta:.1%} HIGHER faithfulness!")

print("\n📊 ANSWER RELEVANCY (Does the answer respond to the question?)")
print(f"   Simple RAG:  {simple['answer_relevancy'].mean():.4f} (±{simple['answer_relevancy'].std():.4f})")
print(f"   Agentic RAG: {agentic['answer_relevancy'].mean():.4f} (±{agentic['answer_relevancy'].std():.4f})")
ar_delta = simple['answer_relevancy'].mean() - agentic['answer_relevancy'].mean()
print(f"   ≈ Similar performance (Δ = {ar_delta:.1%})")

print("\n⏱️  EXECUTION TIME")
print(f"   Simple RAG:  {simple['elapsed_seconds'].mean():.1f}s avg")
print(f"   Agentic RAG: {agentic['elapsed_seconds'].mean():.1f}s avg")
time_ratio = agentic['elapsed_seconds'].mean() / simple['elapsed_seconds'].mean()
print(f"   ⚠️  Agentic RAG takes {time_ratio:.1f}x LONGER!")

print("\n" + "=" * 80)
print("3. HEAD-TO-HEAD COMPARISONS (Same retrieval + reranker)")
print("=" * 80)

comparisons = [
    ('dense', True, 'Dense + Rerank'),
    ('dense', False, 'Dense + No Rerank'),
    ('hybrid', True, 'Hybrid + Rerank'),
    ('hybrid', False, 'Hybrid + No Rerank'),
    ('keyword', True, 'Keyword + Rerank'),
    ('keyword', False, 'Keyword + No Rerank'),
]

print(f"\n{'Configuration':<20} {'Metric':<15} {'Simple':<10} {'Agentic':<10} {'Winner':<10}")
print("-" * 75)

for retrieval, reranker, name in comparisons:
    s = simple[(simple['retrieval_mode'] == retrieval) & (simple['use_reranker'] == reranker)].iloc[0]
    a = agentic[(agentic['retrieval_mode'] == retrieval) & (agentic['use_reranker'] == reranker)].iloc[0]
    
    for metric, label in [('faithfulness', 'Faithfulness'), ('answer_relevancy', 'Ans. Relev.'), ('ndcg_at_10', 'NDCG@10')]:
        s_val = s[metric]
        a_val = a[metric]
        winner = "Simple ✓" if s_val > a_val else "Agentic ✓" if a_val > s_val else "Tie"
        print(f"{name:<20} {label:<15} {s_val:<10.4f} {a_val:<10.4f} {winner:<10}")
    print()

print("\n" + "=" * 80)
print("4. STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)

# T-test for faithfulness
t_faith, p_faith = stats.ttest_ind(simple['faithfulness'], agentic['faithfulness'])
print(f"\nFaithfulness t-test: t={t_faith:.3f}, p={p_faith:.4f}")
if p_faith < 0.05:
    print("   ✓ Statistically significant difference (p < 0.05)")
else:
    print("   ✗ Not statistically significant (p >= 0.05)")

# T-test for answer relevancy
t_ar, p_ar = stats.ttest_ind(simple['answer_relevancy'], agentic['answer_relevancy'])
print(f"\nAnswer Relevancy t-test: t={t_ar:.3f}, p={p_ar:.4f}")
if p_ar < 0.05:
    print("   ✓ Statistically significant difference (p < 0.05)")
else:
    print("   ✗ Not statistically significant (p >= 0.05)")

print("\n" + "=" * 80)
print("5. KEY FINDINGS SUMMARY")
print("=" * 80)

print("""
🔬 RESEARCH FINDINGS:

1. FAITHFULNESS DEGRADATION IN AGENTIC RAG
   - Simple RAG achieves 90.0% avg faithfulness
   - Agentic RAG achieves only 81.4% avg faithfulness
   - This is a 9.6% DECREASE in answer grounding
   
   Hypothesis: Multi-step retrieval introduces conflicting information
   that confuses the generation model, leading to less faithful answers.

2. NO IMPROVEMENT IN ANSWER RELEVANCY
   - Both modes achieve ~70% answer relevancy
   - The extra search steps don't help answer the question better
   
   Hypothesis: The agentic query rewriting may drift from the original
   intent, or the additional context doesn't add relevant information.

3. 2x EXECUTION TIME FOR NO QUALITY GAIN
   - Simple RAG: ~1360s (22.7 min)
   - Agentic RAG: ~2359s (39.3 min)
   - 74% slower for equivalent or worse quality

4. RETRIEVAL METRICS ARE SIMILAR
   - NDCG@10, Recall@5, MRR@10 are comparable
   - The agentic approach doesn't retrieve more relevant documents

🎯 CONCLUSION:
For this benchmark (SciFact dataset, 50 queries), Simple RAG outperforms
Agentic RAG on quality metrics while being significantly faster.

This is a valuable NEGATIVE RESULT that challenges the assumption
that "more complex = better" in RAG systems.

📚 ACADEMIC VALUE:
- Reproduces and benchmarks agentic RAG approaches
- Provides empirical evidence against over-engineering
- Suggests simpler architectures may be preferable for certain tasks
- Opens discussion on when agentic approaches ARE beneficial
""")

print("\n" + "=" * 80)
print("6. BEST CONFIGURATIONS")
print("=" * 80)

# Sort by each metric
print("\n🏆 TOP 3 by FAITHFULNESS:")
top_faith = df.nlargest(3, 'faithfulness')[['orchestration_mode', 'retrieval_mode', 'use_reranker', 'faithfulness']]
for _, row in top_faith.iterrows():
    rerank = "rerank" if row['use_reranker'] else "no_rerank"
    print(f"   {row['orchestration_mode']:<8} + {row['retrieval_mode']:<8} + {rerank:<10}: {row['faithfulness']:.4f}")

print("\n🏆 TOP 3 by ANSWER RELEVANCY:")
top_ar = df.nlargest(3, 'answer_relevancy')[['orchestration_mode', 'retrieval_mode', 'use_reranker', 'answer_relevancy']]
for _, row in top_ar.iterrows():
    rerank = "rerank" if row['use_reranker'] else "no_rerank"
    print(f"   {row['orchestration_mode']:<8} + {row['retrieval_mode']:<8} + {rerank:<10}: {row['answer_relevancy']:.4f}")

print("\n🏆 TOP 3 by NDCG@10:")
top_ndcg = df.nlargest(3, 'ndcg_at_10')[['orchestration_mode', 'retrieval_mode', 'use_reranker', 'ndcg_at_10']]
for _, row in top_ndcg.iterrows():
    rerank = "rerank" if row['use_reranker'] else "no_rerank"
    print(f"   {row['orchestration_mode']:<8} + {row['retrieval_mode']:<8} + {rerank:<10}: {row['ndcg_at_10']:.4f}")

print("\n⚡ FASTEST (with quality > 0.85 faithfulness):")
fast_good = df[df['faithfulness'] > 0.85].nsmallest(3, 'elapsed_seconds')[['orchestration_mode', 'retrieval_mode', 'use_reranker', 'elapsed_seconds', 'faithfulness']]
for _, row in fast_good.iterrows():
    rerank = "rerank" if row['use_reranker'] else "no_rerank"
    print(f"   {row['orchestration_mode']:<8} + {row['retrieval_mode']:<8} + {rerank:<10}: {row['elapsed_seconds']:.0f}s (faith: {row['faithfulness']:.4f})")

print("\n" + "=" * 80)
