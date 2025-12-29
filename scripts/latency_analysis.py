#!/usr/bin/env python
"""Analyse des latences et trade-off qualité/latence."""

import json
import yaml
from pathlib import Path
from collections import defaultdict

def main():
    data = []
    for run_dir in sorted(Path('results/runs').iterdir()):
        if not run_dir.is_dir():
            continue
        
        with open(run_dir / 'config.yaml') as f:
            config = yaml.safe_load(f)
        with open(run_dir / 'metrics.json') as f:
            metrics = json.load(f)
        
        orch = config.get('orchestration_mode', 'simple')
        ret = config.get('retrieval_mode', 'dense')
        rerank = config.get('use_reranker', False)
        rerank_str = '+rerank' if rerank else ''
        label = f'{orch}/{ret}{rerank_str}'
        
        retrieval_lat = metrics.get('retrieval_latency_mean', 0) or 0
        rerank_lat = metrics.get('reranking_latency_mean', 0) or 0
        gen_lat = metrics.get('generation_latency_mean', 0) or 0
        total_latency = retrieval_lat + rerank_lat + gen_lat
        
        data.append({
            'label': label,
            'ndcg': metrics.get('ndcg_at_10', 0),
            'recall': metrics.get('recall_at_5', 0),
            'mrr': metrics.get('mrr_at_10', 0),
            'faith': metrics.get('faithfulness', 0),
            'relevancy': metrics.get('answer_relevancy', 0),
            'retrieval_lat': retrieval_lat,
            'rerank_lat': rerank_lat,
            'gen_lat': gen_lat,
            'total_lat': total_latency,
            'reranker': rerank,
            'orch': orch,
            'ret': ret,
        })

    # Sort by overall score
    for d in data:
        d['overall'] = (d['ndcg'] + d['recall'] + d['mrr'] + d['faith'] + d['relevancy']) / 5

    data.sort(key=lambda x: x['overall'], reverse=True)

    print('RANKING AVEC LATENCE')
    print('=' * 120)
    header = f"{'Rank':<5} {'Configuration':<25} {'NDCG@10':>8} {'Recall@5':>9} {'MRR@10':>8} {'Faith':>8} {'Ans.Rel':>8} {'Latency':>8} {'Overall':>8}"
    print(header)
    print('-' * 120)

    for i, d in enumerate(data, 1):
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
        print(f"{medal:<5} {d['label']:<25} {d['ndcg']:>8.3f} {d['recall']:>9.3f} {d['mrr']:>8.3f} {d['faith']:>8.3f} {d['relevancy']:>8.3f} {d['total_lat']:>7.2f}s {d['overall']:>8.3f}")

    print()
    print('IMPACT DU RERANKER SUR LA LATENCE')
    print('=' * 90)

    # Group by (orch, ret) and compare with/without reranker
    groups = defaultdict(dict)
    for d in data:
        key = (d['orch'], d['ret'])
        if d['reranker']:
            groups[key]['with'] = d
        else:
            groups[key]['without'] = d

    header2 = f"{'Config':<20} {'Sans Rerank':>12} {'Avec Rerank':>12} {'Δ Latence':>12} {'Δ NDCG':>10} {'Trade-off':>15}"
    print(header2)
    print('-' * 90)

    tradeoffs = []
    for key, g in sorted(groups.items()):
        if 'with' in g and 'without' in g:
            without = g['without']
            with_r = g['with']
            delta_lat = with_r['total_lat'] - without['total_lat']
            delta_ndcg = with_r['ndcg'] - without['ndcg']
            # Trade-off: gain NDCG per second of latency
            tradeoff = (delta_ndcg * 100) / delta_lat if delta_lat > 0 else 0
            config = f"{key[0]}/{key[1]}"
            print(f"{config:<20} {without['total_lat']:>11.2f}s {with_r['total_lat']:>11.2f}s {delta_lat:>+11.2f}s {delta_ndcg:>+10.3f} {tradeoff:>+14.2f}%/s")
            tradeoffs.append({
                'config': config,
                'delta_lat': delta_lat,
                'delta_ndcg': delta_ndcg,
                'tradeoff': tradeoff
            })

    print()
    print('RÉSUMÉ LATENCE')
    print('=' * 60)
    
    # Average latencies
    rerank_configs = [d for d in data if d['reranker']]
    no_rerank_configs = [d for d in data if not d['reranker']]
    
    avg_lat_with = sum(d['total_lat'] for d in rerank_configs) / len(rerank_configs) if rerank_configs else 0
    avg_lat_without = sum(d['total_lat'] for d in no_rerank_configs) / len(no_rerank_configs) if no_rerank_configs else 0
    avg_rerank_time = sum(d['rerank_lat'] for d in rerank_configs) / len(rerank_configs) if rerank_configs else 0
    
    avg_ndcg_with = sum(d['ndcg'] for d in rerank_configs) / len(rerank_configs) if rerank_configs else 0
    avg_ndcg_without = sum(d['ndcg'] for d in no_rerank_configs) / len(no_rerank_configs) if no_rerank_configs else 0
    
    print(f"Latence moyenne SANS reranker:  {avg_lat_without:.2f}s")
    print(f"Latence moyenne AVEC reranker:  {avg_lat_with:.2f}s")
    print(f"Temps moyen du reranking:       {avg_rerank_time:.2f}s")
    print(f"Surcoût latence reranker:       +{avg_lat_with - avg_lat_without:.2f}s (+{((avg_lat_with/avg_lat_without)-1)*100:.1f}%)")
    print()
    print(f"NDCG@10 moyen SANS reranker:    {avg_ndcg_without:.3f}")
    print(f"NDCG@10 moyen AVEC reranker:    {avg_ndcg_with:.3f}")
    print(f"Gain NDCG du reranker:          +{avg_ndcg_with - avg_ndcg_without:.3f} (+{((avg_ndcg_with/avg_ndcg_without)-1)*100:.1f}%)")
    
    print()
    print('=' * 60)
    avg_tradeoff = sum(t['tradeoff'] for t in tradeoffs) / len(tradeoffs) if tradeoffs else 0
    print(f"Trade-off moyen: {avg_tradeoff:.2f}% NDCG par seconde de latence")
    
    # Markdown table for README
    print()
    print('=' * 60)
    print('TABLEAU MARKDOWN POUR README')
    print('=' * 60)
    print()
    print("| Rank | Configuration | NDCG@10 | Recall@5 | MRR@10 | Faith. | Ans.Rel. | Latency | Overall |")
    print("|------|---------------|---------|----------|--------|--------|----------|---------|---------|")
    for i, d in enumerate(data, 1):
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else str(i)
        print(f"| {medal} | {d['label']} | {d['ndcg']:.3f} | {d['recall']:.3f} | {d['mrr']:.3f} | {d['faith']:.3f} | {d['relevancy']:.3f} | {d['total_lat']:.2f}s | {d['overall']:.3f} |")

    print()
    print("TABLEAU TRADE-OFF POUR README")
    print()
    print("| Configuration | Sans Rerank | Avec Rerank | Δ Latence | Δ NDCG | Trade-off |")
    print("|---------------|-------------|-------------|-----------|--------|-----------|")
    for t in tradeoffs:
        without = groups[(t['config'].split('/')[0], t['config'].split('/')[1])]['without']
        with_r = groups[(t['config'].split('/')[0], t['config'].split('/')[1])]['with']
        print(f"| {t['config']} | {without['total_lat']:.2f}s | {with_r['total_lat']:.2f}s | +{t['delta_lat']:.2f}s | +{t['delta_ndcg']:.3f} | {t['tradeoff']:.2f}%/s |")


if __name__ == '__main__':
    main()
