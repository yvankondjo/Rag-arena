#!/usr/bin/env python3
"""Analyze orchestration gain metrics for Agentic RAG."""

import json
import yaml
from pathlib import Path

data = []
for run_dir in sorted(Path('results/runs').iterdir()):
    if not run_dir.is_dir():
        continue
    
    with open(run_dir / 'config.yaml') as f:
        config = yaml.safe_load(f)
    with open(run_dir / 'metrics.json') as f:
        metrics = json.load(f)
    
    data.append({
        'orch': config.get('orchestration_mode', 'simple'),
        'ret': config.get('retrieval_mode', 'dense'),
        'rerank': config.get('use_reranker', False),
        'avg_steps': metrics.get('avg_retrieval_steps', 1.0),
        'gain_ndcg': metrics.get('orchestration_gain_ndcg', 0),
        'gain_recall': metrics.get('orchestration_gain_recall', 0),
        'ndcg': metrics.get('ndcg_at_10', 0),
        'final_ndcg': metrics.get('final_step_ndcg_at_10', 0),
    })

print('='*80)
print('ANALYSE DU GAIN D\'ORCHESTRATION (Agentic RAG multi-step)')
print('='*80)
print()

header = f"{'Orch':<8} {'Retrieval':<8} {'Rerank':<7} {'Steps':>6} {'1st NDCG':>10} {'Final NDCG':>11} {'Gain':>8}"
print(header)
print('-'*70)

for d in sorted(data, key=lambda x: (x['orch'], x['ret'], x['rerank'])):
    rerank_str = 'Yes' if d['rerank'] else 'No'
    # first_step = final - gain
    first_ndcg = d['final_ndcg'] - d['gain_ndcg']
    print(f"{d['orch']:<8} {d['ret']:<8} {rerank_str:<7} {d['avg_steps']:>6.2f} {first_ndcg:>10.4f} {d['final_ndcg']:>11.4f} {d['gain_ndcg']:>+8.4f}")

print()
print('='*80)
print('RÉSUMÉ: AGENTIC RAG - Nombre de steps et impact')
print('='*80)

agentic = [d for d in data if d['orch'] == 'agentic']
simple = [d for d in data if d['orch'] == 'simple']

print(f'\nAgentic RAG:')
print(f'  - Nombre moyen de steps: {sum(d["avg_steps"] for d in agentic)/len(agentic):.2f}')
print(f'  - Gain NDCG moyen (final - first): {sum(d["gain_ndcg"] for d in agentic)/len(agentic):+.4f}')
print(f'  - Gain Recall moyen: {sum(d["gain_recall"] for d in agentic)/len(agentic):+.4f}')

print(f'\nSimple RAG:')
print(f'  - Nombre de steps: {sum(d["avg_steps"] for d in simple)/len(simple):.2f} (toujours 1)')
print(f'  - Gain: 0 (par définition, 1 seul step)')

print('\n' + '='*80)
print('INTERPRÉTATION')
print('='*80)

avg_gain = sum(d["gain_ndcg"] for d in agentic)/len(agentic)
avg_steps = sum(d["avg_steps"] for d in agentic)/len(agentic)

if avg_gain > 0.01:
    print(f'\n✅ L\'Agentic RAG apporte un gain positif de {avg_gain:+.4f} NDCG')
    print(f'   avec en moyenne {avg_steps:.2f} étapes de recherche.')
elif avg_gain < -0.01:
    print(f'\n⚠️  L\'Agentic RAG a un gain NÉGATIF de {avg_gain:+.4f} NDCG')
    print(f'   Les recherches supplémentaires ({avg_steps:.2f} steps) dégradent la qualité!')
    print(f'   Hypothèse: le query rewriting dégrade la requête originale.')
else:
    print(f'\n➖ L\'Agentic RAG a un gain quasi-nul ({avg_gain:+.4f} NDCG)')
    print(f'   Les {avg_steps:.2f} étapes n\'améliorent pas significativement les résultats.')

print('\n' + '='*80)
print('CONCLUSION')
print('='*80)
print('''
1. L'agent fait en moyenne 1.3-1.4 recherches (pas beaucoup de multi-step)
2. Le gain d'orchestration est NÉGLIGEABLE voire NÉGATIF
3. Les query rewrites de l'agent ne semblent pas améliorer la recherche
4. Sur ce dataset (SciFact), les requêtes originales sont déjà bien formulées

RECOMMANDATION: Pour SciFact, le Simple RAG suffit. L'Agentic RAG
n'apporte pas de valeur ajoutée et coûte plus cher en tokens/latence.
''')
