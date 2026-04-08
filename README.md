# RAGBench-12x

RAGBench-12x is a reproducible benchmark harness for comparing 12 RAG variants across three public axes:

- orchestration: `simple`, `agentic`
- retrieval: `dense`, `keyword`, `hybrid`
- reranking: `no_rerank`, `rerank`

The benchmark grid is defined in [`configs/base.yaml`](configs/base.yaml) and [`configs/axes.yaml`](configs/axes.yaml). Those YAML files are now the source of truth for both the runtime and the public tests.

## What is stable today

- `load_config_from_yaml(...)` expands the public 12x grid deterministically.
- `AppConfig` exposes a real `results_dir` and creates the expected directories.
- `pytest`, `ruff`, and `mypy` are wired for local use and CI.
- CI runs a deterministic smoke benchmark using fixture data and no external API calls.

## What CI does not claim

The CI smoke test validates benchmark orchestration, config expansion, and report generation. It does not download BEIR, build production indexes, or call external model providers. Full benchmark runs still require local credentials and real data.

## Quick start

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run mypy
```

## Run the benchmark locally

1. Create a `.env` file with the providers you actually want to use.

```env
OPENROUTER_API_KEY=sk-or-v1-xxxx
OPENAI_API_KEY=sk-xxxx
COHERE_API_KEY=xxxx
```

2. Download a BEIR dataset.

```bash
uv run ragbench download --dataset scifact
```

3. Build indexes.

```bash
uv run ragbench index --dataset scifact
```

4. Run a benchmark subset first.

```bash
uv run ragbench benchmark --max-configs 2 --max-queries 10 --parallel
```

5. Generate a report from `results/`.

```bash
uv run ragbench report --results-dir results --output results/report.md
```

## Public quality gates

```bash
uv run pytest
uv run ruff check src tests
uv run mypy
```

## Repository structure

```text
configs/         benchmark defaults and axes
data/            raw datasets and indexes
results/         generated benchmark artifacts
src/ragbench/    benchmark runtime, CLI, reporting
tests/           deterministic smoke and report tests
```

## Notes on reproducibility

- The benchmark grid comes from YAML, not duplicated constants.
- Use `--max-configs` and `--max-queries` for controlled subset runs.
- The smoke test in CI is intentionally offline and fast.
- Full scientific claims should be tied to published result artifacts, not assumed from the repository state alone.
