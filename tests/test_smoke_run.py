"""Deterministic benchmark smoke test for CI."""

import asyncio
import json

import yaml


def test_benchmark_smoke_with_fixture(monkeypatch, tmp_path):
    """Run a tiny benchmark orchestration smoke test without network or API calls."""
    from ragbench import run_experiment

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fixture_queries = {
        "q1": "What is the main claim?",
        "q2": "Which paper supports the result?",
    }
    fixture_qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
    }
    results_dir = tmp_path / "results"

    async def fake_run_single_config(config, queries, qrels, semaphore, results_dir, rate_limiter):
        async with semaphore:
            run_dir = results_dir / "runs" / config.__hash__()
            run_dir.mkdir(parents=True, exist_ok=True)

            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

            with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for query_id, query_text in queries.items():
                    handle.write(
                        json.dumps(
                            {
                                "query_id": query_id,
                                "query": query_text,
                                "response": "fixture answer",
                            }
                        )
                        + "\n"
                    )

            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "successful_queries": len(queries),
                        "error_count": 0,
                        "num_queries": len(queries),
                        "elapsed_seconds": 0.01,
                        "avg_time_per_query": 0.005,
                    },
                    handle,
                    indent=2,
                )

            return {
                "config_hash": config.__hash__(),
                "status": "success",
                "num_queries": len(queries),
                "elapsed": 0.01,
            }

    monkeypatch.setattr(run_experiment, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(run_experiment, "load_beir_queries", lambda _path: fixture_queries)
    monkeypatch.setattr(run_experiment, "load_beir_qrels", lambda _path: fixture_qrels)
    monkeypatch.setattr(run_experiment, "run_single_config", fake_run_single_config)

    results = asyncio.run(
        run_experiment.run_benchmark_async(
            dataset="scifact",
            max_configs=2,
            max_queries=2,
            dry_run=False,
            resume=False,
        )
    )

    assert len(results) == 2
    assert all(result["status"] == "success" for result in results)
    assert (results_dir / "selected_queries.json").exists()
    assert len(list((results_dir / "runs").iterdir())) == 2
