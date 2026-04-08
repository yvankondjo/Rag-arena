"""Tests for the public benchmark configuration surface."""

from ragbench.config import CONFIG_DIR, RESULTS_DIR, AppConfig, load_config_from_yaml


def test_config_loading():
    """The published YAML grid should expand to the advertised 12 runs."""
    base_yaml = CONFIG_DIR / "base.yaml"
    axes_yaml = CONFIG_DIR / "axes.yaml"

    assert base_yaml.exists(), f"base.yaml not found at {base_yaml}"
    assert axes_yaml.exists(), f"axes.yaml not found at {axes_yaml}"

    configs = load_config_from_yaml(base_yaml, axes_yaml)

    assert len(configs) == 12, f"Expected 12 configs, got {len(configs)}"

    run_names = [cfg.get_run_name() for cfg in configs]
    assert len(set(run_names)) == 12, "Run names should be unique"
    assert all(cfg.dataset == "scifact" for cfg in configs)


def test_app_config_uses_results_dir_alias(monkeypatch):
    """AppConfig should expose the public results_dir expected by scripts and tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    config = AppConfig()

    assert config.results_dir == RESULTS_DIR
    assert config.log_file == RESULTS_DIR / "logs" / "rag_logs.jsonl"
