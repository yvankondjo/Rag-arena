"""Test configuration loading and client creation."""

from pathlib import Path
from ragbench.config import load_config_from_yaml, AppConfig, CONFIG_DIR

def test_config_loading():
    """Test loading configurations from YAML."""
    base_yaml = CONFIG_DIR / "base.yaml"
    axes_yaml = CONFIG_DIR / "axes.yaml"
    
    assert base_yaml.exists(), f"base.yaml not found at {base_yaml}"
    assert axes_yaml.exists(), f"axes.yaml not found at {axes_yaml}"
    
    configs = load_config_from_yaml(base_yaml, axes_yaml)

    # Should have 2 * 3 * 2 = 12 configurations
    assert len(configs) == 12, f"Expected 12 configs, got {len(configs)}"

    # Check unique combinations
    run_names = [cfg.get_run_name() for cfg in configs]
    assert len(set(run_names)) == 12, "Run names should be unique"
    
    print("✓ Configuration loading test passed")
    print(f"✓ Generated {len(configs)} configurations:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.get_run_name()}")

if __name__ == "__main__":
    test_config_loading()
