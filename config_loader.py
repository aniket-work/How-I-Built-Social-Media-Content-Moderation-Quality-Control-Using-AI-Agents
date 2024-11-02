import yaml
from pathlib import Path


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: Path) -> dict:
    """Load prompts from YAML file."""
    with open(prompts_path) as f:
        return yaml.safe_load(f)