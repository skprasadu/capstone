from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load the assistant configuration from YAML."""

    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open() as handle:
        return yaml.safe_load(handle)
