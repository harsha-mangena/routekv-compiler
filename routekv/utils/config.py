"""YAML config loader with dot-access."""
import yaml
from pathlib import Path
from typing import Any


class Config(dict):
    """Dictionary subclass supporting dot-access notation."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a Config object."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(raw)
