"""Load the YAML config once and expose resolved paths."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    """Resolve a repo-relative path to an absolute Path."""
    return (ROOT / rel).resolve()
