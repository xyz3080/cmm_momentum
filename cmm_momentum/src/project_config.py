from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_project_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read config.yaml") from exc

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def config_section(name: str, path: Path = CONFIG_PATH) -> dict[str, Any]:
    section = load_project_config(path).get(name, {})
    return section if isinstance(section, dict) else {}
