"""Configuration loading for the freight-rate prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML configuration and identify the repository root.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The parsed configuration mapping and repository root path.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML document is not a mapping.
    """
    config_path = path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    project_root = config_path.parent.parent
    return config, project_root


def resolve_path(project_root: Path, configured_path: str) -> Path:
    """Resolve a configured path against the repository root.

    Args:
        project_root: Repository root directory.
        configured_path: Absolute path or path relative to the repository root.

    Returns:
        The absolute configured path, or the unchanged absolute path supplied.
    """
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path
