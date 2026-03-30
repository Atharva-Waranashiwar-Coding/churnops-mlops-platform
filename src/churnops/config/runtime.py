"""Runtime configuration overrides for local training execution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from churnops.config.models import Settings


def apply_runtime_overrides(
    settings: Settings,
    data_path: str | Path | None = None,
) -> Settings:
    """Apply CLI-level runtime overrides without mutating the loaded config object."""

    if data_path is None:
        return settings

    resolved_data_path = Path(data_path).expanduser()
    if not resolved_data_path.is_absolute():
        resolved_data_path = (settings.project.root_dir / resolved_data_path).resolve()

    return replace(
        settings,
        data=replace(
            settings.data,
            raw_data_path=resolved_data_path,
        ),
    )
