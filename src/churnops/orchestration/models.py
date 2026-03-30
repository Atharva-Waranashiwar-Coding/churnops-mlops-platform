"""Models for orchestrated training execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class TrainingExecutionContext:
    """Filesystem and identity details for an orchestrated training run."""

    run_id: str
    workspace_dir: Path
    orchestrator: str
    orchestrator_run_id: str | None = None
    logical_date_utc: str | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_payload(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation for orchestration systems."""

        return {
            "run_id": self.run_id,
            "workspace_dir": str(self.workspace_dir),
            "orchestrator": self.orchestrator,
            "orchestrator_run_id": self.orchestrator_run_id,
            "logical_date_utc": self.logical_date_utc,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, str | None]) -> TrainingExecutionContext:
        """Rehydrate a training execution context from an orchestration payload."""

        return cls(
            run_id=str(payload["run_id"]),
            workspace_dir=Path(str(payload["workspace_dir"])),
            orchestrator=str(payload["orchestrator"]),
            orchestrator_run_id=payload.get("orchestrator_run_id"),
            logical_date_utc=payload.get("logical_date_utc"),
            created_at_utc=str(
                payload.get("created_at_utc") or datetime.now(timezone.utc).isoformat()
            ),
        )
