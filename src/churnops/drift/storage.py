"""Persistence helpers for drift baselines, live windows, and event logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churnops.config import Settings
from churnops.drift.models import DriftMonitoringState


class DriftStore:
    """Persist rolling observations, state, and events for one monitored model."""

    def __init__(self, settings: Settings, monitor_key: str) -> None:
        self._settings = settings
        self.monitor_key = monitor_key
        self.base_dir = settings.drift.storage_dir / monitor_key
        self.events_dir = self.base_dir / "events"
        self.window_path = self.base_dir / "observation_window.joblib"
        self.state_path = self.base_dir / "state.json"

    def append_observations(
        self,
        feature_frame: pd.DataFrame,
        window_size: int,
    ) -> pd.DataFrame:
        """Append new observations and keep only the configured rolling window."""

        self.base_dir.mkdir(parents=True, exist_ok=True)
        existing_window = self.load_observation_window()
        combined_window = pd.concat([existing_window, feature_frame], ignore_index=True)
        combined_window = combined_window.tail(window_size).reset_index(drop=True)
        joblib.dump(combined_window, self.window_path)
        return combined_window

    def load_observation_window(self) -> pd.DataFrame:
        """Load the current rolling observation window."""

        if not self.window_path.exists():
            return pd.DataFrame()
        return joblib.load(self.window_path)

    def load_state(self, model_source: str) -> DriftMonitoringState:
        """Load the persisted drift monitor state or return a fresh state."""

        if not self.state_path.exists():
            return DriftMonitoringState(
                monitor_key=self.monitor_key,
                model_source=model_source,
            )
        with self.state_path.open("r", encoding="utf-8") as state_file:
            return DriftMonitoringState.from_payload(json.load(state_file))

    def save_state(self, state: DriftMonitoringState) -> None:
        """Persist the current drift monitor state."""

        self.base_dir.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as state_file:
            json.dump(state.to_payload(), state_file, indent=2, sort_keys=True)

    def write_event(self, event_id: str, payload: dict[str, Any]) -> Path:
        """Persist a drift event payload to the event log directory."""

        self.events_dir.mkdir(parents=True, exist_ok=True)
        event_path = self.events_dir / f"{event_id}.json"
        with event_path.open("w", encoding="utf-8") as event_file:
            json.dump(payload, event_file, indent=2, sort_keys=True)
        return event_path
