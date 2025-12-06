"""
State tracking and run directory helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RunRecord:
    run_dir: str
    timestamp: str
    contexts_sent: int


class StateTracker:
    """Track processed conversations and run history."""

    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                raise SystemExit(
                    f"State file {self.path} is corrupted; delete or fix it."
                )
        return {
            "processed_files": {},
            "seen_contexts": [],
            "last_run": None,
            "run_history": [],
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def is_file_processed(self, file_path: Path) -> bool:
        """Check if a conversation file has been processed."""
        return str(file_path) in self.data.get("processed_files", {})

    def mark_file_processed(
        self, file_path: Path, cards_generated: int = 0
    ) -> None:
        """Mark a file as processed."""
        if "processed_files" not in self.data:
            self.data["processed_files"] = {}
        self.data["processed_files"][str(file_path)] = {
            "processed_at": datetime.now().isoformat(),
            "cards_generated": cards_generated,
        }

    def get_seen_context_ids(self) -> set[str]:
        """Get set of previously seen context IDs."""
        return set(self.data.get("seen_contexts", []))

    def add_context_ids(self, context_ids: List[str]) -> None:
        """Add new context IDs to the seen list."""
        seen = self.get_seen_context_ids()
        seen.update(context_ids)
        self.data["seen_contexts"] = list(seen)

    def record_run(self, run_dir: Path, contexts_sent: int) -> None:
        """Record a run in history."""
        if "run_history" not in self.data:
            self.data["run_history"] = []
        self.data["run_history"].append(
            {
                "run_dir": str(run_dir),
                "timestamp": datetime.now().isoformat(),
                "contexts_sent": contexts_sent,
            }
        )
        self.data["last_run"] = datetime.now().isoformat()


def ensure_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


__all__ = ["StateTracker", "ensure_run_dir"]
