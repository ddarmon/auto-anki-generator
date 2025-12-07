"""
State tracking and run directory helpers.

State schema v2 introduces conversation-level tracking alongside per-turn tracking.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Current state schema version
STATE_VERSION = 2


def _generate_migrated_conversation_id(source_path: str) -> str:
    """Generate a conversation_id for migrated contexts.

    Uses source_path + 'migrated' marker since we don't have original timestamps.
    """
    return sha256(f"{source_path}:migrated".encode("utf-8")).hexdigest()


def _convert_contexts_to_conversations(
    contexts: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert a list of per-turn contexts to conversation format.

    Groups contexts by source_path and creates conversation objects.

    Returns:
        Tuple of (conversations_list, conversation_ids)
    """
    # Group contexts by source_path
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for ctx in contexts:
        source_path = ctx.get("source_path", "unknown")
        if source_path not in by_source:
            by_source[source_path] = []
        by_source[source_path].append(ctx)

    conversations = []
    conversation_ids = []

    for source_path, ctx_list in by_source.items():
        # Sort by turn_index if available, otherwise by context order
        ctx_list.sort(key=lambda c: c.get("turn_index", 0))

        # Generate conversation_id
        conv_id = _generate_migrated_conversation_id(source_path)
        conversation_ids.append(conv_id)

        # Build turns with turn_index
        turns = []
        for idx, ctx in enumerate(ctx_list):
            turn = dict(ctx)
            turn["turn_index"] = idx
            turn["conversation_id"] = conv_id
            turns.append(turn)

        # Extract metadata from first context
        first_ctx = ctx_list[0]

        # Compute aggregate score
        scores = [c.get("score", 0) for c in ctx_list]
        aggregate_score = sum(scores) / len(scores) if scores else 0

        # Extract key topics from all contexts
        all_terms = []
        for ctx in ctx_list:
            all_terms.extend(ctx.get("key_terms", []))
        key_topics = list(set(all_terms))[:10]  # Dedupe and limit

        # Build conversation object
        conversation = {
            "conversation_id": conv_id,
            "source_path": source_path,
            "source_title": first_ctx.get("source_title"),
            "source_url": first_ctx.get("source_url") or first_ctx.get("url"),
            "turns": turns,
            "total_char_count": sum(
                len(c.get("assistant_answer", "")) for c in ctx_list
            ),
            "aggregate_score": aggregate_score,
            "aggregate_signals": {"migrated_from_v1": True},
            "key_topics": key_topics,
        }
        conversations.append(conversation)

    return conversations, conversation_ids


def migrate_run_artifacts(runs_dir: Path, verbose: bool = False) -> List[str]:
    """Migrate old selected_contexts.json files to selected_conversations.json.

    Scans all run directories and converts per-turn format to conversation format.

    Args:
        runs_dir: Path to auto_anki_runs directory
        verbose: Whether to print progress

    Returns:
        List of all conversation_ids from migrated artifacts
    """
    if not runs_dir.exists():
        return []

    all_conversation_ids = []
    migrated_count = 0

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
            continue

        old_file = run_dir / "selected_contexts.json"
        new_file = run_dir / "selected_conversations.json"

        # Skip if already migrated or no old file
        if new_file.exists() or not old_file.exists():
            continue

        try:
            contexts = json.loads(old_file.read_text())
            if not contexts:
                continue

            conversations, conv_ids = _convert_contexts_to_conversations(contexts)
            all_conversation_ids.extend(conv_ids)

            # Write new format
            new_file.write_text(json.dumps(conversations, indent=2))
            migrated_count += 1

            if verbose:
                print(f"  Migrated {run_dir.name}: {len(contexts)} contexts → {len(conversations)} conversations")

        except (json.JSONDecodeError, KeyError) as e:
            if verbose:
                print(f"  Warning: Failed to migrate {run_dir.name}: {e}")
            continue

    if verbose and migrated_count > 0:
        print(f"  Total: Migrated {migrated_count} run directories")

    return all_conversation_ids


@dataclass
class RunRecord:
    run_dir: str
    timestamp: str
    contexts_sent: int
    conversations_sent: int = 0  # New in v2


class StateTracker:
    """Track processed conversations and run history."""

    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()
        self._migrate_if_needed()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                raise SystemExit(
                    f"State file {self.path} is corrupted; delete or fix it."
                )
        return self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        """Return default state for new installations."""
        return {
            "state_version": STATE_VERSION,
            "processed_files": {},
            "seen_contexts": [],
            "seen_conversations": [],
            "last_run": None,
            "run_history": [],
        }

    def _migrate_if_needed(self) -> None:
        """Auto-migrate state from older versions."""
        current_version = self.data.get("state_version", 1)
        if current_version < STATE_VERSION:
            self._migrate_v1_to_v2()

    def _migrate_v1_to_v2(self) -> None:
        """Migrate from v1 (per-turn only) to v2 (per-turn + per-conversation).

        V2 adds:
        - state_version field
        - seen_conversations list

        This migration also:
        - Converts old selected_contexts.json files to selected_conversations.json
        - Populates seen_conversations from the converted artifacts

        V1 data is preserved for backward compatibility with per-turn tracking.
        """
        print("Migrating state from v1 to v2 (conversation-level processing)...")

        # Backup old state file
        if self.path.exists():
            backup_path = self.path.with_suffix(".json.v1_backup")
            if not backup_path.exists():
                shutil.copy(self.path, backup_path)
                print(f"  Created backup: {backup_path.name}")

        # Add new fields
        self.data["state_version"] = STATE_VERSION
        if "seen_conversations" not in self.data:
            self.data["seen_conversations"] = []

        # Ensure all expected fields exist
        if "processed_files" not in self.data:
            self.data["processed_files"] = {}
        if "seen_contexts" not in self.data:
            self.data["seen_contexts"] = []
        if "run_history" not in self.data:
            self.data["run_history"] = []

        # Migrate run artifacts and collect conversation IDs
        runs_dir = self.path.parent / "auto_anki_runs"
        if runs_dir.exists():
            print("  Migrating run artifacts...")
            migrated_conv_ids = migrate_run_artifacts(runs_dir, verbose=True)
            if migrated_conv_ids:
                # Add migrated conversation IDs to seen list
                existing = set(self.data["seen_conversations"])
                existing.update(migrated_conv_ids)
                self.data["seen_conversations"] = list(existing)
                print(f"  Added {len(migrated_conv_ids)} conversation IDs to seen list")

        # Save migrated state
        self.save()
        print("  Migration complete!")

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
        """Get set of previously seen context IDs (per-turn)."""
        return set(self.data.get("seen_contexts", []))

    def add_context_ids(self, context_ids: List[str]) -> None:
        """Add new context IDs to the seen list (per-turn)."""
        seen = self.get_seen_context_ids()
        seen.update(context_ids)
        self.data["seen_contexts"] = list(seen)

    def get_seen_conversation_ids(self) -> set[str]:
        """Get set of previously seen conversation IDs."""
        return set(self.data.get("seen_conversations", []))

    def add_conversation_ids(self, conversation_ids: List[str]) -> None:
        """Add new conversation IDs to the seen list."""
        seen = self.get_seen_conversation_ids()
        seen.update(conversation_ids)
        self.data["seen_conversations"] = list(seen)

    def record_run(
        self,
        run_dir: Path,
        contexts_sent: int,
        conversations_sent: Optional[int] = None,
    ) -> None:
        """Record a run in history."""
        if "run_history" not in self.data:
            self.data["run_history"] = []

        record = {
            "run_dir": str(run_dir),
            "timestamp": datetime.now().isoformat(),
            "contexts_sent": contexts_sent,
        }
        if conversations_sent is not None:
            record["conversations_sent"] = conversations_sent

        self.data["run_history"].append(record)
        self.data["last_run"] = datetime.now().isoformat()


def ensure_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


__all__ = ["StateTracker", "ensure_run_dir", "migrate_run_artifacts"]
