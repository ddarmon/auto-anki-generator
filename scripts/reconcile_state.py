#!/usr/bin/env python3
"""
Reconcile state file inconsistencies.

Modes:
1. Default: Find files with conversation_id in seen_conversations but NOT in
   processed_files, and add them to processed_files with cards_generated=0.

2. --mark-stubs: Find stub files (no valid user+assistant turns) that are NOT
   in processed_files, and mark them as processed since they'll never generate cards.

Usage:
    python scripts/reconcile_state.py --dry-run              # Show what would be changed
    python scripts/reconcile_state.py --apply                # Actually apply changes
    python scripts/reconcile_state.py --mark-stubs --dry-run # Preview stub marking
    python scripts/reconcile_state.py --mark-stubs --apply   # Mark stubs as processed
"""

import argparse
import fnmatch
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reconcile state file inconsistencies")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    group.add_argument("--apply", action="store_true", help="Apply changes to state file")
    parser.add_argument(
        "--mark-stubs",
        action="store_true",
        help="Mark stub files (no valid turns) as processed instead of reconciling seen conversations",
    )
    args = parser.parse_args()

    # Import after argparse to avoid slow startup for --help
    from auto_anki.contexts import extract_turns, parse_chat_entries

    # Load config
    config_path = Path("auto_anki_config.json")
    if not config_path.exists():
        print("Error: auto_anki_config.json not found", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    chat_root = Path(config["chat_root"]).expanduser()

    # Load state
    state_path = Path(".auto_anki_agent_state.json")
    if not state_path.exists():
        print("Error: State file not found", file=sys.stderr)
        sys.exit(1)

    state = json.loads(state_path.read_text())
    processed_files = state.get("processed_files", {})
    seen_conversations = set(state.get("seen_conversations", []))

    # Get exclusion patterns from config
    exclude_patterns = config.get("exclude_patterns", [])

    # Build index of base IDs (without _part suffix) for efficient lookup
    seen_base_ids = set()
    for conv_id in seen_conversations:
        if "_part" in conv_id:
            base_id = conv_id.rsplit("_part", 1)[0]
            seen_base_ids.add(base_id)
        else:
            seen_base_ids.add(conv_id)

    print("=" * 70)
    if args.mark_stubs:
        print("MARK STUB FILES AS PROCESSED")
    else:
        print("STATE RECONCILIATION")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Chat root: {chat_root}")
    print(f"Current processed_files: {len(processed_files):,}")
    print(f"Current seen_conversations: {len(seen_conversations):,}")
    if exclude_patterns:
        print(f"Exclusion patterns: {exclude_patterns}")
    print()

    # Find files to process
    files_to_reconcile = []
    all_files = list(chat_root.rglob("*.md"))
    print(f"Scanning {len(all_files):,} markdown files...")

    for i, f in enumerate(all_files):
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1:,} / {len(all_files):,}")

        # Skip if already in processed_files
        if str(f) in processed_files:
            continue

        # Skip files matching exclusion patterns
        if any(fnmatch.fnmatch(f.name, pattern) for pattern in exclude_patterns):
            continue

        # Read and parse
        try:
            text = f.read_text()
        except Exception:
            continue

        if "\n---" in text:
            header, body = text.split("\n---", 1)
        else:
            header, body = text, ""

        entries = parse_chat_entries(body)
        raw_turns = extract_turns(entries)

        if args.mark_stubs:
            # Mark stub files: files with NO valid turns
            if not raw_turns:
                files_to_reconcile.append((str(f), "stub"))
        else:
            # Default reconciliation: files with valid turns in seen_conversations
            if not raw_turns:
                continue

            # Compute conversation_id
            first_timestamp = raw_turns[0][0].get("timestamp", "")
            conversation_id = hashlib.sha256(
                f"{f}:{first_timestamp}".encode("utf-8")
            ).hexdigest()

            # Check if in seen_conversations (base ID or any _part{N} variant)
            if conversation_id in seen_base_ids:
                files_to_reconcile.append((str(f), conversation_id))

    print()
    if args.mark_stubs:
        print(f"Stub files to mark as processed: {len(files_to_reconcile):,}")
    else:
        print(f"Files to reconcile: {len(files_to_reconcile):,}")

    if not files_to_reconcile:
        if args.mark_stubs:
            print("No stub files found.")
        else:
            print("No files need reconciliation.")
        sys.exit(0)

    if args.dry_run:
        if args.mark_stubs:
            print("\nDRY RUN - would mark these stub files as processed:")
        else:
            print("\nDRY RUN - would add these files to processed_files:")
        for file_path, marker in files_to_reconcile[:15]:
            filename = Path(file_path).name
            if args.mark_stubs:
                print(f"  - {filename}")
            else:
                print(f"  - {filename} (conv_id: {marker[:12]}...)")
        if len(files_to_reconcile) > 15:
            print(f"  ... and {len(files_to_reconcile) - 15} more files")
        print("\nRun with --apply to make changes.")
    else:
        # Apply changes
        print("\nApplying changes...")
        timestamp = datetime.now().isoformat()

        for file_path, marker in files_to_reconcile:
            if args.mark_stubs:
                processed_files[file_path] = {
                    "processed_at": timestamp,
                    "cards_generated": 0,
                    "stub_file": True,  # Mark as stub for audit
                }
            else:
                processed_files[file_path] = {
                    "processed_at": timestamp,
                    "cards_generated": 0,
                    "reconciled": True,  # Mark as reconciled for audit
                }

        state["processed_files"] = processed_files

        # Save state
        state_path.write_text(json.dumps(state, indent=2))
        print(f"Added {len(files_to_reconcile):,} files to processed_files")
        print(f"New processed_files count: {len(processed_files):,}")
        print("\nState file updated successfully.")


if __name__ == "__main__":
    main()
