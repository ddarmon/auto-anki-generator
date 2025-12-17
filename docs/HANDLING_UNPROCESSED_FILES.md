# Handling Unprocessed Files: Investigation & Resolution

**Date**: 2025-12-16
**Status**: Resolved

## Executive Summary

This document describes the investigation and resolution of a state tracking inconsistency in the auto-anki system where the batch script reported ~1,967 "unprocessed" files but `auto-anki --unprocessed-only` found 0 new conversations to process.

Two issues were identified and fixed:
1. **State tracking gap**: Files processed before v2 migration had conversation IDs in `seen_conversations` but weren't in `processed_files`
2. **Placeholder files**: ~367 `chat-XXXX.md` stub files with no valid turns were being counted as "unprocessed"

---

## Part 1: State Tracking Inconsistency

### Problem Description

The state file (`.auto_anki_agent_state.json`) has two tracking mechanisms:

| Mechanism | Purpose | Count (before fix) |
|-----------|---------|-------------------|
| `processed_files` | Maps file paths → `{processed_at, cards_generated}` | 9,230 |
| `seen_conversations` | List of conversation_id hashes | 12,150 |

**The inconsistency**: Some files had their `conversation_id` in `seen_conversations` but were NOT in `processed_files`.

### Root Cause

The V1 → V2 state migration (around 2025-11-08):
1. Populated `seen_conversations` from historical run artifacts (`selected_conversations.json`)
2. Did NOT update `processed_files` for those historical files

Additionally, `split_conversation_by_topic()` appends `_part{N}` suffixes to conversation IDs. Some files had their `{base_id}_part1` in `seen_conversations` but not the base ID.

### Discovery vs Harvest Mismatch

| Component | Checks | Result |
|-----------|--------|--------|
| `discover_months_with_work` (batch script) | `is_file_processed()` only | Found 1,967 "unprocessed" |
| `harvest_conversations()` | Both `is_file_processed()` AND `conversation_id in seen_conversations` | Found 0 new |

### Resolution: Reconciliation Script

Created `scripts/reconcile_state.py` to add missing files to `processed_files`:

```bash
# Dry run - see what would be changed
uv run python scripts/reconcile_state.py --dry-run

# Apply changes
uv run python scripts/reconcile_state.py --apply
```

**Key feature**: The script checks both base IDs and `_part{N}` variants:

```python
# Build index of base IDs (without _part suffix)
seen_base_ids = set()
for conv_id in seen_conversations:
    if "_part" in conv_id:
        base_id = conv_id.rsplit("_part", 1)[0]
        seen_base_ids.add(base_id)
    else:
        seen_base_ids.add(conv_id)
```

**Results**:
- First reconciliation: Added 1,535 files (base ID matches)
- Second reconciliation: Added 200 files (`_part{N}` matches)
- Final `processed_files` count: 10,965

---

## Part 2: Placeholder File Exclusion

### Problem Description

After reconciliation, 232 files remained "unprocessed". Investigation revealed ALL of them were placeholder/stub files with 0 valid turns:

```
2025-11-27_chat-0009.md
2025-11-27_chat-0021.md
2024-03-02_chat-5066.md
...
```

These files:
1. Get counted as "unprocessed" by discovery
2. Are correctly skipped by `harvest_conversations()` (no turns to parse)
3. Never get marked as "processed" since no work is done
4. Cause the batch script to loop endlessly

### Resolution: Exclusion Patterns

Added configurable `exclude_patterns` feature to skip files matching glob patterns.

#### Files Modified

| File | Change |
|------|--------|
| `auto_anki_agent.py` | Added `--exclude-patterns` CLI argument |
| `auto_anki/contexts.py` | Added `matches_exclusion_pattern()` function, applied in `harvest_conversations()` |
| `scripts/auto_anki_batch.sh` | Updated `discover_months_with_work` to use exclusion patterns |
| `auto_anki_config.json` | Added `"exclude_patterns": ["*_chat-*.md"]` |

#### Implementation Details

**New function in `contexts.py`**:
```python
def matches_exclusion_pattern(path: Path, patterns: List[str]) -> bool:
    """Check if file matches any exclusion pattern (glob-style)."""
    import fnmatch
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
```

**Applied in `harvest_conversations()` after date filter**:
```python
# Apply exclusion patterns
exclude_patterns = getattr(args, "exclude_patterns", None) or []
if exclude_patterns:
    files = [f for f in files if not matches_exclusion_pattern(f, exclude_patterns)]
```

**Config merging in `auto_anki_agent.py`**:
```python
# Merge exclude_patterns from config if not set via CLI
if args.exclude_patterns is None:
    args.exclude_patterns = config.get("exclude_patterns", [])
```

#### Usage

```bash
# Uses config file pattern (recommended)
uv run auto-anki --unprocessed-only --verbose

# Override via CLI
uv run auto-anki --exclude-patterns '*.draft.md' '*_chat-*.md'

# Check what would be excluded
uv run auto-anki --dry-run --verbose
```

#### Config File

```json
{
  "exclude_patterns": ["*_chat-*.md"],
  ...
}
```

---

## Current State (as of 2025-12-16)

### State File Counts

| Metric | Value |
|--------|-------|
| `processed_files` | 11,024 |
| `seen_conversations` | 12,150 |
| Files excluded by `*_chat-*.md` | 367 |
| Total conversations (after exclusion) | 10,829 |
| **Unprocessed files** | **0** |

### Resolution Summary

The original 58 "unprocessed" files were all **stub files** with no valid user+assistant turns. After investigation:

1. **Reconciliation** (default mode): Added 1,735 files to `processed_files` that had conversation IDs in `seen_conversations`
2. **Stub marking** (`--mark-stubs`): Added 59 stub files to `processed_files` since they can never generate cards

### Tests

All 185 tests pass after changes.

---

## Scripts & Tools Created

### 1. `scripts/reconcile_state.py`

Reconciles state file inconsistencies. Has two modes:

**Default mode**: Add files with seen conversation IDs to `processed_files`.
```bash
uv run python scripts/reconcile_state.py --dry-run  # Preview
uv run python scripts/reconcile_state.py --apply    # Execute
```

**Stub marking mode** (`--mark-stubs`): Mark stub files (no valid user+assistant turns) as processed. These files will never generate cards, so marking them prevents them from appearing as "unprocessed".
```bash
uv run python scripts/reconcile_state.py --mark-stubs --dry-run  # Preview
uv run python scripts/reconcile_state.py --mark-stubs --apply    # Execute
```

Stub files are identified as files with 0 valid turns after parsing. Common causes:
- Empty entries (role lines but no text content)
- Only user messages (no assistant response)
- Only assistant messages (no user prompt)
- Metadata-only files (just URL/date headers)

### 2. Exclusion Pattern Feature

Built into the main CLI and batch script. Configured via:
- Config file: `"exclude_patterns": ["*_chat-*.md"]`
- CLI flag: `--exclude-patterns '*_chat-*.md'`

---

## Backup Files

Created during investigation:
- `.auto_anki_agent_state.json.bak-20251216193016` - Before first reconciliation
- `.auto_anki_agent_state.json.v1_backup` - Original v1 state before migration

---

## Lessons Learned

1. **Dual tracking mechanisms need consistency**: When migrating state formats, ensure all tracking mechanisms are updated together.

2. **Split conversation IDs need special handling**: The `_part{N}` suffix from `split_conversation_by_topic()` creates derived IDs that must be considered when checking "seen" status.

3. **Discovery should match harvest logic**: The batch script's discovery function should apply the same filters as `harvest_conversations()` to avoid reporting work that can't be done.

4. **Placeholder files need explicit exclusion**: Files imported from external sources may include stubs that will never have valid content.

5. **All components must apply exclusion patterns consistently**: The progress TUI (`auto-anki-progress`) was counting excluded files in its totals, causing the completion percentage to appear lower than actual. Fixed by adding `exclude_patterns` support to `progress.py:scan_conversations()`.

---

## Future Recommendations

1. **Consider consolidating tracking**: The dual `processed_files` / `seen_conversations` mechanism could be simplified to a single source of truth.

2. **Add validation on startup**: Check for and warn about state inconsistencies when the tool starts.

3. ~~**Log excluded files**: Add verbose logging when files are excluded by patterns for debugging.~~ *(Lower priority now that exclusion is working correctly)*

4. **Document placeholder file sources**: Identify where `chat-XXXX.md` files come from and whether they can be filtered at import time.
