# CLAUDE.md - Auto Anki Agent Project Guide

## Project Overview

**Auto Anki Agent** generates Anki flashcards from ChatGPT/Claude conversation exports using LLM-based decision-making via pluggable backends (Codex CLI, Claude Code).

**Core Purpose**: Transform learning conversations into flashcards following Anki best practices.

## Architecture (7-Phase Pipeline)

1. **HARVEST**: Load Anki cards via AnkiConnect, scan markdown exports → `Conversation` objects, split long conversations, apply filters
2. **SCORING** (optional, `--use-filter-heuristics`): Aggregate heuristic scoring. OFF by default - all go to Stage 1 LLM
3. **DEDUPLICATION**: Conversation-level dedup via string + semantic similarity
4. **BATCHING**: Group conversations, build prompts with system instructions + deck list
5. **TWO-STAGE LLM**:
   - Stage 1 (filter): Fast model decides keep/skip
   - Stage 2 (generate): Powerful model creates cards (3 parallel workers)
6. **POST-DEDUP**: Semantic similarity against full card database, enriches cards with `duplicate_flags`
7. **OUTPUT**: Save to run directory, generate markdown/JSON, update state

## Key Components

### Data Structures (in `auto_anki/`)

- **`Card`** (`cards.py`): Flashcard with deck, front, back, tags, url
- **`Conversation`** (`contexts.py`): Full conversation with turns, aggregate_score, key_topics
- **`ChatTurn`** (`contexts.py`): Single user-assistant exchange with context_id, score, signals
- **`DateRangeFilter`** (`contexts.py`): Filter files by date (e.g., `2025-10` or `2025-10-01:2025-10-31`)
- **`StateTracker`** (`state.py`): Manages `.auto_anki_agent_state.json`, tracks seen conversations (v2 schema)

### LLM Backend Abstraction (`auto_anki/llm_backends/`)

| Backend | CLI | Input | Output |
|---------|-----|-------|--------|
| `codex` | `codex exec` | stdin | `--output-last-message` file |
| `claude-code` | `claude --print` | `-p "prompt"` | stdout |

**Config** (`auto_anki_config.json`): Set `llm_backend` and per-backend `model`, `model_stage1`, `model_stage2`.

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `load_cards_from_anki()` | `cards.py` | Load existing cards via AnkiConnect (requires Anki + plugin 2055492159) |
| `harvest_conversations()` | `contexts.py` | Parse markdown exports → Conversation objects, split long convos |
| `detect_signals()` | `contexts.py` | Heuristic scoring (only with `--use-filter-heuristics`) |
| `prune_conversations()` | `dedup.py` | Filter/annotate duplicates via string + semantic similarity |
| `enrich_cards_with_duplicate_flags()` | `dedup.py` | Post-generation semantic dedup against full card DB |
| `build_conversation_prompt()` | `codex.py` | Build LLM prompt with system instructions, deck list, conversations |
| `run_codex_pipeline()` | `codex.py` | Two-stage LLM execution with parallel Stage 2 |
| `is_file_zero_card()` | `state.py` | Check if file was processed but generated 0 cards |
| `get_zero_card_files()` | `state.py` | List all files that generated 0 cards (for backfill) |

## File Organization

```
auto-anki-generator/
├── auto_anki/                   # Core package
│   ├── cli.py                   # `auto-anki` entrypoint
│   ├── cards.py, contexts.py, dedup.py, codex.py, state.py
│   ├── progress.py              # `auto-anki-progress` TUI
│   ├── import_conversations.py  # `auto-anki-import` CLI
│   └── llm_backends/            # Codex, Claude Code backends
├── tests/                       # pytest suite
├── anki_review_ui.py            # Shiny review UI
├── anki_connect.py              # AnkiConnect client
├── scripts/                     # auto_anki_batch.sh, estimate_completion.sh, usage scripts
├── docs/                        # README_AUTO_ANKI.md, FUTURE_DIRECTIONS.md, etc.
└── auto_anki_runs/              # Output: run-*/selected_conversations.json, *_cards.json
```

## Common Tasks

### Key Defaults
- `max_contexts=24`, `contexts_per_run=8`, `similarity_threshold=0.82`, `semantic_threshold=0.85`
- `--conversation-max-turns=10`, `--conversation-max-chars=8000`
- Heuristics OFF by default, Stage 2 parallel workers: 3

### Debugging
1. Check `.auto_anki_agent_state.json` (state_version=2, seen_conversations)
2. Check `auto_anki_runs/run-*/` for `selected_conversations.json`, `codex_response_*.json`
3. Use `--verbose` and `--dry-run` to inspect decisions/prompts

### Running Tests
```bash
uv run pytest                                    # All tests
uv run pytest --cov=auto_anki --cov-report=term  # With coverage
```

## CLI Usage

```bash
# Daily processing
uv run auto-anki --unprocessed-only --verbose

# Month review
uv run auto-anki --date-range 2025-11 --verbose

# Dry run (no LLM calls)
uv run auto-anki --dry-run --max-contexts 5 --verbose

# Switch backend
uv run auto-anki --llm-backend claude-code --verbose

# Import ChatGPT/Claude JSON export
uv run auto-anki-import ~/Downloads/conversations.json

# Batch processing with throttling
./scripts/auto_anki_batch.sh

# Backfill zero-card files (reprocess files that generated 0 cards)
uv run auto-anki --only-zero-card-files --date-range 2025-12 --verbose

# Batch backfill mode
BACKFILL_MODE=1 ./scripts/auto_anki_batch.sh

# Progress dashboard
uv run auto-anki-progress
```

**Key flags:** `--dry-run`, `--verbose`, `--unprocessed-only`, `--date-range`, `--llm-backend`, `--llm-model`, `--llm-model-stage1/2`, `--only-zero-card-files`, `--reprocess-zero-card-files`

## Interactive Review UI

Launch: `./launch_ui.sh` - Shiny app with keyboard shortcuts (A/R/E/S), AnkiConnect integration, batch import.
See `docs/UI_README.md` and `docs/ANKICONNECT_GUIDE.md` for details.

## LLM Response Schema

```json
{
  "cards": [{"conversation_id": "...", "turn_index": 2, "deck": "...", "front": "...", "back": "...", "confidence": 0.85, "tags": [...]}],
  "skipped_conversations": [{"conversation_id": "...", "reason": "..."}],
  "skipped_turns": [{"conversation_id": "...", "turn_index": 0, "reason": "..."}]
}
```

## Development Guidelines

**Before implementing features:**
1. Check `docs/FUTURE_DIRECTIONS.md` for roadmap
2. Consider state migration if changing `.auto_anki_agent_state.json` format
3. Test with `--dry-run` and `--verbose`

**Priorities:** Card quality > Reduced friction > Better scaling

**Debugging targets:** `detect_signals()`, `prune_conversations()`, `build_conversation_prompt()`, `json_repair`

## Resources

- `docs/README_AUTO_ANKI.md` - User documentation
- `docs/FUTURE_DIRECTIONS.md` - Roadmap (comprehensive!)
- `docs/ANKICONNECT_GUIDE.md` - AnkiConnect setup

---

**Status**: Production-ready. CLIs: `auto-anki`, `auto-anki-import`, `auto-anki-progress`

**Last updated**: 2025-12-13
