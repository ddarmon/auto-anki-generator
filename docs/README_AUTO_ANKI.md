# Auto Anki Agent - Enhanced Usage Guide

An agentic pipeline that automatically generates Anki flashcards from ChatGPT conversation exports using pluggable LLM backends (Codex CLI, Claude Code) for intelligent card generation.

## What Works

1. **State Tracking**: Tracks processed conversation files to avoid reprocessing
2. **Date Range Filtering**: Process conversations from specific time periods
3. **Review-Ready JSON Output**: Generates `all_proposed_cards.json` for the review UI
4. **Enhanced Deduplication**: Better HTML parsing with full card content extraction
5. **Anki Best Practices**: Codex prompt includes comprehensive flashcard design guidelines
6. **JSON Import**: Import ChatGPT/Claude `conversations.json` exports directly

## Importing Conversations

Before generating cards, you need conversation files in Markdown format. The `auto-anki-import` command converts raw JSON exports from ChatGPT or Claude into the expected format.

### Export from ChatGPT/Claude

1. **ChatGPT**: Go to Settings → Data controls → Export data
2. **Claude**: Go to Settings → Export conversations

You'll receive a `conversations.json` file containing all your conversations.

### Import to Markdown

```bash
# Basic import (outputs to configured chat_root)
uv run auto-anki-import ~/Downloads/conversations.json

# Import with verbose output
uv run auto-anki-import ~/Downloads/conversations.json -v

# Import to custom directory
uv run auto-anki-import ~/Downloads/conversations.json -o ~/chatgpt-archive

# Import and immediately generate cards
uv run auto-anki-import ~/Downloads/conversations.json --run
```

**Features:**
- Auto-detects ChatGPT vs Claude format
- Incremental updates (only writes changed files)
- Creates nested directory structure: `year/month/day/YYYY-MM-DD_slug.md`
- Sets file timestamps to conversation creation date
- Parallel processing for speed

### Import Options

- `JSON_PATH`: Path to conversations.json file (required)
- `-o, --output PATH`: Output directory (default: `chat_root` from config)
- `--base-url URL`: Base URL for conversation links (auto-detected)
- `--workers N`: Parallel worker threads (default: auto)
- `-v, --verbose`: Show progress during export
- `--run`: Run auto-anki after import completes

## Basic Usage

### First Run - Process October 2025 Conversations

```bash
python3 auto_anki_agent.py \
  --date-range 2025-10 \
  --verbose
```

### Daily Run - Only New/Unprocessed Conversations

```bash
python3 auto_anki_agent.py \
  --unprocessed-only \
  --verbose
```

### Dry Run - Preview Without Calling Codex

```bash
python3 auto_anki_agent.py \
  --date-range 2025-10 \
  --max-contexts 5 \
  --dry-run \
  --verbose
```

## Command-Line Options

### Core Options
- `--date-range RANGE`: Filter conversations by date
  - Format: `2025-10` (entire month)
  - Format: `2025-10-01:2025-10-31` (specific range)
- `--unprocessed-only`: Only process files not yet in state file
- `--reprocess-zero-card-files`: Also reprocess files that previously generated 0 cards (use with `--unprocessed-only`)
- `--only-zero-card-files`: Process ONLY files that previously generated 0 cards (exclusive backfill mode)
- `--chat-root PATH`: Root directory containing conversation files
- `--decks DECK [DECK ...]`: Anki deck names to load existing cards from (overrides config)
- `--anki-cache-ttl MINUTES`: Cache TTL for AnkiConnect card fetch (default: 5)
- `--state-file PATH`: Custom state file location
- `--output-dir PATH`: Where to save run artifacts
- `--cache-dir PATH`: Directory for Anki card cache + semantic embedding cache (default: .deck_cache)

### Conversation Processing
- `--max-contexts N`: Maximum conversations to gather per run (default: 24)
- `--contexts-per-run N`: Conversations per codex exec call (default: 8)
- `--conversation-max-turns N`: Split conversations longer than N turns (default: 10)
- `--conversation-max-chars N`: Split conversations larger than N chars (default: 8000)

### Two-Stage Pipeline
- `--two-stage`: Enable two-stage LLM pipeline (stage-1 filter + stage-2 generator, **default: enabled**)
- `--single-stage`: Disable two-stage pipeline and use single-stage card generation
- `--llm-model-stage1 MODEL`: Model for stage-1 filtering (default from config)
- `--llm-model-stage2 MODEL`: Model for stage-2 card generation (default from config)

### LLM Backend Selection
- `--llm-backend {codex,claude-code}`: Select which agentic CLI to use (overrides config)
- `--llm-model MODEL`: Override model for all stages
- `--llm-extra-arg ARG`: Pass extra arguments to the LLM CLI (repeatable)

**Deprecated** (still work, but prefer `--llm-*` equivalents):
- `--codex-model-stage1` → use `--llm-model-stage1`
- `--codex-model-stage2` → use `--llm-model-stage2`
- `--codex-model` → use `--llm-model`
- `--codex-extra-arg` → use `--llm-extra-arg`

### Deduplication
- `--similarity-threshold FLOAT`: String-based similarity threshold for dedup (default: 0.82)
- `--dedup-method {string,semantic,hybrid}`: Choose dedup strategy (default: **hybrid**, auto-falls back to string if dependencies unavailable)
- `--semantic-model NAME`: SentenceTransformers model for semantic dedup (default: all-MiniLM-L6-v2)
- `--semantic-similarity-threshold FLOAT`: Cosine similarity threshold for semantic dedup (default: 0.85)

### Post-Generation Duplicate Detection ✅ NEW!
- `--post-dedup-threshold FLOAT`: Similarity threshold for post-generation dedup (default: 0.85)
- `--skip-post-dedup`: Disable post-generation duplicate detection (restores old behavior)

### Heuristic Filtering (Optional)
- `--use-filter-heuristics`: Enable heuristic pre-filtering before Stage 1 LLM (disabled by default)
- `--min-score FLOAT`: Minimum aggregate score for conversations (only applies with `--use-filter-heuristics`)

### Other Options
- `--dry-run`: Build prompts without calling the LLM backend
- `--verbose`: Print progress information

### Configuration via `auto_anki_config.json`

Instead of passing the same paths and options every time, you can define defaults in a JSON config file. The agent looks for config in this order:

1. Path from the `AUTO_ANKI_CONFIG` environment variable (if set)
2. `./auto_anki_config.json` in the current working directory
3. `~/.auto_anki_config.json` in your home directory

Rules:
- CLI flags always override values from the config file.
- Relative paths inside the config are resolved relative to the config file’s directory.

Supported keys mirror the CLI options:

- `chat_root`
- `decks` - List of Anki deck names to load cards from
- `state_file`
- `output_dir`
- `cache_dir`
- `exclude_patterns` - List of glob patterns to exclude files (e.g., `["*_chat-*.md"]`)
- `llm_backend` - Which LLM CLI to use (`codex` or `claude-code`)
- `llm_config` - Per-backend model and reasoning configuration

Example `auto_anki_config.json`:

```json
{
  "chat_root": "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/chatgpt",
  "decks": [
    "Research Learning",
    "Technology Learning",
    "Personal Learning"
  ],
  "state_file": ".auto_anki_agent_state.json",
  "output_dir": "auto_anki_runs",
  "cache_dir": ".deck_cache",
  "exclude_patterns": ["*_chat-*.md"],

  "llm_backend": "codex",
  "llm_config": {
    "codex": {
      "model": "gpt-5.1",
      "reasoning_effort": "medium",
      "reasoning_effort_stage1": "low",
      "reasoning_effort_stage2": "high"
    },
    "claude-code": {
      "model": "claude-sonnet-4-5-20250929",
      "model_stage1": "claude-haiku-4-5-20251001",
      "model_stage2": "claude-opus-4-5-20251101"
    }
  }
}
```

**LLM Backend Notes:**
- The `llm_backend` key selects Codex (`codex`) or Claude Code (`claude-code`)
- Per-backend configs can specify different models for Stage 1 (filtering) and Stage 2 (generation)
- Use `model_stage1`/`model_stage2` to override the default `model` for specific stages
- Codex supports `reasoning_effort` (`low`/`medium`/`high`); Claude Code ignores this setting
- CLI flags (`--llm-backend`, `--llm-model-*`) override config file values

**Important:** Anki must be running with AnkiConnect plugin installed (code: 2055492159) for card loading to work.

## Output Files

Markdown exports have been retired. Review happens via the Shiny UI and JSON artifacts.

### Proposed Cards (JSON)

```
auto_anki_runs/run-TIMESTAMP/all_proposed_cards.json
```

Structured format for the Shiny review UI and any automation.

### Accepted Cards Export (from UI)

```
auto_anki_runs/run-TIMESTAMP/accepted_cards_YYYYMMDD-HHMMSS.json
```

Contains only accepted/edited cards ready for Anki import.

### State File

```
.auto_anki_agent_state.json
```

Tracks:
- Processed conversation files
- Seen context IDs
- Run history

## Workflow

1. **Script harvests** full conversations from your ChatGPT export folder
   - Each conversation contains multiple turns (user-assistant exchanges)
   - Long conversations are split at topic boundaries
2. **Applies filters**: date range, unprocessed-only flag
3. **Deduplicates** against existing Anki cards
   - Only skips conversations where ALL turns are duplicates
   - Annotates which turns are "already covered" for LLM guidance
4. **Two-stage LLM pipeline** (default):
   - **Stage 1 (Filter)**: Fast LLM reviews full conversations, selects best candidates
   - **Stage 2 (Generate)**: Strong LLM generates cards in parallel (3 concurrent workers)
5. **Post-generation deduplication** (NEW!):
   - After LLM generates cards, semantic similarity check against **full** existing card database
   - Enriches cards with `duplicate_flags` indicating likely duplicates
   - UI displays color-coded warnings for review
7. **Codex sees full learning journey**:
   - Follow-up questions (indicates user confusion)
   - Corrections in later turns
   - Topic evolution across turns
8. **Codex generates** proposed cards linked to specific turns
9. **Outputs** JSON for review (`all_proposed_cards.json`)
10. **Updates state** to track processed conversations

**Note**: Heuristic scoring (`--use-filter-heuristics`) is optional and disabled by default. The Stage 1 LLM now handles quality filtering directly.

## Anki Best Practices (Embedded in Prompt)

The codex prompt instructs the LLM to:

- **Minimum Information Principle**: One atomic fact per card
- **Clear Questions**: Precise, unambiguous prompts
- **Concise Answers**: Short but complete responses
- **Break Down Lists**: Never ask for more than 2-3 items
- **Use Cloze Deletion**: For definitions and facts
- **Combat Interference**: Distinguish confusable concepts
- **Add Context Tags**: Disambiguate terms
- **Use LaTeX**: For mathematical notation

## Examples

### Process Specific Month

```bash
python3 auto_anki_agent.py --date-range 2025-10 --verbose
```

### Process Date Range

```bash
python3 auto_anki_agent.py --date-range 2025-10-01:2025-10-15
```

### Resume from Last Run (Unprocessed Only)

```bash
python3 auto_anki_agent.py --unprocessed-only --verbose
```

### High-Volume Processing

```bash
python3 auto_anki_agent.py \
  --date-range 2025-10 \
  --max-contexts 50 \
  --contexts-per-run 10 \
  --codex-model gpt-5-codex
```

### Two-Stage Pipeline (Default)

```bash
python3 auto_anki_agent.py \
  --unprocessed-only \
  --verbose
```

The two-stage pipeline is enabled by default and works with any LLM backend:

**With Codex (default):**
- **Stage 1**: `gpt-5.1` with `model_reasoning_effort=low` (fast filtering)
- **Stage 2**: `gpt-5.1` with `model_reasoning_effort=high` (quality generation)

**With Claude Code:**
- **Stage 1**: `claude-haiku-4-5-20251001` (fast, cheap)
- **Stage 2**: `claude-opus-4-5-20251101` (powerful)

Both run **3 concurrent workers** for parallel Stage 2 processing.

Use `--single-stage` to disable and use single-stage card generation instead.

### Switching LLM Backends

```bash
# Use Claude Code instead of Codex
python3 auto_anki_agent.py \
  --llm-backend claude-code \
  --unprocessed-only \
  --verbose
```

### Backfill Zero-Card Files

If conversations were processed but generated 0 cards (e.g., due to parsing bugs), you can reprocess them:

```bash
# Preview what zero-card files exist for a date range
uv run auto-anki --only-zero-card-files --date-range 2025-12 --dry-run --verbose

# Reprocess ONLY zero-card files (dedicated backfill)
uv run auto-anki --only-zero-card-files --date-range 2025-12 --verbose

# Process both unprocessed AND zero-card files together
uv run auto-anki --unprocessed-only --reprocess-zero-card-files --date-range 2025-12 --verbose
```

**Use cases:**
- After fixing a parsing bug that caused conversations to be skipped
- When LLM errors caused a batch to generate no cards
- To retry conversations that may now yield cards with improved prompts

## Tips

1. **Start with `--dry-run`**: Preview prompts before spending tokens
2. **Use `--verbose`**: See what's happening under the hood
3. **Review in the Shiny UI**: Launch `./launch_ui.sh` to triage `all_proposed_cards.json`
4. **Use heuristic filtering** (optional): Add `--use-filter-heuristics` and `--min-score` for pre-LLM filtering
5. **Control deduplication**:
   - **Default**: Hybrid mode (semantic + string) - automatically falls back to string if dependencies unavailable
   - Use `--similarity-threshold` to tune string-based matching (default: 0.82)
   - Use `--semantic-similarity-threshold` to tune semantic matching (default: 0.85)
   - Semantic dedup uses FAISS IndexFlatIP with a persistent cache in `.deck_cache/embeddings/` (auto-invalidated when HTML decks change)
   - For best results, install semantic extras:
     - `uv pip install -e ".[semantic]"` or
     - `pip install sentence-transformers numpy faiss-cpu`
   - Force string-only mode: `--dedup-method string`
6. **Process incrementally**: Use `--unprocessed-only` for daily runs
7. **Control post-dedup** (NEW!):
   - After LLM generates cards, semantic similarity runs against ALL existing cards
   - Cards with similarity >0.85 are flagged as likely duplicates
   - Adjust threshold: `--post-dedup-threshold 0.90` (stricter)
   - Disable entirely: `--skip-post-dedup`
   - UI shows color-coded warnings: red (>95%), orange (>90%), yellow (>85%)

## Troubleshooting

### "No new contexts found"

- All conversations already processed (check state file)
- Score threshold too high (lower `--min-score`)
- Date range excludes all files

### "No zero-card files found"

- No files in the date range were processed with 0 cards generated
- The backfill for this date range is complete

### State file corrupted

```bash
rm .auto_anki_agent_state.json
```

### Want to reprocess files

**Option 1: Backfill zero-card files (recommended)**

If files were processed but generated 0 cards:

```bash
uv run auto-anki --only-zero-card-files --date-range 2025-12 --verbose
```

**Option 2: Manual state edit**

Remove specific files from `processed_files` in `.auto_anki_agent_state.json`

**Option 3: Full reset**

Delete state file and start fresh (reprocesses everything):

```bash
rm .auto_anki_agent_state.json
```

### Conversations were skipped silently

This can happen if:
- The conversation markdown format wasn't recognized (check for `user:` or `human:` roles)
- The file had no valid turns after parsing

Use `--verbose` to see which files are being processed and why some might be skipped.

## Architecture

```
┌────────────────┐
│  Conversations │
│  (Markdown)    │
└───────┬────────┘
        │
        ▼
┌──────────────────┐
│  Harvest Full   │  ← DateRangeFilter
│  Conversations   │  ← StateTracker
│  (multi-turn)    │  ← Topic Splitting
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Deduplicate vs  │  ← AnkiConnect card loading
│  Existing Cards  │  ← Annotate covered turns
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  LLM Backend     │  ← Pluggable: Codex or Claude Code
│  Selection       │  ← Config: auto_anki_config.json
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Stage 1 (Filter)│  ← Codex: gpt-5.1 w/ low reasoning
│  Full context    │  ← Claude: haiku-4.5 (fast)
└───────┬──────────┘
        │ (keep only best conversations)
        ▼
┌──────────────────────────────────────┐
│  Stage 2 (Cards) - PARALLEL          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ Worker 1│ │ Worker 2│ │ Worker 3│ │  ← 3 concurrent workers
│  └─────────┘ └─────────┘ └─────────┘ │
│  Codex: gpt-5.1 w/ high reasoning    │
│  Claude: opus-4.5 (powerful)         │
└───────┬──────────────────────────────┘
        │
        ▼
┌──────────────────┐
│  Post-Generation │  ← Semantic similarity vs FULL card DB
│  Dedup           │  ← Enriches cards with duplicate_flags
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Write JSON      │  ← all_proposed_cards.json
│  Update State    │  ← Track seen_conversations
└──────────────────┘
```

## Interactive Review UI ✨ NEW!

Review and manage proposed cards with a visual web interface:

```bash
# Install UI dependencies (one time)
uv pip install -e ".[ui]"

# Launch the review UI
./launch_ui.sh
# or
shiny run anki_review_ui.py
```

### Core Features

- ✓ **Card-by-card review** with accept/reject/edit/skip actions
- ✓ **Keyboard shortcuts** (A/R/E/S for actions, arrows for navigation)
- ✓ **Source context** display showing original conversations
- ✓ **Quality signals** and confidence scores
- ✓ **Progress tracking** with real-time statistics
- ✓ **Filtering** by deck and confidence threshold
- ✓ **Bulk operations** (accept all high-confidence cards)
- ✓ **Rejection tracking** with reasons for data-driven improvement

### AnkiConnect Integration 🎉 NEW!

**Direct import to Anki - no more copy/paste!**

```bash
# 1. Install AnkiConnect plugin in Anki (code: 2055492159)
# 2. Start Anki
# 3. Launch UI - it will detect AnkiConnect automatically
./launch_ui.sh
```

**Features:**
- ✓ **Real-time connection status** indicator
- ✓ **Import current card** with one click
- ✓ **Batch import all accepted** cards at once
- ✓ **Duplicate detection** (configurable)
- ✓ **Auto-create decks** if they don't exist
- ✓ **30-60x faster** than manual import workflow

**See `ANKICONNECT_GUIDE.md` for complete setup and usage guide.**

### Documentation

- `UI_README.md` - Complete UI documentation
- `ANKICONNECT_GUIDE.md` - AnkiConnect setup and workflows
- `UI_ENHANCEMENTS_SUMMARY.md` - Technical enhancement details
- `INTEGRATION_COMPLETE.md` - Integration summary and quick start

## Progress Dashboard

Track your conversation processing progress with a beautiful TUI dashboard:

```bash
# Show progress for last 12 weeks
uv run auto-anki-progress

# Show more history
uv run auto-anki-progress --weeks 24

# JSON output for scripting
uv run auto-anki-progress --json
```

### Features

- **Overall progress bar** - See total completion percentage
- **Weekly breakdown** - Conversations processed vs. available per week
- **Activity streak** - Track consecutive weeks of activity
- **Cards generated** - See card output per week

### Sample Output

```
╭─────────────────── Auto-Anki Progress ───────────────────╮
│  Overall: [████████░░░░░░░░░░░░░░░░░░░░░░] 42.3%         │
│  127 / 300 conversations  |  342 cards generated         │
│                                                          │
│  Activity: . . . . # # # . # # # #                       │
│  Current streak: 4 weeks  |  Longest: 5 weeks            │
│                                                          │
│  Week         │ Done │ Total │ Progress  │ Cards         │
│  Dec 2-8      │   12 │    18 │ [=====]   │    28         │
│  Nov 25-Dec 1 │   24 │    24 │ [======]  │    52         │
│  ...                                                     │
╰──────────────────────────────────────────────────────────╯
```

## Automated Batch Processing

For processing large backlogs of conversations, use the batch automation script:

```bash
# Normal mode: process unprocessed files
./scripts/auto_anki_batch.sh

# Backfill mode: reprocess zero-card files only
BACKFILL_MODE=1 ./scripts/auto_anki_batch.sh

# Press Ctrl+C to stop gracefully
# macOS: allow sleep (disable caffeinate)
PREVENT_SLEEP=0 ./scripts/auto_anki_batch.sh
```

### Features

- **Usage-aware throttling** - Monitors LLM API usage (Codex or Claude) and pauses when exceeding sustainable pace
- **Month-by-month processing** - Works backwards from current month (newest conversations first)
- **Auto-advance** - Moves to next month when current month is exhausted
- **Graceful shutdown** - Ctrl+C stops cleanly, logs final state
- **Sleep prevention (macOS)** - Prevents idle sleep while running (uses `caffeinate`, disable with `PREVENT_SLEEP=0`)
- **Comprehensive logging** - All output saved to `auto_anki_runs/batch_*.log`

### How It Works

1. Detects configured LLM backend from `auto_anki_config.json` (`codex` or `claude-code`)
2. Checks 5h window usage via appropriate script (`codex-usage.sh` or `claude-usage.sh`)
3. If usage exceeds pace + 10%, waits 15 minutes (wall-clock)
4. Runs auto-anki command based on mode:
   - **Normal mode**: `uv run auto-anki --date-range YYYY-MM --unprocessed-only --verbose`
   - **Backfill mode** (`BACKFILL_MODE=1`): `uv run auto-anki --date-range YYYY-MM --only-zero-card-files --verbose`
5. Shows progress via `uv run auto-anki-progress`
6. Detects exhaustion:
   - Normal mode: "No new conversations found" → advances to previous month
   - Backfill mode: "No zero-card files found" → advances to previous month
7. Loops indefinitely until manually stopped

### Requirements

- LLM CLI logged in:
  - For Codex: `~/.codex/auth.json` must exist
  - For Claude Code: OAuth credentials in macOS keychain
- Anki running with AnkiConnect plugin
- `auto_anki_config.json` configured with your decks, `chat_root`, and `llm_backend`

### Configuration

Edit `scripts/auto_anki_batch.sh` to customize:

```bash
PACE_BUFFER=10          # Wait threshold: pace + this value (%)
WAIT_DURATION=900       # Wait time in seconds (default: 15 min)
PREVENT_SLEEP=1         # macOS: prevent idle sleep (disable with PREVENT_SLEEP=0)
BACKFILL_MODE=0         # Set to 1 to reprocess zero-card files instead of unprocessed files
END_YEAR=2022           # How far back to process
END_MONTH=1
```

Or set environment variables at runtime:

```bash
# Run backfill batch
BACKFILL_MODE=1 ./scripts/auto_anki_batch.sh

# Run without preventing sleep
PREVENT_SLEEP=0 ./scripts/auto_anki_batch.sh
```

### Estimate Time to Completion

Monitor batch progress and estimate remaining time:

```bash
# Analyze most recent batch log
./scripts/estimate_completion.sh

# Analyze specific log file
./scripts/estimate_completion.sh auto_anki_runs/batch_20251209_193717.log
```

**Sample output:**

```
Analyzing: batch_20251209_193717.log
==========================================

Progress Summary
----------------
Start time:              2025-12-09 19:37:23
Latest time:             2025-12-10 07:41:19
Conversations at start:  954 / 11088
Conversations now:       1655 / 11088
Progress:                14.9%

Processing Rate
---------------
Elapsed time:            12.1 hours (43436 seconds)
Conversations processed: 701
Rate:                    58.0 conversations/hour
                         0.96 conversations/minute

Time Estimate
-------------
Conversations remaining: 9433
Estimated time:          162.6 hours (6.8 days)
Estimated completion:    2025-12-17 02:19

Recent Rate (last ~10 runs)
---------------------------
Rate:                    227.2 conversations/hour
Estimated time:          41.5 hours (1.7 days)
```

**Note:** The "Recent Rate" is typically higher than the overall rate because it excludes throttling pauses. Use it for a more accurate estimate of active processing time.

## Future Enhancements

- [x] ~~Automatic HTML card import via AnkiConnect~~ ✅ **DONE!**
- [x] ~~Semantic deduplication with embeddings~~ ✅ **DONE!** (SentenceTransformers + FAISS vector cache)
- [x] ~~Conversation-level processing~~ ✅ **DONE!** (LLM sees full learning journey)
- [x] ~~Two-stage LLM pipeline~~ ✅ **DONE!** (Fast filter + slow generation)
- [x] ~~Test suite~~ ✅ **DONE!** (137 tests covering core functions)
- [x] ~~Progress dashboard~~ ✅ **DONE!** (TUI with weekly stats + streak tracking)
- [x] ~~Batch automation with usage throttling~~ ✅ **DONE!** (`scripts/auto_anki_batch.sh`)
- [x] ~~Pluggable LLM backends~~ ✅ **DONE!** (Codex CLI + Claude Code)
- [x] ~~Post-generation duplicate detection~~ ✅ **DONE!** (Semantic similarity vs full card DB, UI warnings)
- [x] ~~Zero-card file backfill~~ ✅ **DONE!** (`--only-zero-card-files`, `--reprocess-zero-card-files`, `BACKFILL_MODE`)
- [ ] Tag taxonomy management
- [ ] Multi-deck routing logic with ML
- [ ] Topic distribution visualization
- [ ] Active learning from user feedback

## Development

### Running Tests

The project includes a pytest-based test suite covering core functions:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=auto_anki --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_scoring.py -v
```

**Test files:**

| File | Tests |
|------|-------|
| `test_scoring.py` | `detect_signals`, `extract_key_terms`, `extract_key_points` |
| `test_normalization.py` | `normalize_text`, `quick_similarity` |
| `test_parsing.py` | `parse_chat_entries`, `extract_turns`, `parse_chat_metadata` |
| `test_date_filter.py` | `DateRangeFilter` class |
| `test_dedup.py` | `is_duplicate_context`, `DuplicateFlags`, `enrich_cards_with_duplicate_flags` (21 tests) |
| `test_llm_backends.py` | LLM backend abstraction (26 tests) |

### Dev Dependencies

Install test dependencies:

```bash
uv pip install pytest pytest-cov
# or install from pyproject.toml dev-dependencies
```
