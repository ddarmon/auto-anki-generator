# Auto Anki Agent - Enhanced Usage Guide

An agentic pipeline that automatically generates Anki flashcards from ChatGPT conversation exports using `codex exec` for intelligent card generation.

## What Works

1. **State Tracking**: Tracks processed conversation files to avoid reprocessing
2. **Date Range Filtering**: Process conversations from specific time periods
3. **Markdown Output**: Generate human-readable markdown files for review
4. **Enhanced Deduplication**: Better HTML parsing with full card content extraction
5. **Anki Best Practices**: Codex prompt includes comprehensive flashcard design guidelines

## Basic Usage

### First Run - Process October 2025 Conversations

```bash
python3 auto_anki_agent.py \
  --date-range 2025-10 \
  --output-format markdown \
  --verbose
```

### Daily Run - Only New/Unprocessed Conversations

```bash
python3 auto_anki_agent.py \
  --unprocessed-only \
  --output-format both \
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
- `--output-format {json,markdown,both}`: Choose output format (default: both)
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
- `--codex-model-stage1 MODEL`: Codex model for fast stage-1 filtering (default: `gpt-5.1`)
- `--codex-model-stage2 MODEL`: Codex model for stage-2 card generation (default: `gpt-5.1`)

### Deduplication
- `--similarity-threshold FLOAT`: String-based similarity threshold for dedup (default: 0.82)
- `--dedup-method {string,semantic,hybrid}`: Choose dedup strategy (default: **hybrid**, auto-falls back to string if dependencies unavailable)
- `--semantic-model NAME`: SentenceTransformers model for semantic dedup (default: all-MiniLM-L6-v2)
- `--semantic-similarity-threshold FLOAT`: Cosine similarity threshold for semantic dedup (default: 0.85)

### Heuristic Filtering (Optional)
- `--use-filter-heuristics`: Enable heuristic pre-filtering before Stage 1 LLM (disabled by default)
- `--min-score FLOAT`: Minimum aggregate score for conversations (only applies with `--use-filter-heuristics`)

### Other Options
- `--dry-run`: Build prompts without calling codex
- `--verbose`: Print progress information
- `--codex-model MODEL`: Override default codex model (falls back to `gpt-5.1` if unset)
- `--codex-extra-arg ARG`: Passthrough args to codex (repeatable)

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
  "cache_dir": ".deck_cache"
}
```

**Important:** Anki must be running with AnkiConnect plugin installed (code: 2055492159) for card loading to work.

## Output Files

### Markdown Output (for review)

```
auto_anki_runs/proposed_cards_2025-11-08.md
```

Human-readable format with:
- Card front/back
- Deck assignment
- Tags and confidence scores
- Source conversation metadata
- Notes on why each card was created

### JSON Output (for automation)

```
auto_anki_runs/run-TIMESTAMP/all_proposed_cards.json
```

Structured format for potential automated import.

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
5. **Codex sees full learning journey**:
   - Follow-up questions (indicates user confusion)
   - Corrections in later turns
   - Topic evolution across turns
6. **Codex generates** proposed cards linked to specific turns
7. **Outputs** markdown and/or JSON for review
8. **Updates state** to track processed conversations

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

The two-stage pipeline is enabled by default:
- **Stage 1**: Fast model (`gpt-5.1` with `model_reasoning_effort=low`) reviews full conversations
  and selects the best candidates for card generation
- **Stage 2**: Strong model (`gpt-5.1` with `model_reasoning_effort=high`) generates cards
  with **3 concurrent workers** for parallel processing

Use `--single-stage` to disable and use single-stage card generation instead.

## Tips

1. **Start with `--dry-run`**: Preview prompts before spending tokens
2. **Use `--verbose`**: See what's happening under the hood
3. **Review markdown output**: Easier to scan than JSON
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

## Troubleshooting

### "No new contexts found"

- All conversations already processed (check state file)
- Score threshold too high (lower `--min-score`)
- Date range excludes all files

### State file corrupted

```bash
rm .auto_anki_agent_state.json
```

### Want to reprocess files

- Remove from state file manually, or
- Delete state file and start fresh

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
│  Deduplicate vs  │  ← HTML Deck Parser
│  Existing Cards  │  ← Annotate covered turns
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Stage 1 (Filter)│  ← Fast model (gpt-5.1 w/ low reasoning)
│  Full context    │  ← LLM sees full conversations
└───────┬──────────┘
        │ (keep only best conversations)
        ▼
┌──────────────────────────────────────┐
│  Stage 2 (Cards) - PARALLEL          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ Worker 1│ │ Worker 2│ │ Worker 3│ │  ← 3 concurrent workers
│  └─────────┘ └─────────┘ └─────────┘ │
│  Strong model (gpt-5.1 w/ high)      │
└───────┬──────────────────────────────┘
        │
        ▼
┌──────────────────┐
│  Format Output   │  ← Markdown/JSON
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
./scripts/auto_anki_batch.sh
# Press Ctrl+C to stop gracefully
```

### Features

- **Usage-aware throttling** - Monitors Codex API usage and pauses when exceeding sustainable pace
- **Month-by-month processing** - Works backwards from current month (newest conversations first)
- **Auto-advance** - Moves to next month when current month is exhausted
- **Graceful shutdown** - Ctrl+C stops cleanly, logs final state
- **Comprehensive logging** - All output saved to `auto_anki_runs/batch_*.log`

### How It Works

1. Checks Codex 5h window usage via `scripts/codex-usage.sh`
2. If usage exceeds pace + 10%, waits 15 minutes
3. Runs `uv run auto-anki --date-range YYYY-MM --unprocessed-only --verbose`
4. Shows progress via `uv run auto-anki-progress`
5. When "No new conversations found", advances to previous month
6. Loops indefinitely until manually stopped

### Requirements

- Codex CLI logged in (`~/.codex/auth.json` must exist)
- Anki running with AnkiConnect plugin
- `auto_anki_config.json` configured with your decks and `chat_root`

### Configuration

Edit `scripts/auto_anki_batch.sh` to customize:

```bash
PACE_BUFFER=10          # Wait threshold: pace + this value (%)
WAIT_DURATION=900       # Wait time in seconds (default: 15 min)
END_YEAR=2022           # How far back to process
END_MONTH=1
```

## Future Enhancements

- [x] ~~Automatic HTML card import via AnkiConnect~~ ✅ **DONE!**
- [x] ~~Semantic deduplication with embeddings~~ ✅ **DONE!** (SentenceTransformers + FAISS vector cache)
- [x] ~~Conversation-level processing~~ ✅ **DONE!** (LLM sees full learning journey)
- [x] ~~Two-stage LLM pipeline~~ ✅ **DONE!** (Fast filter + slow generation)
- [x] ~~Test suite~~ ✅ **DONE!** (97 tests covering core functions)
- [x] ~~Progress dashboard~~ ✅ **DONE!** (TUI with weekly stats + streak tracking)
- [x] ~~Batch automation with usage throttling~~ ✅ **DONE!** (`scripts/auto_anki_batch.sh`)
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
| `test_dedup.py` | `is_duplicate_context` |

### Dev Dependencies

Install test dependencies:

```bash
uv pip install pytest pytest-cov
# or install from pyproject.toml dev-dependencies
```
