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

- `--date-range RANGE`: Filter conversations by date
  - Format: `2025-10` (entire month)
  - Format: `2025-10-01:2025-10-31` (specific range)
- `--unprocessed-only`: Only process files not yet in state file
- `--output-format {json,markdown,both}`: Choose output format (default: both)
- `--chat-root PATH`: Root directory containing conversation files
- `--deck-glob PATTERN`: Glob pattern for HTML deck files
- `--state-file PATH`: Custom state file location
- `--output-dir PATH`: Where to save run artifacts
- `--max-contexts N`: Maximum contexts to gather per run (default: 24)
- `--contexts-per-run N`: Contexts per codex exec call (default: 8)
- `--similarity-threshold FLOAT`: String-based similarity threshold for dedup (default: 0.82)
- `--dedup-method {string,semantic,hybrid}`: Choose dedup strategy (default: string)
- `--semantic-model NAME`: SentenceTransformers model for semantic dedup (default: all-MiniLM-L6-v2)
- `--semantic-similarity-threshold FLOAT`: Cosine similarity threshold for semantic dedup (default: 0.85)
- `--dry-run`: Build prompts without calling codex
- `--verbose`: Print progress information
- `--codex-model MODEL`: Override default codex model
- `--codex-extra-arg ARG`: Passthrough args to codex (repeatable)

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

1. **Script harvests** conversations from your ChatGPT export folder
2. **Applies filters**: date range, unprocessed-only flag
3. **Scores contexts** using heuristics (questions, definitions, bullet points, etc.)
4. **Deduplicates** against existing Anki cards (HTML collections)
5. **Batches contexts** and sends to `codex exec`
6. **Codex generates** proposed cards following Anki best practices
7. **Outputs** markdown and/or JSON for review
8. **Updates state** to track processed files

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

## Tips

1. **Start with `--dry-run`**: Preview prompts before spending tokens
2. **Use `--verbose`**: See what's happening under the hood
3. **Review markdown output**: Easier to scan than JSON
4. **Adjust scoring**: Use `--min-score` to filter more/less aggressively
5. **Control deduplication**:
   - Use `--similarity-threshold` to tune string-based matching
   - Enable semantic dedup with `--dedup-method semantic` or `--dedup-method hybrid`
   - Install semantic extras with:
     - `uv pip install -e ".[semantic]"` or
     - `pip install sentence-transformers numpy`
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
│  Harvest & Score │  ← DateRangeFilter
│  Contexts        │  ← StateTracker
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Deduplicate vs  │  ← HTML Deck Parser
│  Existing Cards  │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Batch & Build   │  ← Anki Best Practices
│  Codex Prompts   │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  codex exec      │  ← LLM Decision Layer
│  (Agentic)       │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│  Format Output   │  ← Markdown/JSON
│  Update State    │
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

## Future Enhancements

- [x] ~~Automatic HTML card import via AnkiConnect~~ ✅ **DONE!**
- [ ] Tag taxonomy management
- [ ] Semantic deduplication with embeddings
- [ ] Multi-deck routing logic with ML
- [ ] Topic distribution visualization
