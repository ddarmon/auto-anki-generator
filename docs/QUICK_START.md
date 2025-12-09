# Auto Anki Agent - Quick Start

## Daily Workflow

### 1. Process New Conversations (Recommended)

```bash
cd ~/Dropbox/Reference/L/llms/aianki/collections
python3 auto_anki_agent.py --unprocessed-only --verbose
```

This will:
- ✅ Only process conversations not yet in state file
- ✅ Generate markdown output for review
- ✅ Update state to prevent reprocessing
- ✅ Show progress with verbose output

### 2. Review Proposed Cards

```bash
cat auto_anki_runs/proposed_cards_$(date +%Y-%m-%d).md | less
```

Or open in your text editor to review and select cards to import.

### 3. Import Selected Cards to Anki

Manually copy/paste cards you want to keep from the markdown file into your Anki HTML collections or import dialog.

## Common Commands

### View Processing Progress

```bash
uv run auto-anki-progress           # Show last 12 weeks
uv run auto-anki-progress --weeks 24  # Show more history
```

### Process Specific Month

```bash
python3 auto_anki_agent.py --date-range 2025-10 --verbose
```

### Dry Run (Preview Only)

```bash
python3 auto_anki_agent.py --date-range 2025-10 --max-contexts 5 --dry-run
```

### Process with Custom Model

```bash
python3 auto_anki_agent.py \
  --unprocessed-only \
  --llm-model gpt-5.1 \
  --verbose
```

### Switch to Claude Code Backend

```bash
python3 auto_anki_agent.py \
  --llm-backend claude-code \
  --unprocessed-only \
  --verbose
```

### Force Reprocess Everything in October

```bash
# First, back up and clear state
cp .auto_anki_agent_state.json .auto_anki_agent_state.json.backup
python3 auto_anki_agent.py --date-range 2025-10 --verbose
```

## Output Locations

- **Markdown cards**: `auto_anki_runs/proposed_cards_YYYY-MM-DD.md`
- **JSON cards**: `auto_anki_runs/run-TIMESTAMP/all_proposed_cards.json`
- **Run artifacts**: `auto_anki_runs/run-TIMESTAMP/`
- **State file**: `.auto_anki_agent_state.json`

## Optional Configuration (`auto_anki_config.json`)

To avoid repeating long command lines, you can create an `auto_anki_config.json` file:

- Search order:
  - `AUTO_ANKI_CONFIG` env var (if set)
  - `./auto_anki_config.json` in the current directory
  - `~/.auto_anki_config.json` in your home directory
- CLI flags override anything in the config.
- Relative paths are resolved relative to the config file.

Minimal example:

```json
{
  "chat_root": "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/chatgpt",
  "decks": ["Research Learning", "Technology Learning"]
}
```

With LLM backend configuration:

```json
{
  "chat_root": "...",
  "decks": ["..."],
  "llm_backend": "codex",
  "llm_config": {
    "codex": { "model": "gpt-5.1" },
    "claude-code": { "model_stage2": "claude-opus-4-5-20251101" }
  }
}
```

**Prerequisite:** Anki must be running with AnkiConnect plugin (code: 2055492159).

## Troubleshooting

**No contexts found?**
- Check date range: `--date-range 2025-10`
- Check if already processed: remove state file or don't use `--unprocessed-only`

**Too many/few cards?**
- Enable heuristic filtering: `--use-filter-heuristics --min-score 1.5` (higher = more selective)
- Adjust similarity:
  - String-based: `--similarity-threshold 0.9` (higher = less dedup)
  - Semantic: use `--dedup-method semantic` or `hybrid` plus
    `--semantic-similarity-threshold 0.9`

**Note**: Heuristic filtering is OFF by default. The Stage 1 LLM judges quality directly.

**Want to test without API calls?**
```bash
python3 auto_anki_agent.py --dry-run --max-contexts 3 --verbose
```

## What Gets Scored Highly? (Optional Heuristics)

**Note**: Heuristic scoring is **OFF by default**. The Stage 1 LLM judges quality directly.
Use `--use-filter-heuristics` to enable pre-LLM filtering.

When enabled, the heuristics look for:
- ❓ Questions (starts with what/why/how/when)
- 📚 Definitions ("stands for", "is defined as", "refers to")
- 📝 Bullet points and lists
- 💻 Code blocks
- 📊 Structured content (headings)
- ✍️ Medium-length answers (80-2200 chars)

## File Structure

```
collections/
├── auto_anki_agent.py          # Main script
├── auto_anki/                  # Core Python package
│   ├── contexts.py             # Conversation & ChatTurn dataclasses
│   ├── codex.py                # Prompt builders
│   ├── dedup.py                # Deduplication
│   ├── state.py                # State tracking (v2 schema)
│   ├── progress.py             # TUI progress dashboard
│   └── llm_backends/           # Pluggable LLM backend abstraction
│       ├── base.py             # LLMBackend ABC, LLMConfig
│       ├── codex.py            # Codex CLI backend
│       └── claude_code.py      # Claude Code backend
├── auto_anki_config.json       # Config (optional, includes llm_backend)
├── README_AUTO_ANKI.md         # Full documentation
├── QUICK_START.md              # This file
├── .auto_anki_agent_state.json # Processing state (v2 with seen_conversations)
└── auto_anki_runs/             # Generated outputs
    ├── proposed_cards_2025-11-08.md
    └── run-20251108-125440/
        ├── all_proposed_cards.json
        ├── selected_conversations.json  # Full conversation context
        └── ...
```
