# Auto Anki Agent

This repository contains the Auto Anki Agent, an agentic pipeline that
turns ChatGPT conversations into high‑quality Anki cards and provides a
review UI with direct AnkiConnect integration.

**Minimal end-to-end workflow (using `uv run`):**

``` bash
# 1. Install (from repo root)
uv pip install -e ".[ui,semantic]"

# 2. Import conversations from ChatGPT/Claude export (creates markdown files)
uv run auto-anki-import ~/Downloads/conversations.json

# 3. Generate cards from new conversations
uv run auto-anki --unprocessed-only --verbose

# 4. Review and import to Anki in the browser
./launch_ui.sh
```

## Optional configuration (`auto_anki_config.json`)

You can avoid repeating common CLI flags by creating a JSON config:

- Search order:
  - Path from `AUTO_ANKI_CONFIG` (if set)
  - `./auto_anki_config.json` in the current working directory
  - `~/.auto_anki_config.json` in your home directory
- CLI flags still override any values set in the config file.
- Relative paths in the config are resolved relative to the config file's directory.

Example `auto_anki_config.json` next to this repo:

```json
{
  "chat_root": "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/chatgpt",
  "decks": ["Research Learning", "Technology Learning"],
  "state_file": ".auto_anki_agent_state.json",
  "output_dir": "auto_anki_runs",
  "cache_dir": ".deck_cache",

  "llm_backend": "codex",
  "llm_config": {
    "codex": {
      "model": "gpt-5.1",
      "reasoning_effort_stage1": "low",
      "reasoning_effort_stage2": "high"
    },
    "claude-code": {
      "model_stage1": "claude-haiku-4-5-20251001",
      "model_stage2": "claude-opus-4-5-20251101"
    }
  }
}
```

The `llm_backend` key selects which agentic CLI tool to use (`codex` or `claude-code`).
Use `--llm-backend` on the CLI to override.

## Installation (Quick)

Using `uv` (recommended):

``` bash
# From the repo root
uv pip install -e ".[ui,semantic]"
```

Or with plain `pip` (less isolated, but works):

``` bash
pip install -e ".[ui,semantic]"
```

This installs:

-   The `auto-anki` console script (generate flashcards)
-   The `auto-anki-import` console script (import ChatGPT/Claude JSON exports)
-   The `auto-anki-progress` console script (view processing progress)
-   Core dependencies (BeautifulSoup, json-repair)
-   Optional extras:
    -   `ui` -- Shiny UI + plotting for interactive review
    -   `semantic` -- SentenceTransformers + FAISS for semantic dedup

For more detail (platform notes, troubleshooting), see
`docs/INSTALL.md`.

## Examples

**Basic usage (generate cards from a date range):**

``` bash
python3 auto_anki_agent.py --date-range 2025-10 --max-contexts 10 --verbose
```

**Daily workflow using the legacy script:**

``` bash
# 1. Generate cards from new conversations
python3 auto_anki_agent.py --unprocessed-only --verbose

# 2. Launch the review UI and import to Anki
./launch_ui.sh
```

**Using the installed console script (recommended via `uv run`):**

``` bash
uv run auto-anki --unprocessed-only --verbose
```

**Tuning deduplication behaviour:**

``` bash
# Hybrid (default): string + semantic
python3 auto_anki_agent.py --dedup-method hybrid --verbose

# Semantic only
python3 auto_anki_agent.py --dedup-method semantic --verbose

# String only (no embeddings)
python3 auto_anki_agent.py --dedup-method string --verbose
```

**Switching LLM backends:**

``` bash
# Use Codex (default)
uv run auto-anki --unprocessed-only --verbose

# Use Claude Code instead
uv run auto-anki --llm-backend claude-code --verbose

# Override model for a specific run
uv run auto-anki --llm-model gpt-5.1 --verbose
```

**Importing ChatGPT/Claude conversation exports:**

``` bash
# Import conversations.json to markdown directory
uv run auto-anki-import ~/Downloads/conversations.json

# Import with verbose output
uv run auto-anki-import ~/Downloads/conversations.json -v

# Import and immediately generate cards
uv run auto-anki-import ~/Downloads/conversations.json --run
```

**Monitoring batch progress:**

``` bash
# View processing progress dashboard
uv run auto-anki-progress

# Estimate time to complete (analyzes batch log)
./scripts/estimate_completion.sh
```
