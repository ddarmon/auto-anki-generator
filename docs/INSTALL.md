# Installation Guide

## Prerequisites

### AnkiConnect Plugin (Required)

The tool loads existing cards directly from Anki via the AnkiConnect plugin. **Anki must be running** whenever you use the tool.

1. Open Anki
2. Go to **Tools → Add-ons → Get Add-ons...**
3. Enter code: `2055492159`
4. Restart Anki

Verify it's working:
```bash
curl http://localhost:8765 -X POST -d '{"action": "version", "version": 6}'
# Should return: {"result": 6, "error": null}
```

## Using uv (Recommended)

The easiest way to use this tool is with `uv`, which handles dependencies automatically.

### One-Time Setup

```bash
cd ~/Dropbox/Reference/L/llms/aianki/collections

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the tool (creates isolated environment)
uv pip install -e .

# Optional: Install semantic deduplication dependencies
uv pip install -e ".[semantic]"
```

### Configuration

Create `auto_anki_config.json` in your working directory:

```json
{
  "chat_root": "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/chatgpt",
  "decks": [
    "Research Learning",
    "Technology Learning"
  ]
}
```

**Important:** The `decks` list must match your actual Anki deck names exactly.

### Running the Tool

After installation, you can run it from anywhere:

```bash
# Using uv run (recommended - always uses correct environment)
uv run auto-anki --date-range 2025-10 --max-contexts 20 --verbose

# Or if you've activated the environment
auto-anki --date-range 2025-10 --max-contexts 20 --verbose
```

### Quick Commands

```bash
# Daily run (unprocessed only)
uv run auto-anki --unprocessed-only --verbose

# Process specific month
uv run auto-anki --date-range 2025-10 --verbose

# Dry run to test
uv run auto-anki --date-range 2025-10 --max-contexts 5 --dry-run

# Show all options
uv run auto-anki --help
```

## Performance Improvements

The tool caches card data from AnkiConnect in `.deck_cache/` directory:

- **First run**: Fetches cards from Anki (fast, ~1-5 seconds depending on deck size)
- **Subsequent runs within TTL**: Loads from cache (instant)
- **Cache invalidation**: Automatic when cards are added/modified in Anki
- **Default TTL**: 5 minutes (configurable via `--anki-cache-ttl`)

Semantic deduplication embeddings are also cached for faster duplicate detection.

## Directory Structure

After setup:
```
collections/
├── auto_anki_agent.py          # Main script
├── auto_anki/                  # Core Python package
├── pyproject.toml              # Package config
├── auto_anki_config.json       # Your configuration
├── .deck_cache/                # Card + embedding cache (auto-created)
│   ├── anki_cards_cache.json   # Cached card data from Anki
│   └── embeddings/             # Semantic dedup embeddings
├── .auto_anki_agent_state.json # Processing state
└── auto_anki_runs/             # Output directory
```

## Troubleshooting

### "Cannot connect to Anki"

1. Make sure Anki is running
2. Verify AnkiConnect plugin is installed (code: 2055492159)
3. Test connection:
   ```bash
   curl http://localhost:8765 -X POST -d '{"action": "version", "version": 6}'
   ```

### "No decks specified"

Add a `decks` list to your `auto_anki_config.json` or use `--decks` flag:
```bash
uv run auto-anki --decks "Research Learning" "Technology Learning" --verbose
```

### Command not found: auto-anki

Use `uv run auto-anki` instead of just `auto-anki`.

### Cache issues

Delete the cache and let it regenerate:
```bash
rm -rf .deck_cache
uv run auto-anki --verbose
```

### Want to upgrade dependencies

```bash
uv pip install --upgrade -e .
```

## Development

If you want to modify the script:

```bash
# Make changes to auto_anki/*.py files

# No reinstall needed! Changes take effect immediately
uv run auto-anki --verbose
```

### Running Tests

The project includes a pytest-based test suite:

```bash
# Install dev dependencies (pytest, pytest-cov)
uv pip install pytest pytest-cov

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=auto_anki --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_scoring.py -v
```

**97 tests** cover core functions: scoring, normalization, parsing, date filtering, and deduplication.

## Uninstall

```bash
uv pip uninstall auto-anki-agent
rm -rf .venv .deck_cache .auto_anki_agent_state.json auto_anki_runs/
```
