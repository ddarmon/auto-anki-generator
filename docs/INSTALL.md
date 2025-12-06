# Installation Guide

## Using uv (Recommended)

The easiest way to use this tool is with `uv`, which handles dependencies automatically.

### One-Time Setup

```bash
cd ~/Dropbox/Reference/L/llms/aianki/collections

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the tool (creates isolated environment)
uv pip install -e .
```

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

# Process specific month with mini model
uv run auto-anki --date-range 2025-10 --codex-model gpt-5-codex-mini

# Dry run to test
uv run auto-anki --date-range 2025-10 --max-contexts 5 --dry-run

# Show all options
uv run auto-anki --help
```

## Performance Improvements

The tool now caches parsed HTML decks in `.deck_cache/` directory. This makes subsequent runs much faster:

- **First run**: Parses all HTML files (slow, ~10-30 seconds)
- **Subsequent runs**: Loads from cache (fast, <1 second)
- **Cache invalidation**: Automatic when HTML files are modified

## Directory Structure

After setup:
```
collections/
├── auto_anki_agent.py          # Main script
├── pyproject.toml              # Package config
├── .deck_cache/                # Parsed card cache (auto-created)
│   ├── Research_Learning_cards.json
│   ├── Technology_Learning_cards.json
│   └── Moody_s_Learning_cards.json
├── .auto_anki_agent_state.json # Processing state
├── auto_anki_runs/             # Output directory
└── *.html                      # Your existing Anki decks
```

## Troubleshooting

### Command not found: auto-anki

Use `uv run auto-anki` instead of just `auto-anki`.

### Still hanging on HTML parsing

Delete the cache and regenerate:
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
# Make changes to auto_anki_agent.py

# No reinstall needed! Changes take effect immediately
uv run auto-anki --verbose
```

## Uninstall

```bash
uv pip uninstall auto-anki-agent
rm -rf .venv .deck_cache .auto_anki_agent_state.json auto_anki_runs/
```
