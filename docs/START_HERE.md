# 🎓 Auto Anki Agent - Start Here!

**Welcome!** This is your quick start guide to the Auto Anki Agent.

## What Is This?

An **end-to-end system** that generates flashcards from your ChatGPT conversations and imports them directly to Anki.

**Total workflow time**: ~10 minutes for 50 high-quality cards 🚀

## Quick Start (5 Steps)

### 1. Install Dependencies

```bash
# Install UI dependencies (for interactive review)
uv pip install -e ".[ui]"

# Install semantic dependencies (recommended for better duplicate detection)
uv pip install -e ".[semantic]"

# Or install both at once
uv pip install -e ".[ui,semantic]"
```

### 2. Install AnkiConnect Plugin

In Anki:
1. Tools → Add-ons → Get Add-ons...
2. Enter code: **2055492159**
3. Click OK
4. Restart Anki

### 3. Test Connection

```bash
# Make sure Anki is running, then:
python3 anki_connect.py
```

You should see:
```
✓ Connected to Anki
✓ AnkiConnect version: 6
✓ Found X decks
✓ AnkiConnect is working correctly!
```

### 4. Generate Cards

```bash
# Generate from your ChatGPT conversations
python3 auto_anki_agent.py --date-range 2025-10 --max-contexts 10 --verbose
```

### 5. Review & Import

```bash
# Launch the interactive UI
./launch_ui.sh
```

Then in the browser:
1. Select the latest run from dropdown
2. Review cards with keyboard shortcuts:
   - `A` - Accept
   - `R` - Reject
   - `E` - Edit
   - `→` - Next card
3. Click "Import All Accepted to Anki"
4. Done! Cards appear in Anki immediately

## Daily Workflow

```bash
# 1. Generate cards from new conversations (1-2 min)
python3 auto_anki_agent.py --unprocessed-only --verbose

# 2. Review and import (5-10 min for 50 cards)
./launch_ui.sh

# 3. Study in Anki!
```

## Keyboard Shortcuts

**In the review UI:**

| Key | Action |
|-----|--------|
| `A` | Accept card |
| `R` | Reject card |
| `E` | Edit card |
| `S` | Skip card |
| `→` | Next card |
| `←` | Previous card |

**Pro tip**: Use keyboard shortcuts for 3x faster review!

## Documentation

- **README_AUTO_ANKI.md** - Complete user guide
- **ANKICONNECT_GUIDE.md** - AnkiConnect setup & workflows
- **UI_README.md** - Interactive UI features
- **PROJECT_STATUS.md** - Current status & features
- **CLAUDE.md** - Technical architecture (for AI assistants)

## Key Features

✅ **Autonomous Card Generation**
- Harvests ChatGPT conversations
- Intelligent context scoring
- LLM-based card generation
- Follows Anki best practices

✅ **Interactive Review**
- Web-based UI (Shiny)
- Keyboard shortcuts
- Source context display
- Progress tracking

✅ **Advanced Features**
- Filter by deck & confidence
- Bulk accept high-confidence
- Track rejection reasons
- Export feedback data
- Semantic deduplication (embeddings-based)

✅ **Direct Anki Import**
- One-click import
- Batch import all accepted
- Auto-create decks
- Duplicate detection
- **30-60x faster** than manual

## Troubleshooting

### "Cannot connect to Anki"

1. ✓ Make sure Anki is running
2. ✓ Install AnkiConnect (code: 2055492159)
3. ✓ Restart Anki after installing
4. ✓ Test: `python3 anki_connect.py`

### "No runs found"

1. ✓ Run card generation first: `python3 auto_anki_agent.py --date-range 2025-10 --verbose`
2. ✓ Check `auto_anki_runs/` directory exists

### "Card already exists (duplicate)"

This is **normal** - AnkiConnect prevents duplicates by default.

**If you want to import anyway:**
- Check "Allow duplicate cards" checkbox in UI

## Performance

**Card Generation:**
- ~2-5 minutes for 24 contexts

**Review (with keyboard shortcuts):**
- ~5 seconds per card
- 50 cards = 4-5 minutes

**AnkiConnect Import:**
- 50 cards (batch) = 2-3 seconds
- vs manual import = 25-50 minutes
- **80-90% time savings!**

## Semantic Deduplication (Recommended) 🚀

**Now enabled by default!** The system automatically uses AI embeddings for better duplicate detection.

### Quick Start

```bash
# For best results, install semantic dependencies (includes FAISS for fast vector search)
uv pip install -e ".[semantic]"
# or
pip install sentence-transformers numpy faiss-cpu

# Then use normally - hybrid mode is now the default!
python3 auto_anki_agent.py --unprocessed-only --verbose
```

**Performance boost with FAISS:**
- First run: Builds embedding cache (~2-5 seconds for 10K cards)
- Subsequent runs: Loads from cache (~100ms) - **20-50x faster!**
- Cache auto-invalidates when deck files are modified

**Without dependencies:**
- Automatically falls back to string-based deduplication
- Shows a warning with installation instructions
- Still works, but won't catch paraphrased duplicates

### Advanced Usage

```bash
# Use semantic only (no string matching)
python3 auto_anki_agent.py --dedup-method semantic --verbose

# Force string-only mode (legacy behavior)
python3 auto_anki_agent.py --dedup-method string --verbose

# Adjust semantic threshold (default 0.85 = 85% similarity)
python3 auto_anki_agent.py --semantic-similarity-threshold 0.90 --verbose

# Use the installed console script (equivalent to calling auto_anki_agent.py)
auto-anki --unprocessed-only --verbose
```

**Why hybrid mode (default)?**
- Catches exact duplicates (string matching)
- Catches paraphrased duplicates (semantic embeddings)
- Best of both worlds!

**Technology stack:**
- **Model:** `all-MiniLM-L6-v2` (fast, local, no API calls)
- **Vector DB:** FAISS IndexFlatIP (exact cosine similarity)
- **Cache:** Persistent embeddings in `.deck_cache/embeddings/`

## Next Steps

After your first successful run:

1. **Adjust settings** - CLI options:
   - `--max-contexts N` - More/fewer contexts per run
   - `--min-score FLOAT` - Quality threshold
   - `--similarity-threshold FLOAT` - String dedup sensitivity
   - `--dedup-method {string,semantic,hybrid}` - Dedup strategy

2. **Enable semantic dedup** - Install `[semantic]` extras for better duplicate detection

3. **Review feedback** - Export feedback data to see rejection patterns

4. **Optimize prompts** - Improve card generation based on review patterns

5. **Explore advanced features**:
   - Semantic deduplication
   - Custom deck routing
   - Tag taxonomy
   - Bulk operations
   - Filter presets

## Get Help

**Check documentation:**
- `README_AUTO_ANKI.md` - Comprehensive guide
- `ANKICONNECT_GUIDE.md` - Setup troubleshooting
- `UI_README.md` - UI features and tips

**Common issues:**
- AnkiConnect not working → See ANKICONNECT_GUIDE.md troubleshooting
- Cards not generating → Check `--verbose` output
- UI not launching → Ensure dependencies installed with `uv pip install -e ".[ui]"`

## Success!

Once you complete your first run, you should have:

✅ Cards generated from conversations
✅ Cards reviewed in interactive UI
✅ Cards imported to Anki
✅ Ready to study with spaced repetition

**You're all set! Happy learning! 🎓**

---

**Need more details?** See PROJECT_STATUS.md for complete feature overview.

**Want to understand the code?** See CLAUDE.md for architecture details.

**Ready for advanced usage?** See FUTURE_DIRECTIONS.md for enhancement ideas.
