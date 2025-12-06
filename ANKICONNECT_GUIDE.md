# AnkiConnect Integration Guide

## Overview

The Auto Anki Agent now supports **direct import to Anki** via the AnkiConnect plugin. No more manual copy/paste!

## Setup

### 1. Install AnkiConnect Plugin

1. Open Anki
2. Go to **Tools** → **Add-ons**
3. Click **Get Add-ons...**
4. Enter code: **2055492159**
5. Click **OK**
6. Restart Anki

### 2. Verify Installation

```bash
# Test AnkiConnect (with Anki running)
python3 anki_connect.py
```

Expected output:
```
Testing AnkiConnect...
============================================================
✓ Connected to Anki
✓ AnkiConnect version: 6
✓ Found X decks:
  - Default: Y cards
  ...
✓ Available note types: Basic, Cloze, ...
============================================================
✓ AnkiConnect is working correctly!
```

### 3. Launch Enhanced UI

```bash
./launch_ui.sh
```

The UI will show connection status in the AnkiConnect panel:
- **✓ Connected (v6)** - Ready to import (green)
- **✗ Anki not running** - Start Anki (red)
- **✗ Cannot connect** - Install AnkiConnect (red)

## Features

### 1. Connection Status Indicator

**Real-time status:**
- **✓ Connected (v6)** - Anki is running, AnkiConnect is working
- **✗ Anki not running** - Start Anki to enable import
- **⚠ AnkiConnect module not available** - Install anki_connect.py

**Auto-refresh:**
- Status updates when you interact with AnkiConnect features
- Check manually by clicking any import button

### 2. Import Current Card

**Single-card import:**
1. Review a card you like
2. Click **"Import Current Card to Anki"**
3. See status: `✓ Imported to Technology Learning (ID: 1234567890123)`

**What happens:**
- Card is added to the specified deck
- Deck is created automatically if it doesn't exist
- Card is automatically marked as "accepted" in the review session
- Duplicate detection prevents re-importing the same card

**Status messages:**
- `✓ Imported to [Deck] (ID: ...)` - Success
- `⚠ Card already exists in Anki (duplicate)` - Duplicate detected
- `✗ Anki is not running. Please start Anki.` - Connection issue

### 3. Batch Import All Accepted

**Bulk import workflow:**
1. Review cards and accept the good ones (A key or Accept button)
2. Optionally edit cards before accepting
3. Click **"Import All Accepted to Anki"**
4. See status: `✓ Imported 18 cards, 2 duplicates skipped`

**What happens:**
- All accepted and edited cards are imported at once
- Missing decks are created automatically
- Duplicate cards are detected and skipped
- Progress summary shows imported/duplicates/failed counts

**Status messages:**
- `✓ Imported X cards` - All successful
- `✓ Imported X cards, Y duplicates skipped` - Some duplicates
- `✓ Imported X cards, Y failed` - Some errors
- `⚠ No accepted cards to import` - Nothing to import

### 4. Duplicate Detection

**Smart duplicate handling:**
- Uses Anki's built-in duplicate detection
- Checks for exact matches in front/back content
- Controlled by "Allow duplicate cards" checkbox

**Checkbox options:**
- **Unchecked** (default): Skip duplicates, show warning
- **Checked**: Import anyway, create duplicate cards

**When to allow duplicates:**
- Testing the import functionality
- Intentionally creating similar cards with different wording
- Re-importing after editing cards significantly

**When to prevent duplicates:**
- Normal workflow (recommended)
- Cards already exist in Anki
- Avoiding clutter in decks

### 5. Auto-Create Decks

**Automatic deck creation:**
- If deck doesn't exist, it's created automatically
- Supports nested decks (e.g., "Parent::Child")
- No manual deck setup required

**Example:**
```
Card deck: "Machine Learning::Neural Networks"
→ Creates "Machine Learning" deck
→ Creates "Neural Networks" subdeck under it
→ Imports card to the subdeck
```

## Usage Workflows

### Workflow 1: Review → Import Individual Cards

**Best for:** Careful review with immediate import

1. Launch UI and select run
2. Review cards one by one (keyboard shortcuts!)
3. For good cards: Click "Import Current Card"
4. Card is imported AND marked as accepted
5. Move to next card
6. Repeat

**Advantages:**
- Immediate feedback
- See each card in Anki right away
- No need to remember which cards to import later

### Workflow 2: Review → Batch Import

**Best for:** Fast review, import at end

1. Launch UI and select run
2. Review all cards, accepting good ones (A key)
3. (Optional) Use bulk accept for high-confidence cards
4. When done reviewing, click "Import All Accepted"
5. All cards imported at once

**Advantages:**
- Faster review (no waiting for import)
- See summary of what was imported
- Can review decisions before importing

### Workflow 3: Hybrid

**Best for:** Flexibility

1. Import some cards individually as you go
2. Accept others for batch import later
3. Use both buttons as needed

**Advantages:**
- Immediate import for must-have cards
- Batch import for routine cards
- Maximum control

## Troubleshooting

### "✗ Anki is not running"

**Problem:** Anki application is not open

**Solution:**
1. Launch Anki
2. Wait for it to fully load
3. Try import again

### "✗ Cannot connect to Anki"

**Problem:** AnkiConnect plugin not installed or not responding

**Solutions:**
1. Install AnkiConnect plugin (code: 2055492159)
2. Restart Anki after installation
3. Check Anki → Tools → Add-ons shows "AnkiConnect"
4. Disable firewall/antivirus blocking localhost:8765

### "⚠ Card already exists in Anki (duplicate)"

**Problem:** Card with same content already in deck

**Solutions:**
- This is normal! Card won't be re-imported
- If you want duplicates: Check "Allow duplicate cards"
- If card was edited: Import with duplicates allowed, delete old one in Anki

### "✗ Anki error: ..."

**Problem:** Anki returned an error

**Common causes:**
- Invalid deck name (use alphanumeric, avoid special chars)
- Anki is busy (wait and try again)
- Anki database locked (close other Anki windows)

**Solutions:**
1. Check the error message for clues
2. Try importing a different card
3. Restart Anki
4. If persistent, export to JSON and import manually

### Import succeeds but card not visible in Anki

**Problem:** Card imported to unexpected deck

**Solution:**
1. In Anki, go to Browse (B key)
2. Search for: `added:1` (cards added today)
3. Check which deck the card went to
4. Move to correct deck if needed

## Technical Details

### AnkiConnect API

**Endpoint:** `http://localhost:8765`

**Actions used:**
- `version` - Check connection
- `deckNames` - List decks
- `createDeck` - Create missing decks
- `addNote` - Add single card
- `addNotes` - Add multiple cards (batch)
- `findNotes` - Search for duplicates
- `notesInfo` - Get card details

### Card Format

**Auto Anki cards are converted to:**
- **Note type:** Basic
- **Front field:** Card front text
- **Back field:** Card back text
- **Tags:** Card tags (if any)
- **Deck:** Card deck (created if needed)

### Batch Import Performance

**Batch import is efficient:**
- Single API call for all cards
- Anki processes them in one transaction
- Much faster than individual imports

**Expected performance:**
- 10 cards: ~1 second
- 50 cards: ~2-3 seconds
- 100 cards: ~5 seconds

### Error Handling

**Robust error handling:**
- Connection errors are caught and displayed
- Individual card failures don't stop batch import
- Detailed error messages help debugging
- Graceful degradation (JSON export still works)

## Best Practices

### 1. Always Review Before Importing

**Don't blindly import everything!**
- Review each card (or use bulk accept wisely)
- Edit cards to improve quality
- Reject poor cards
- Import only what you want to study

### 2. Use Meaningful Deck Names

**Good deck names:**
- "Machine Learning"
- "Technology::Python"
- "Research::Statistics"

**Avoid:**
- Special characters: `!@#$%`
- Very long names
- Names with just numbers

### 3. Leverage Duplicate Detection

**Default behavior (no duplicates) is best:**
- Prevents clutter
- Avoids studying same card twice
- Keeps decks clean

**Allow duplicates only when:**
- Intentionally creating similar cards
- Testing import functionality
- Cards are edited significantly from original

### 4. Create Deck Hierarchy

**Use nested decks:**
```
Technology
├── Technology::Python
├── Technology::JavaScript
└── Technology::DevOps
```

**Benefits:**
- Better organization
- Study parent deck to include all subdecks
- Easy to see topic coverage

### 5. Monitor Import Results

**Always check the status message:**
- Verify count matches expectations
- Note any duplicates or failures
- Investigate errors promptly

**In Anki:**
- Browse deck to see imported cards
- Check that formatting is correct
- Verify tags are applied

## Advanced Usage

### Testing Connection

**From command line:**
```bash
python3 anki_connect.py
```

**From Python:**
```python
from anki_connect import AnkiConnectClient

client = AnkiConnectClient()

if client.check_connection():
    print("Connected!")
    print(f"Decks: {client.get_deck_names()}")
else:
    print("Not connected")
```

### Custom Import Logic

**Modify `anki_connect.py` for custom needs:**
- Add support for Cloze cards
- Include media files (images, audio)
- Set custom note types
- Apply advanced tagging logic

### Querying Existing Cards

**Get cards for deduplication:**
```python
client = AnkiConnectClient()
cards = client.get_existing_cards_for_dedup("Technology Learning")
print(f"Found {len(cards)} existing cards")
```

## Comparison to Manual Import

### Before AnkiConnect

**Manual workflow:**
1. Review cards in UI
2. Export accepted cards to JSON
3. Open JSON file
4. Copy card content
5. Open Anki
6. Create note manually
7. Paste front
8. Paste back
9. Add tags
10. Choose deck
11. Repeat for each card

**Time:** ~30-60 seconds per card

### With AnkiConnect

**Automated workflow:**
1. Review cards in UI
2. Click "Import All Accepted"
3. Done!

**Time:** ~1 second for all cards

**Improvement:** **30-60x faster!**

## Future Enhancements

Possible future additions:

1. **Cloze card support** - Import cloze deletion cards
2. **Media import** - Include images and audio
3. **Custom note types** - Support for custom card templates
4. **Reverse cards** - Option to create reverse cards (back→front)
5. **Update existing cards** - Modify cards instead of creating new ones
6. **Sync stats** - Pull study statistics back from Anki
7. **Deck recommendations** - Auto-suggest which deck based on content

## Resources

- **AnkiConnect Documentation**: https://foosoft.net/projects/anki-connect/
- **Anki Manual**: https://docs.ankiweb.net/
- **Auto Anki Agent**: See `UI_README.md`, `README_AUTO_ANKI.md`

## Support

**If AnkiConnect isn't working:**

1. Check Anki is running
2. Verify AnkiConnect is installed (Tools → Add-ons)
3. Test connection: `python3 anki_connect.py`
4. Check localhost:8765 is not blocked by firewall
5. Restart Anki
6. Reinstall AnkiConnect if necessary

**If you encounter issues:**
- Check error messages carefully
- Test with a single card first
- Try "Allow duplicates" if duplicate detection is failing
- Export to JSON as fallback
- Restart both Anki and the UI

---

**Status:** Production-ready ✅

**Last updated:** 2025-12-06

**AnkiConnect version:** 6
