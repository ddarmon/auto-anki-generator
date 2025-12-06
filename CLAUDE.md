# CLAUDE.md - Auto Anki Agent Project Guide

This document helps Claude (or any AI assistant) quickly understand and
work with the Auto Anki Agent codebase.

## Project Overview

**Auto Anki Agent** is an autonomous pipeline that generates
high-quality Anki flashcards from ChatGPT conversation exports. It uses
LLM-based decision-making (via `codex exec`) to intelligently select and
format learning-worthy content as spaced repetition cards.

### Core Purpose

Transform casual learning conversations into actionable flashcards
following Anki best practices (minimum information principle, atomic
facts, clear context, etc.).

### Key Innovation

Rather than using simple templates or rules, the system delegates card
generation to an LLM "decision layer" that understands:

-   What makes good flashcard material
-   How to avoid duplicates
-   When to skip low-quality content
-   How to format cards for optimal learning

## Architecture

    ┌─────────────────────────────────────────────────────────────────┐
    │                     Auto Anki Agent Pipeline                    │
    └─────────────────────────────────────────────────────────────────┘

    1. HARVEST PHASE
       ├─ Parse existing Anki decks (HTML) → Card objects
       ├─ Scan ChatGPT export folder (markdown) → ChatTurn objects
       └─ Apply filters: DateRangeFilter, StateTracker

    2. SCORING PHASE
       ├─ Heuristic scoring for each ChatTurn
       │  ├─ Question signals (what/why/how)
       │  ├─ Definition signals (is defined as, stands for)
       │  ├─ Structure signals (bullet points, code blocks)
       │  └─ Quality signals (length, complexity)
       └─ Threshold filtering (min_score)

    3. DEDUPLICATION PHASE
       ├─ String similarity vs existing cards (SequenceMatcher)
       ├─ Threshold-based filtering (similarity_threshold)
       └─ Remove redundant contexts

    4. BATCHING PHASE
       ├─ Group contexts into batches (contexts_per_run)
       ├─ Build prompts with:
       │  ├─ System instructions (Anki best practices)
       │  ├─ Existing cards summary (for dedup awareness)
       │  └─ Context batch (user-assistant pairs)
       └─ Cap total contexts (max_contexts)

    5. CODEX PHASE
       ├─ Call `codex exec` with JSON contract
       ├─ LLM decides: create card / skip / note
       ├─ Parse JSON response (with repair_json fallback)
       └─ Collect proposed cards

    6. OUTPUT PHASE
       ├─ Save artifacts to run directory
       ├─ Generate markdown preview
       ├─ Generate JSON for automation
       └─ Update state file

## Key Components

### Data Structures

#### `Card` (dataclass)

Represents an existing or proposed Anki flashcard.

``` python
@dataclass
class Card:
    deck: str           # Target deck name
    front: str          # Question/prompt
    back: str           # Answer/explanation
    tags: List[str]     # Anki tags
    url: Optional[str]  # Source URL (if from chat)
```

#### `ChatTurn` (dataclass)

Represents a user-assistant exchange from ChatGPT.

``` python
@dataclass
class ChatTurn:
    user_prompt: str         # User's question/request
    assistant_answer: str    # Assistant's response
    context_id: str          # Unique hash
    source_path: Path        # Originating file
    score: float             # Heuristic quality score
    signals: Dict[str, Any]  # Why it scored this way
    url: Optional[str]       # Chat URL if available
```

### Core Classes

#### `DateRangeFilter`

Filters conversation files by date.

-   Supports formats: `2025-10` (month), `2025-10-01:2025-10-31` (range)
-   Extracts dates from filenames like `2025-10-15_topic.md`

#### `StateTracker`

Manages `.auto_anki_agent_state.json` for incremental processing.

-   Tracks processed files
-   Stores seen context IDs
-   Records run history
-   Enables `--unprocessed-only` mode

### Key Functions

#### `parse_decks(deck_files) -> List[Card]`

Parses HTML Anki deck exports using BeautifulSoup.

-   Extracts front/back from `<td>` elements
-   Handles malformed HTML gracefully
-   Returns list of existing cards for deduplication

#### `harvest_chat_contexts(...) -> List[ChatTurn]`

Walks chat export directory and extracts user-assistant pairs.

-   Parses markdown conversation files
-   Applies date range filtering
-   Applies state-based filtering (unprocessed-only)
-   Scores each context using `detect_signals()`

#### `detect_signals(user_text, assistant_text) -> Tuple[float, Dict]`

Heuristic scoring function that identifies quality signals:

**Positive signals:**

-   `question_like`: Question words (what/why/how)
-   `definition_like`: Definition patterns (is defined as)
-   `bullet_points`: Lists and structure
-   `code_blocks`: Technical content
-   `headings`: Organized content
-   `medium_length`: Goldilocks zone (80-2200 chars)

**Negative signals:**

-   `too_short`: Less than 80 chars
-   `too_long`: More than 2200 chars

Returns: `(total_score, signals_dict)`

#### `deduplicate / prune_contexts(contexts, cards, args) -> List[ChatTurn]`

Filters contexts that are too similar to existing cards.

-   String-based dedup: `SequenceMatcher.ratio()` over normalized text
-   Semantic dedup: optional SentenceTransformers embeddings via
    `--dedup-method semantic` / `hybrid`
-   String threshold: `--similarity-threshold` (default: 0.82)
-   Semantic threshold: `--semantic-similarity-threshold` (default: 0.85)
-   Helps avoid generating redundant cards

#### `build_codex_prompt(cards, contexts, args) -> str`

Constructs the prompt sent to `codex exec`.

**Prompt structure:**

1.  System instructions (Anki best practices)
2.  JSON contract specification
3.  Existing cards summary (compact)
4.  Contexts to evaluate (full detail)
5.  Output format requirements

#### `call_codex_exec(prompt, args) -> Dict`

Invokes `codex exec` subprocess and parses response.

-   Uses `json_repair` for malformed responses
-   Supports `--dry-run` mode (skip actual call)
-   Handles error cases gracefully

#### `write_markdown_output(cards, output_path)`

Generates human-readable card preview.

-   Grouped by deck
-   Shows confidence scores
-   Includes source metadata
-   Easy to review before import

## File Organization

    auto-anki-generator/
    ├── auto_anki_agent.py           # Main script (1200+ lines)
    ├── anki_review_ui.py            # Interactive review UI (Shiny app, 1050+ lines)
    ├── anki_connect.py              # AnkiConnect HTTP client (437 lines)
    ├── launch_ui.sh                 # Launch script for review UI
    ├── CLAUDE.md                    # This file
    ├── README_AUTO_ANKI.md          # User documentation
    ├── UI_README.md                 # Interactive UI documentation
    ├── ANKICONNECT_GUIDE.md         # AnkiConnect setup and workflows
    ├── UI_ENHANCEMENTS_SUMMARY.md   # UI enhancement technical details
    ├── INTEGRATION_COMPLETE.md      # AnkiConnect integration summary
    ├── QUICK_START.md               # Quick reference
    ├── INSTALL.md                   # Setup instructions
    ├── FUTURE_DIRECTIONS.md         # Roadmap (comprehensive!)
    ├── pyproject.toml               # uv project config
    ├── uv.lock                      # Dependency lock
    ├── .auto_anki_agent_state.json  # Runtime state (git-ignored)
    └── auto_anki_runs/              # Output directory
        ├── proposed_cards_YYYY-MM-DD.md
        └── run-YYYYMMDD-HHMMSS/
            ├── selected_contexts.json
            ├── codex_response_chunk_01.json
            ├── codex_parsed_response_chunk_01.json
            ├── all_proposed_cards.json
            ├── accepted_cards_TIMESTAMP.json      # User-reviewed cards
            └── feedback_data_TIMESTAMP.json       # Review feedback

## Common Tasks

### Understanding the Current System

**What to read first:**

1.  This file (CLAUDE.md) - architecture overview
2.  `auto_anki_agent.py` lines 1-150 - core data structures
3.  `detect_signals()` function - scoring logic
4.  `build_codex_prompt()` function - LLM interface

**Key configuration points (via CLI defaults):**

-   `DEFAULT_MAX_CONTEXTS = 24` - contexts per run
-   `DEFAULT_CONTEXTS_PER_RUN = 8` - batch size
-   `DEFAULT_MIN_SCORE = 1.2` - quality threshold
-   `DEFAULT_SIMILARITY_THRESHOLD = 0.82` - string dedup threshold
-   `DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.85` - semantic dedup threshold
-   `DEFAULT_PER_FILE_LIMIT = 3` - max contexts per conversation

### Debugging a Run

**Check these in order:**

1.  State file: `.auto_anki_agent_state.json` - what's processed?
2.  Run directory: `auto_anki_runs/run-YYYYMMDD-HHMMSS/`
    -   `selected_contexts.json` - what got scored/selected?
    -   `codex_response_chunk_*.json` - raw LLM output
    -   `codex_parsed_response_chunk_*.json` - parsed cards
3.  Markdown output: `proposed_cards_YYYY-MM-DD.md` - final result

**Common issues:**

-   "No contexts found" → Check date range, state file, scoring
    threshold
-   "No cards generated" → LLM rejected all contexts (inspect codex
    response)
-   "Duplicate cards" →
    -   For string-only runs: lower `--similarity-threshold`
    -   For semantic runs: lower `--semantic-similarity-threshold` or
        switch to `--dedup-method hybrid`
-   "Low quality cards" → Raise min_score threshold

### Adding a New Heuristic Signal

1.  Add detection logic to `detect_signals()`:

    ``` python
    signals['has_examples'] = bool(re.search(
        r'for example|e\.g\.|such as',
        assistant_text, re.I
    ))
    ```

2.  Add weight to score calculation:

    ``` python
    score += 0.6 * signals['has_examples']
    ```

3.  Test with `--dry-run` to see impact on scoring

### Modifying the Codex Prompt

**Location:** `build_codex_prompt()` function

**Sections to modify:**

-   **System instructions**: Anki best practices, card quality rules
-   **Contract specification**: JSON schema for LLM response
-   **Examples**: Few-shot learning examples (currently inline)
-   **Existing cards format**: How cards are presented for dedup

**Best practices:**

-   Use HEREDOC-style strings for readability
-   Keep instructions concise (token cost)
-   Be explicit about JSON format
-   Include examples of good/bad cards

### Processing a New Data Source

**To add support for new input format (e.g., PDF highlights):**

1.  Create parser function:

    ``` python
    def parse_pdf_highlights(pdf_path: Path) -> List[ChatTurn]:
        # Extract highlights
        # Convert to ChatTurn objects
        # Return list
    ```

2.  Integrate into `harvest_chat_contexts()`:

    ``` python
    if file_path.suffix == '.pdf':
        contexts.extend(parse_pdf_highlights(file_path))
    ```

3.  Update state tracking to handle new file types

4.  Test deduplication against existing cards

## Important Patterns & Conventions

### Error Handling Philosophy

**The script is defensive but not silent:**

-   HTML parsing: Gracefully handle malformed decks
-   JSON parsing: Use `json_repair` for broken responses
-   File I/O: Check existence, create directories as needed
-   LLM responses: Validate structure, provide helpful errors

### State Management

**Incremental processing is key:**

-   Never reprocess the same conversation file
-   Track seen context_ids to avoid duplicates
-   Support `--unprocessed-only` for daily runs
-   Allow state reset for full reprocessing

### Scoring Design

**Current approach: Additive heuristics**

-   Each signal contributes to total score
-   Threshold-based filtering (`min_score`)
-   Transparent (signals dict shows why)

**Future approach (from FUTURE_DIRECTIONS.md):**

-   Semantic embeddings for deduplication
-   Two-stage LLM pipeline (fast filter → slow generation)
-   Active learning from user feedback

### Prompt Engineering

**Current codex prompt includes:**

-   Anki best practices (minimum information principle, etc.)
-   JSON contract (strict schema)
-   Existing cards (for dedup awareness)
-   Quality guidelines (atomic facts, clear context)

**Design principles:**

-   Be prescriptive about card quality
-   Provide negative examples (what NOT to do)
-   Allow LLM to skip low-quality contexts
-   Require rationale for decisions

## CLI Usage Patterns

### Typical Workflows

**Daily processing:**

``` bash
python3 auto_anki_agent.py --unprocessed-only --verbose
```

**Month-end review:**

``` bash
python3 auto_anki_agent.py --date-range 2025-11 --verbose
```

**Exploratory/testing:**

``` bash
python3 auto_anki_agent.py --dry-run --max-contexts 5 --verbose
```

**High-volume batch:**

``` bash
python3 auto_anki_agent.py \
  --date-range 2025-11 \
  --max-contexts 50 \
  --contexts-per-run 10 \
  --codex-model gpt-5-codex
```

### Important Flags

-   `--dry-run`: Build prompts but don't call codex (FREE)
-   `--verbose`: Show progress and decisions (HELPFUL)
-   `--unprocessed-only`: Skip processed files (INCREMENTAL)
-   `--date-range`: Time-based filtering (FOCUSED)
-   `--output-format {json,markdown,both}`: Output preference

## Integration Points

### Input: ChatGPT Exports

**Expected format:**

-   Markdown files with conversation structure
-   User/assistant turn demarcation
-   Optional URL metadata
-   Date-prefixed filenames (e.g., `2025-11-08_topic.md`)

**Parse logic:**

-   Regex-based turn detection
-   Section headers for conversation boundaries
-   Timestamp extraction for filtering

### Input: Anki HTML Decks

**Expected format:**

-   HTML table structure (`<table>`, `<tr>`, `<td>`)
-   First `<td>` = Front, second `<td>` = Back
-   Deck name from filename

**Parse logic:**

-   BeautifulSoup HTML parsing
-   Text extraction with `.get_text()`
-   Deduplication via string similarity

### Output: Codex Exec

**Interface:**

-   Subprocess call to `codex exec`
-   Stdin: Prompt text
-   Stdout: JSON response
-   Args: `--model`, `--reasoning-effort`, custom args

**Response schema:**

``` json
{
  "generated_cards": [
    {
      "context_id": "hash123",
      "deck": "Research_Learning",
      "front": "What is...",
      "back": "...",
      "card_style": "basic",
      "confidence": 0.85,
      "tags": ["ml", "concepts"],
      "notes": "Rationale for creation"
    }
  ],
  "skipped_contexts": [
    {
      "context_id": "hash456",
      "reason": "Too vague, lacks concrete facts"
    }
  ]
}
```

### Output: Generated Cards

**Markdown format:**

-   Human-readable sections
-   Grouped by deck
-   Includes metadata and rationale
-   Easy to review and selectively import

**JSON format:**

-   Machine-readable
-   Full card objects
-   Structured for automation
-   Includes all metadata

## Interactive Review UI & AnkiConnect

### Overview

The project now includes a **production-ready** interactive review UI with direct Anki integration!

**Launch the UI:**
```bash
./launch_ui.sh  # Auto-detects AnkiConnect, launches Shiny app
```

### Key Features

1.  **Interactive Review** ✅
    - Card-by-card review (accept/reject/edit/skip)
    - Keyboard shortcuts (A/R/E/S, arrow keys)
    - Source context display
    - Progress tracking and statistics

2.  **Advanced Filtering** ✅
    - Filter by deck
    - Filter by confidence threshold
    - Combine multiple filters
    - Jump to filtered results

3.  **Bulk Operations** ✅
    - Auto-accept high-confidence cards (configurable threshold)
    - Batch import all accepted cards
    - Rejection reason tracking (7 predefined + custom)

4.  **AnkiConnect Integration** ✅
    - Real-time connection status indicator
    - Import current card with one click
    - Batch import all accepted cards
    - Duplicate detection (configurable)
    - Auto-create missing decks
    - **30-60x faster** than manual import

### Architecture

**Components:**
- `anki_review_ui.py` - Shiny web app (1050+ lines)
  - `CardReviewSession` class manages state
  - Reactive UI updates
  - Keyboard shortcut handling

- `anki_connect.py` - HTTP client for AnkiConnect API (437 lines)
  - `AnkiConnectClient` class
  - Methods: `add_note()`, `import_cards_batch()`, `create_deck()`
  - Robust error handling

**Workflow:**
1. User reviews cards in web UI
2. Accepts/rejects/edits cards with keyboard shortcuts
3. Clicks "Import All Accepted to Anki"
4. `anki_review_ui.py` → `anki_connect.py` → HTTP POST localhost:8765
5. AnkiConnect plugin adds cards to Anki database
6. Cards appear in Anki immediately

### Documentation

- `UI_README.md` - Complete UI feature documentation
- `ANKICONNECT_GUIDE.md` - Setup, workflows, troubleshooting
- `UI_ENHANCEMENTS_SUMMARY.md` - Technical implementation details
- `INTEGRATION_COMPLETE.md` - Quick start and integration summary

## Future Directions

This project has **significant** planned enhancements documented in
`FUTURE_DIRECTIONS.md` (1600+ lines!).

### Highest Priority (from roadmap)

1.  **Semantic deduplication** - Embeddings + FAISS/ChromaDB for better
    duplicate detection
2.  ~~**Interactive review mode**~~ ✅ **DONE!**
3.  **Two-stage LLM pipeline** - Fast pre-filter + slow generation (70%
    cost reduction)
4.  ~~**AnkiConnect integration**~~ ✅ **DONE!**
5.  **Active learning** - Feedback loop to improve quality over time

### Architecture Improvements

-   **Plugin system** - Extensible scorers, parsers, exporters
-   **Config management** - YAML-based configuration, profiles
-   **SQLite backend** - Replace JSON state file, better scalability
-   **Parallel processing** - Multiprocessing for harvesting/dedup
-   **Observability** - Structured logging, metrics, dashboards

### Intelligence Upgrades

-   **Context clustering** - Topic modeling, coherent card sets
-   **Dependency ordering** - Prerequisite detection, learning paths
-   **Novelty detection** - Prioritize underrepresented topics
-   **Quality validation** - Post-generation auto-fix pipeline
-   **Domain-specific agents** - Math, programming, language learning

See `FUTURE_DIRECTIONS.md` for detailed proposals with code examples.

## Working with This Codebase

### When Asked to Add Features

**Consider these priorities:**

1.  **Does it improve card quality?** (scoring, prompts, validation)
2.  **Does it reduce friction?** (automation, UX, integration)
3.  **Does it scale better?** (performance, cost, reliability)
4.  **Is it in FUTURE_DIRECTIONS.md?** (check roadmap first)

**Before implementing:**

1.  Read the relevant section in `FUTURE_DIRECTIONS.md`
2.  Check if it affects core data structures (Card, ChatTurn)
3.  Consider state migration if changing state file format
4.  Think about backward compatibility

### When Debugging Issues

**Start with observation:**

1.  Run with `--verbose` to see decisions
2.  Use `--dry-run` to inspect prompts
3.  Check state file for unexpected state
4.  Review run artifacts in `auto_anki_runs/`

**Common debugging targets:**

-   Scoring logic: `detect_signals()`
-   Deduplication: `deduplicate()`
-   Prompt construction: `build_codex_prompt()`
-   Response parsing: `call_codex_exec()`, `json_repair`

### When Optimizing

**Current bottlenecks (from FUTURE_DIRECTIONS.md):**

1.  Sequential file parsing (use multiprocessing)
2.  O(n) deduplication (use embeddings + vector DB)
3.  Full state file reload (use SQLite)
4.  Expensive LLM calls (two-stage pipeline)

**Cost optimization:**

-   Prompt caching (Anthropic ephemeral cache)
-   Batch similar contexts
-   Pre-filter with cheap model
-   Compress card summaries in prompt

## Development Philosophy

**This tool is:**

-   **Opinionated** about card quality (Anki best practices)
-   **Transparent** about decisions (signals, scores, rationale)
-   **Incremental** by design (state tracking, unprocessed-only)
-   **LLM-native** (trust codex for complex decisions)
-   **Extensible** (planned plugin system)

**This tool is NOT:**

-   A generic flashcard generator
-   A real-time interactive system (batch-oriented)
-   Fully automated import (requires human review)
-   A one-size-fits-all solution (intentionally configurable)

## Quick Reference

### Key Files

-   `auto_anki_agent.py` - Main script (all logic)
-   `FUTURE_DIRECTIONS.md` - Comprehensive roadmap
-   `.auto_anki_agent_state.json` - Runtime state

### Key Functions

-   `parse_decks()` - Load existing cards
-   `harvest_chat_contexts()` - Extract conversations
-   `detect_signals()` - Score contexts
-   `deduplicate()` - Filter similar content
-   `build_codex_prompt()` - LLM interface
-   `call_codex_exec()` - LLM invocation

### Key Data Structures

-   `Card` - Flashcard (front/back/deck/tags)
-   `ChatTurn` - Conversation exchange + metadata
-   `StateTracker` - Incremental processing state

### Key Constants

-   `DEFAULT_MAX_CONTEXTS = 24`
-   `DEFAULT_CONTEXTS_PER_RUN = 8`
-   `DEFAULT_MIN_SCORE = 1.2`
-   `DEFAULT_SIMILARITY_THRESHOLD = 0.82`

## Questions to Ask When Working on This Project

1.  **Does this change affect card quality?** → Test with representative
    conversations
2.  **Does this change prompt construction?** → Use `--dry-run` to
    validate
3.  **Does this change state format?** → Consider migration path
4.  **Does this add dependencies?** → Update `pyproject.toml`
5.  **Does this increase cost?** → Estimate token/API call impact
6.  **Is there a future direction for this?** → Check
    `FUTURE_DIRECTIONS.md`
7.  **Does this need user documentation?** → Update README_AUTO_ANKI.md

## Resources

-   **Main documentation**: `README_AUTO_ANKI.md`
-   **Quick reference**: `QUICK_START.md`
-   **Setup guide**: `INSTALL.md`
-   **Future roadmap**: `FUTURE_DIRECTIONS.md` (comprehensive!)
-   **This guide**: `CLAUDE.md`

------------------------------------------------------------------------

**Last updated**: 2025-12-05

**Project status**: Production-ready MVP with extensive roadmap for
enhancements

**Current version**: Single-file Python script (\~1200 lines)

**Future vision**: Intelligent, adaptive learning companion with plugin
architecture, semantic deduplication, and active learning capabilities.
