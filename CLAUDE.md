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
       ├─ Load existing cards from Anki via AnkiConnect → Card objects
       │  └─ Requires Anki running with AnkiConnect plugin
       ├─ Scan ChatGPT export folder (markdown) → Conversation objects
       │  └─ Each Conversation contains multiple ChatTurn objects
       ├─ Split long conversations at topic boundaries
       └─ Apply filters: DateRangeFilter, StateTracker

    2. SCORING PHASE (optional, requires --use-filter-heuristics)
       ├─ Aggregate scoring across conversation turns
       │  ├─ Question signals (what/why/how)
       │  ├─ Definition signals (is defined as, stands for)
       │  ├─ Structure signals (bullet points, code blocks)
       │  └─ Quality signals (length, complexity)
       ├─ Extract key topics from all turns
       └─ Threshold filtering (min_score on aggregate)
       NOTE: By default, heuristics are OFF - all conversations go to Stage 1 LLM

    3. DEDUPLICATION PHASE
       ├─ Conversation-level deduplication
       │  ├─ Check if ALL turns are duplicates → skip entire conversation
       │  ├─ Annotate which turns are "already covered"
       │  └─ LLM can skip covered turns intelligently
       ├─ String + semantic similarity (hybrid mode)
       └─ Preserve conversations with any novel content

    4. BATCHING PHASE
       ├─ Group conversations into batches (contexts_per_run)
       ├─ Build prompts with:
       │  ├─ System instructions (Anki best practices)
       │  ├─ Existing cards summary (for dedup awareness)
       │  └─ Full conversations with all turns
       └─ Cap total conversations (max_contexts)

    5. TWO-STAGE CODEX PHASE
       ├─ STAGE 1: LLM Filter (fast, GPT-5.1 Low)
       │  ├─ Receives FULL conversations (not truncated)
       │  ├─ LLM judges quality directly (no heuristic scores)
       │  └─ Decides: keep or skip each conversation
       ├─ STAGE 2: Card Generation (parallel, 3 workers)
       │  ├─ Call `codex exec` with JSON contract
       │  ├─ LLM sees full learning journey
       │  ├─ LLM decides: create cards / skip turns / note insights
       │  └─ Cards linked to specific turns via (conversation_id, turn_index)
       └─ Batches processed in parallel for ~3x speedup

    6. OUTPUT PHASE
       ├─ Save artifacts to run directory
       │  └─ selected_conversations.json (full conversation context)
       ├─ Generate markdown preview
       ├─ Generate JSON for automation
       └─ Update state file (seen_conversations)

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

#### `Conversation` (dataclass)

Represents a full ChatGPT conversation containing multiple turns.

``` python
@dataclass
class Conversation:
    conversation_id: str           # SHA256(source_path + first_timestamp)
    source_path: str
    source_title: Optional[str]
    source_url: Optional[str]
    turns: List[ChatTurn]          # Ordered list of turns
    total_char_count: int          # Sum of all assistant responses
    aggregate_score: float         # Combined score across turns
    aggregate_signals: Dict        # Merged signals + duplicate_turns info
    key_topics: List[str]          # Extracted from all turns
```

#### `ChatTurn` (dataclass)

Represents a single user-assistant exchange within a conversation.

``` python
@dataclass
class ChatTurn:
    context_id: str              # Unique hash for this turn
    turn_index: int              # Position in conversation (0-indexed)
    conversation_id: str         # Link to parent Conversation
    source_path: str             # Originating file
    source_title: Optional[str]
    source_url: Optional[str]
    user_prompt: str             # User's question/request
    assistant_answer: str        # Assistant's response
    score: float                 # Heuristic quality score
    signals: Dict[str, Any]      # Why it scored this way
    key_terms: List[str]         # Extracted key terms
```

### Core Classes

#### `DateRangeFilter`

Filters conversation files by date.

-   Supports formats: `2025-10` (month), `2025-10-01:2025-10-31` (range)
-   Extracts dates from filenames like `2025-10-15_topic.md`

#### `StateTracker`

Manages `.auto_anki_agent_state.json` for incremental processing.

-   Tracks processed files
-   Stores seen context IDs (legacy, preserved for compatibility)
-   Stores seen conversation IDs (v2 schema)
-   Records run history
-   Enables `--unprocessed-only` mode
-   Auto-migrates from v1 to v2 schema on first run

### Key Functions

#### `load_cards_from_anki(deck_names, ...) -> List[Card]`

Loads existing cards from Anki via AnkiConnect.

-   Requires Anki running with AnkiConnect plugin (code: 2055492159)
-   Fetches cards from specified deck names
-   Caches results with configurable TTL (default: 5 minutes)
-   Returns list of existing cards for deduplication

#### `harvest_conversations(...) -> List[Conversation]`

Walks chat export directory and extracts full conversations.

-   Parses markdown conversation files into Conversation objects
-   Each Conversation contains multiple ChatTurn objects
-   Splits long conversations at topic boundaries (configurable limits)
-   Applies date range filtering
-   Applies state-based filtering (unprocessed-only)
-   Computes aggregate scores across all turns
-   Extracts key topics from conversation content

#### `detect_signals(user_text, assistant_text) -> Tuple[float, Dict]`

Heuristic scoring function that identifies quality signals.

**NOTE:** Only used when `--use-filter-heuristics` flag is enabled.
By default, heuristics are OFF and all conversations go directly to Stage 1 LLM filter.

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

#### `prune_conversations(conversations, cards, args) -> List[Conversation]`

Filters and annotates conversations based on similarity to existing cards.

-   Conversation-level deduplication (not per-turn)
-   Only removes conversations where ALL turns are duplicates
-   Annotates `duplicate_turns` and `non_duplicate_turns` in conversation signals
-   LLM can use annotations to skip already-covered content
-   String-based dedup: `SequenceMatcher.ratio()` over normalized text
-   Semantic dedup: optional SentenceTransformers embeddings via
    `--dedup-method semantic` / `hybrid`
-   String threshold: `--similarity-threshold` (default: 0.82)
-   Semantic threshold: `--semantic-similarity-threshold` (default: 0.85)

#### `build_conversation_prompt(cards, conversations, args) -> str`

Constructs the conversation-aware prompt sent to `codex exec`.

**Prompt structure:**

1.  System instructions (Anki best practices + conversation analysis)
2.  JSON contract specification (includes `conversation_id`, `turn_index`, `depends_on`)
3.  Existing cards summary (compact)
4.  Full conversations with all turns (enables learning journey analysis)
5.  Output format requirements

**LLM capabilities enabled by conversation context:**

-   See follow-up questions that indicate user confusion
-   Skip early turns if later corrected
-   Create coherent card sets that build on each other
-   Use `depends_on` field for card ordering

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
    ├── auto_anki_agent.py           # Legacy CLI script (now mostly orchestration)
    ├── auto_anki/                   # Core Python package
    │   ├── __init__.py              # Package entry
    │   ├── cli.py                   # `auto-anki` console script entrypoint
    │   ├── cards.py                 # Card dataclass, AnkiConnect card loading
    │   ├── contexts.py              # Chat harvesting, scoring, ChatTurn
    │   ├── dedup.py                 # String + semantic dedup (FAISS/embeddings)
    │   ├── codex.py                 # Prompt builders, two-stage pipeline, parsing
    │   └── state.py                 # StateTracker, run directory helpers
    ├── tests/                       # Test suite (pytest)
    │   ├── conftest.py              # Shared fixtures
    │   ├── test_scoring.py          # Tests for detect_signals, extract_key_terms
    │   ├── test_normalization.py    # Tests for normalize_text, quick_similarity
    │   ├── test_parsing.py          # Tests for parse_chat_entries, extract_turns
    │   ├── test_date_filter.py      # Tests for DateRangeFilter
    │   └── test_dedup.py            # Tests for is_duplicate_context
    ├── anki_review_ui.py            # Interactive review UI (Shiny app, 1050+ lines)
    ├── anki_connect.py              # AnkiConnect HTTP client (437 lines)
    ├── launch_ui.sh                 # Launch script for review UI
    ├── CLAUDE.md                    # This file
    ├── docs/                        # User & technical documentation
    │   ├── README_AUTO_ANKI.md      # User documentation
    │   ├── UI_README.md             # Interactive UI documentation
    │   ├── ANKICONNECT_GUIDE.md     # AnkiConnect setup and workflows
    │   ├── UI_ENHANCEMENTS_SUMMARY.md
    │   ├── INTEGRATION_COMPLETE.md
    │   ├── QUICK_START.md
    │   ├── INSTALL.md
    │   ├── START_HERE.md
    │   ├── PROJECT_STATUS.md
    │   └── FUTURE_DIRECTIONS.md     # Roadmap (comprehensive!)
    ├── pyproject.toml               # uv project config (defines `auto-anki` script)
    ├── uv.lock                      # Dependency lock
    ├── .auto_anki_agent_state.json  # Runtime state (git-ignored)
    └── auto_anki_runs/              # Output directory
        ├── proposed_cards_YYYY-MM-DD.md
        └── run-YYYYMMDD-HHMMSS/
            ├── selected_conversations.json        # Full conversation context (v2)
            ├── selected_contexts.json             # Legacy per-turn format (v1)
            ├── codex_response_chunk_01.json
            ├── codex_parsed_response_chunk_01.json
            ├── all_proposed_cards.json
            ├── accepted_cards_TIMESTAMP.json      # User-reviewed cards
            └── feedback_data_TIMESTAMP.json       # Review feedback

## Common Tasks

### Understanding the Current System

**What to read first:**

1.  This file (CLAUDE.md) - architecture overview
2.  `auto_anki/cards.py` - `Card`, AnkiConnect card loading
3.  `auto_anki/contexts.py` - `Conversation`, `ChatTurn`, `harvest_conversations()`
4.  `auto_anki/dedup.py` - `SemanticCardIndex`, `prune_conversations()`
5.  `auto_anki/codex.py` - `build_conversation_prompt()`, `run_codex_pipeline()`
6.  `auto_anki/state.py` - `StateTracker`, state migration, `ensure_run_dir()`

**Key configuration points (via CLI defaults):**

-   `DEFAULT_MAX_CONTEXTS = 24` - conversations per run
-   `DEFAULT_CONTEXTS_PER_RUN = 8` - batch size
-   `DEFAULT_MIN_SCORE = 1.2` - quality threshold (only with `--use-filter-heuristics`)
-   `DEFAULT_SIMILARITY_THRESHOLD = 0.82` - string dedup threshold
-   `DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.85` - semantic dedup threshold
-   `--conversation-max-turns = 10` - split conversations longer than this
-   `--conversation-max-chars = 8000` - split conversations larger than this
-   `--use-filter-heuristics` - OFF by default (LLM-only filtering)
-   Stage 2 parallel workers: 3 concurrent (hardcoded)

### Debugging a Run

**Check these in order:**

1.  State file: `.auto_anki_agent_state.json` - what's processed?
    -   Check `state_version` (should be 2)
    -   Check `seen_conversations` list
2.  Run directory: `auto_anki_runs/run-YYYYMMDD-HHMMSS/`
    -   `selected_conversations.json` - full conversations with turns
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

**NOTE:** Heuristics are OFF by default. Only relevant when using `--use-filter-heuristics`.

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

3.  Test with `--use-filter-heuristics --dry-run` to see impact on scoring

### Modifying the Codex Prompt

**Location:** `build_conversation_prompt()` function (conversation-level pipeline)

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

### Running Tests

The project has a pytest-based test suite covering core pure functions.

**Run all tests:**

```bash
uv run pytest
```

**Run with coverage report:**

```bash
uv run pytest --cov=auto_anki --cov-report=term-missing
```

**Run specific test file:**

```bash
uv run pytest tests/test_scoring.py -v
```

**Test coverage by module:**

| Module | Coverage | What's Tested |
|--------|----------|---------------|
| `contexts.py` | ~51% | `detect_signals`, `extract_key_terms`, `parse_chat_entries`, `extract_turns`, `DateRangeFilter` |
| `cards.py` | ~31% | `normalize_text` |
| `dedup.py` | ~23% | `is_duplicate_context`, `quick_similarity` |

**What's NOT tested (requires mocking):**

- `codex.py` - LLM integration
- `state.py` - File I/O operations
- `load_cards_from_anki` - AnkiConnect dependency

**Adding new tests:**

1. Create test file in `tests/` following `test_*.py` naming
2. Import functions from `auto_anki.*` modules
3. Use fixtures from `conftest.py` for sample data
4. Run `uv run pytest tests/your_test.py -v` to verify

## Important Patterns & Conventions

### Error Handling Philosophy

**The script is defensive but not silent:**

-   AnkiConnect: Clear error if Anki not running
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

### Input: Anki Cards via AnkiConnect

**Requirements:**

-   Anki must be running
-   AnkiConnect plugin installed (code: 2055492159)
-   Deck names specified in config or via `--decks` flag

**Fetch logic:**

-   Query note IDs via `findNotes(deck:"DeckName")`
-   Fetch note details via `notesInfo()`
-   Extract Front/Back fields
-   Cache with configurable TTL (default: 5 minutes)

### Output: Codex Exec

**Interface:**

-   Subprocess call to `codex exec`
-   Stdin: Prompt text
-   Stdout: JSON response
-   Args: `--model`, `--reasoning-effort`, custom args

**Response schema (conversation-aware):**

``` json
{
  "cards": [
    {
      "conversation_id": "abc123",
      "turn_index": 2,
      "context_id": "hash123",
      "deck": "Research_Learning",
      "front": "What is...",
      "back": "...",
      "card_style": "basic",
      "confidence": 0.85,
      "tags": ["ml", "concepts"],
      "depends_on": ["hash122"],
      "notes": "Rationale for creation"
    }
  ],
  "skipped_conversations": [
    {
      "conversation_id": "def456",
      "reason": "All turns already covered by existing cards"
    }
  ],
  "skipped_turns": [
    {
      "conversation_id": "abc123",
      "turn_index": 0,
      "reason": "Corrected in later turn"
    }
  ],
  "learning_insights": [
    "User struggled with X concept across multiple turns"
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

1.  ~~**Semantic deduplication**~~ ✅ **DONE!**
    - Embeddings + FAISS vector cache for better duplicate detection
    - Hybrid string + embedding-based dedup (see `SemanticCardIndex`)
    - CLI: `--dedup-method semantic|hybrid`, `--semantic-*` flags
2.  ~~**Interactive review mode**~~ ✅ **DONE!**
3.  ~~**Two-stage LLM pipeline**~~ ✅ **DONE!**
    - Initial implementation available via `--two-stage`
    - Stage 1: filter contexts (`gpt-5.1` with `model_reasoning_effort=low` by default)
    - Stage 2: generate cards (`gpt-5.1` with `model_reasoning_effort=high` by default)
4.  ~~**AnkiConnect integration**~~ ✅ **DONE!**
5.  ~~**Conversation-level processing**~~ ✅ **DONE!**
    - Full conversations sent to LLM instead of individual turns
    - LLM sees learning journey, follow-ups, corrections
    - Topic-boundary splitting for long conversations
    - State schema v2 with automatic migration
6.  **Active learning** - Feedback loop to improve quality over time

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
-   Prompt construction: `build_conversation_prompt()`
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

-   `load_cards_from_anki()` - Load cards via AnkiConnect
-   `harvest_conversations()` - Extract full conversations
-   `split_conversation_by_topic()` - Split long conversations
-   `detect_signals()` - Score individual turns
-   `detect_conversation_signals()` - Aggregate scoring
-   `prune_conversations()` - Filter/annotate duplicates
-   `build_conversation_prompt()` - LLM interface
-   `run_codex_exec()` - LLM invocation

### Key Data Structures

-   `Card` - Flashcard (front/back/deck/tags)
-   `Conversation` - Full conversation with turns + metadata
-   `ChatTurn` - Single exchange within conversation
-   `StateTracker` - Incremental processing state (v2 schema)

### Key Constants

-   `DEFAULT_MAX_CONTEXTS = 24` - conversations per run
-   `DEFAULT_CONTEXTS_PER_RUN = 8` - batch size
-   `DEFAULT_MIN_SCORE = 1.2` - aggregate score threshold (only with `--use-filter-heuristics`)
-   `DEFAULT_SIMILARITY_THRESHOLD = 0.82`
-   `--conversation-max-turns = 10`
-   `--conversation-max-chars = 8000`
-   `--use-filter-heuristics` - OFF by default
-   Stage 2 parallel workers: 3

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

**Last updated**: 2025-12-07

**Project status**: Production-ready with conversation-level processing,
semantic deduplication, two-stage pipeline, parallel execution, and interactive review UI.

**Current version**: Modular Python package with CLI entrypoint

**Recent additions**:
-   Heuristics OFF by default - LLM-only filtering via Stage 1
-   Stage 1 receives full conversations (not truncated)
-   Stage 2 runs in parallel (3 concurrent workers, ~3x speedup)
-   `--use-filter-heuristics` flag to enable optional heuristic pre-filtering

**Future vision**: Intelligent, adaptive learning companion with plugin
architecture and active learning capabilities.
