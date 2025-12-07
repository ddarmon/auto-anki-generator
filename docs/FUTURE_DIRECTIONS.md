# Future Directions for Auto Anki Agent

This document captures potential innovations, improvements, and
architectural enhancements for the auto_anki_agent.py system. Ideas are
organized by impact area and include implementation considerations.

## Executive Summary

The system is now production-ready with major features implemented:

### ✅ Completed
-   **Semantic deduplication**: FAISS + SentenceTransformers with caching
-   **Interactive review UI**: Shiny web app with keyboard shortcuts
-   **AnkiConnect integration**: Direct import to Anki
-   **Two-stage LLM pipeline**: Fast filter + slow generation
-   **Conversation-level processing**: LLM sees full learning journey

### 🚧 Remaining Opportunities
-   **Performance**: Stage 2 now parallelized; further optimization possible
-   **Active learning**: No feedback loop to improve quality over time
-   **Plugin architecture**: Not yet extensible for custom scorers/parsers

## Performance & Scalability

### 1. Parallel Processing Architecture ✅ PARTIAL

**Status**: Stage 2 (card generation) now runs in parallel with 3 concurrent workers.

**What's implemented**:
-   `ThreadPoolExecutor` with 3 workers for Stage 2 batches
-   Stage 1 (filter) remains sequential (typically fast enough)
-   Concurrent codex exec calls for card generation

**Remaining opportunities**:
-   Use `multiprocessing.Pool` for chat transcript parsing
-   Parallelize deduplication checks across card database
-   Add progress tracking with `tqdm` for parallel operations

**Impact**: Achieved ~3x speedup on Stage 2; further optimization possible for harvesting.

**Implementation example** (for future harvesting parallelization):

``` python
from multiprocessing import Pool
from functools import partial

def parse_file_wrapper(path, args):
    # Worker function for parallel parsing
    return harvest_single_file(path, args)

with Pool(processes=8) as pool:
    contexts = pool.map(
        partial(parse_file_wrapper, args=args),
        chat_files
    )
```

### 2. Semantic Deduplication with Embeddings ✅ DONE

**Status**: Fully implemented with FAISS IndexFlatIP and persistent caching.

**What's implemented**:
-   `SemanticCardIndex` class in `auto_anki/dedup.py`
-   FAISS vector index for fast similarity search
-   Persistent cache in `.deck_cache/embeddings/` (auto-invalidated when decks change)
-   Hybrid mode combining string + semantic deduplication
-   CLI: `--dedup-method semantic|hybrid`, `--semantic-similarity-threshold`

**Potential enhancements**:
-   Use embeddings for novelty scoring (underrepresented topics)

**Impact**: Better duplicate detection, especially for paraphrased
content; quality improvement scales with deck size.

**Cost consideration**: \~\$0.0001 per card for OpenAI embeddings, or
use local models (sentence-transformers)

**Architecture**:

``` python
from sentence_transformers import SentenceTransformer
import faiss

class SemanticCardIndex:
    def __init__(self, cards: List[Card]):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cards = cards
        self.embeddings = self.model.encode([
            f"{c.front} {c.back}" for c in cards
        ])
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def find_duplicates(self, context: ChatTurn, threshold=0.85):
        query_emb = self.model.encode([
            f"{context.user_prompt} {context.assistant_answer}"
        ])
        distances, indices = self.index.search(query_emb, k=5)
        return [self.cards[i] for i, d in zip(indices[0], distances[0])
                if d > threshold]
```

### 3. Incremental State Management

**Current limitation**: Full JSON state file reload on every run

**Proposal**:

-   Migrate to SQLite database for state tracking
-   Index on file paths, context IDs, timestamps
-   Track file modification times to skip unchanged files
-   Support partial state queries (don't load everything)

**Impact**: Faster startup, better scalability to 10k+ contexts

**Schema**:

``` sql
CREATE TABLE processed_files (
    path TEXT PRIMARY KEY,
    mtime REAL,
    processed_at TEXT,
    cards_generated INTEGER
);

CREATE TABLE seen_contexts (
    context_id TEXT PRIMARY KEY,
    file_path TEXT,
    processed_at TEXT,
    resulted_in_card BOOLEAN,
    FOREIGN KEY (file_path) REFERENCES processed_files(path)
);

CREATE INDEX idx_contexts_file ON seen_contexts(file_path);
```

## Intelligence & Card Quality

### 4. Two-Stage LLM Pipeline ✅ DONE

**Status**: Fully implemented and enabled by default.

**What's implemented**:
-   **Stage 1**: Fast filter using `gpt-5.1` with `model_reasoning_effort=low`
    -   Receives **full conversations** (complete user prompts + assistant responses)
    -   LLM judges quality directly (no heuristic scores in prompt)
    -   Heuristic pre-filtering available via `--use-filter-heuristics` (off by default)
    -   CLI: `--codex-model-stage1`
-   **Stage 2**: Card generation using `gpt-5.1` with `model_reasoning_effort=high`
    -   **Parallel execution** with 3 concurrent workers via `ThreadPoolExecutor`
    -   Full card generation with quality prompts
    -   CLI: `--codex-model-stage2`
-   CLI flags: `--two-stage` (default), `--single-stage` to disable

**Impact**: Significant cost reduction by filtering before expensive generation; ~3x faster Stage 2 via parallelization

### 5. Context Clustering & Topic Modeling (Partially Addressed)

**Status**: Conversation-level processing provides natural topic grouping.

**What's implemented**:
-   Full conversations sent to LLM (related turns grouped together)
-   `key_topics` extracted from each conversation
-   Topic-boundary splitting for long conversations
-   LLM can create coherent card sets with `depends_on` links

**Remaining opportunities**:
-   Cross-conversation topic clustering using embeddings
-   BERTopic for automatic topic discovery
-   Topic distribution visualization
-   Coverage gap detection by topic

**Visualization idea**: Topic distribution heatmap showing deck coverage
vs incoming contexts

### 6. Active Learning from User Feedback

**Current limitation**: No feedback loop to improve quality over time

**Proposal**:

-   Track which generated cards get imported vs rejected
-   Log rejection reasons (duplicate, low quality, not relevant, etc.)
-   Use feedback to:
    -   Adjust heuristic weights (if definition_like cards always
        rejected, lower that weight)
    -   Fine-tune pre-filter model (few-shot examples of good/bad)
    -   Build user-specific quality predictor

**Implementation**:

``` python
class FeedbackTracker:
    def record_card_decision(self, card_id, action, reason=None):
        # action: 'imported', 'rejected', 'edited'
        self.db.execute("""
            INSERT INTO card_feedback
            (card_id, action, reason, timestamp)
            VALUES (?, ?, ?, ?)
        """, (card_id, action, reason, datetime.now()))

    def get_acceptance_rate_by_signal(self):
        # Calculate which signals correlate with accepted cards
        return self.db.execute("""
            SELECT signal_name,
                   AVG(CASE WHEN action = 'imported' THEN 1 ELSE 0 END) as acceptance_rate
            FROM card_feedback f
            JOIN card_signals s ON f.card_id = s.card_id
            GROUP BY signal_name
        """).fetchall()
```

### 7. Smart Dependency Ordering

**Current limitation**: Cards generated in arbitrary order, missing
learning prerequisites

**Proposal**:

-   Detect prerequisite relationships:
    -   "X is a type of Y" → Y should be learned before X
    -   "Before understanding X, you need to know Y"
    -   Topic hierarchy (machine learning → neural networks →
        transformers)
-   Generate dependency graph
-   Order card generation/presentation by topological sort
-   Add "prerequisite" tags to cards

**Use cases**:

-   New learners get foundation-first card sequence
-   Avoid "forward references" in card text
-   Better integration with spaced repetition scheduling

### 8. Advanced Heuristics & Signals

**New signals to detect**:

``` python
def detect_enhanced_signals(user_text, assistant_text):
    signals = {}

    # Negative signals (lower quality)
    signals['hedging'] = any(phrase in assistant_text.lower() for phrase in [
        "i'm not sure", "i don't know", "it's complicated",
        "it depends", "maybe", "possibly"
    ])

    # Positive signals
    signals['has_examples'] = bool(re.search(
        r'for example|e\.g\.|such as|consider',
        assistant_text, re.I
    ))

    signals['has_comparison'] = bool(re.search(
        r'compared to|versus|vs\.|unlike|similar to',
        assistant_text, re.I
    ))

    signals['has_analogy'] = bool(re.search(
        r'like a|think of.*as|analogous to|metaphorically',
        assistant_text, re.I
    ))

    # Detect procedural content (worse for SRS)
    signals['too_procedural'] = (
        assistant_text.count('```') > 2 or
        len(re.findall(r'step \d+', assistant_text.lower())) > 5
    )

    # Named entity recognition for technical concepts
    # (requires spaCy or similar)
    signals['technical_term_count'] = count_technical_entities(assistant_text)

    return signals
```

### 9. Novelty Detection & Topic Diversity

**Goal**: Prioritize contexts covering underrepresented topics

**Approach**:

1.  Extract topic distribution from existing decks
    -   Use LDA or BERTopic on all card fronts
    -   Calculate topic frequencies
2.  Score new contexts by novelty
    -   Contexts on rare topics get boosted score
    -   Contexts on over-represented topics get penalized
3.  Maintain diversity in card generation
    -   Ensure each run covers multiple topics
    -   Avoid creating 50 cards about the same narrow subject

**Metric**: Topic entropy (higher = more diverse deck)

### 10. Quality Validation & Auto-Fix

**Post-processing pipeline for generated cards**:

``` python
class CardValidator:
    def validate_card(self, card: Dict) -> Tuple[bool, List[str]]:
        issues = []

        # Check for ambiguous questions
        if not card['front'].endswith('?') and card['card_style'] == 'basic':
            issues.append("Question doesn't end with ?")

        # Check length (minimum information principle)
        if len(card['back']) > 500:
            issues.append("Answer too long (>500 chars)")

        # Check for missing context
        ambiguous_terms = ['it', 'this', 'that', 'he', 'she']
        if any(card['front'].lower().startswith(term + ' ')
               for term in ambiguous_terms):
            issues.append("Question starts with ambiguous pronoun")

        # Check for empty cloze deletions
        if card['card_style'] == 'cloze' and '[...]' not in card['front']:
            issues.append("Cloze card missing [...] deletion")

        return len(issues) == 0, issues

    def auto_fix(self, card: Dict, issues: List[str]) -> Dict:
        # Attempt simple fixes
        if "Question doesn't end with ?" in issues:
            card['front'] = card['front'].rstrip('.') + '?'

        # For complex issues, flag for manual review
        if issues:
            card['needs_review'] = True
            card['validation_issues'] = issues

        return card
```

## User Experience

### 11. Interactive Review Mode ✅ DONE

**Status**: Fully implemented as Shiny web app (`anki_review_ui.py`).

**What's implemented**:
-   Card-by-card review with accept/reject/edit/skip
-   Keyboard shortcuts (A/R/E/S, arrow keys)
-   Source context display (now with conversation info)
-   Progress tracking and statistics
-   Filtering by deck and confidence
-   Bulk operations (accept all high-confidence)
-   Rejection reason tracking
-   Codex-powered card updates
-   Export accepted cards to JSON

**Launch**: `./launch_ui.sh` or `shiny run anki_review_ui.py`

See `UI_README.md` for full documentation.

### 12. Rich Previews & Context

**Show why each card was generated**:

``` markdown
╭─ Card 3/24 ──────────────────────────────────────────╮
│ What is the primary optimization objective in        │
│ gradient boosting?                                    │
│                                                       │
│ [Back]                                                │
│ Minimize a differentiable loss function using        │
│ gradient descent in function space.                  │
│                                                       │
│ Confidence: 0.87 | Deck: Machine Learning            │
├───────────────────────────────────────────────────────┤
│ SOURCE CONTEXT                                        │
│ File: 2025-11-15_ml-concepts.md                      │
│ Score: 2.4 (question_like + definition_like + ...)   │
│                                                       │
│ User: "Explain how gradient boosting works"          │
│ Assistant: "Gradient boosting builds an ensemble..." │
│                                                       │
│ Similar existing cards (0 found)                      │
├───────────────────────────────────────────────────────┤
│ [a]ccept [r]eject [e]dit [s]kip [v]iew full context  │
╰───────────────────────────────────────────────────────╯
```

### 13. Progress Dashboard & Observability

**Real-time progress tracking**:

``` python
from tqdm import tqdm
import time

class ProgressTracker:
    def __init__(self):
        self.pbar = tqdm(total=100, desc="Processing contexts")
        self.stats = {
            'files_scanned': 0,
            'contexts_found': 0,
            'contexts_filtered': 0,
            'cards_generated': 0,
            'current_file': None
        }

    def update_display(self):
        self.pbar.set_postfix({
            'files': self.stats['files_scanned'],
            'contexts': self.stats['contexts_found'],
            'cards': self.stats['cards_generated'],
            'current': self.stats['current_file'][:30] if self.stats['current_file'] else ''
        })
```

**Web dashboard option** (Flask/FastAPI):

-   Real-time progress via WebSocket
-   Historical run statistics
-   Cost tracking and projections
-   Topic distribution visualization

### 14. Smart Notifications

**Integration with notification systems**:

``` python
def notify_completion(stats: Dict):
    # macOS notification
    os.system(f"""
        osascript -e 'display notification
        "{stats['cards_generated']} new cards ready for review"
        with title "Auto Anki Agent"'
    """)

    # Slack webhook (optional)
    if SLACK_WEBHOOK_URL:
        requests.post(SLACK_WEBHOOK_URL, json={
            'text': f"🎓 Generated {stats['cards_generated']} cards from {stats['contexts_processed']} contexts"
        })

# Daily digest email
def send_daily_digest():
    # Aggregate stats from runs in last 24h
    # Email summary with top cards and topic breakdown
    pass
```

## Integration & Automation

### 15. Direct Anki Integration via AnkiConnect ✅ DONE

**Status**: Fully implemented in `anki_connect.py` with UI integration.

**What's implemented**:
-   `AnkiConnectClient` class with full API support
-   Real-time connection status indicator in UI
-   Import current card with one click
-   Batch import all accepted cards
-   Duplicate detection (configurable)
-   Auto-create missing decks
-   30-60x faster than manual import

**Setup**: Install AnkiConnect plugin (code: 2055492159), start Anki, launch UI.

See `ANKICONNECT_GUIDE.md` for full documentation.

### 16. Multi-Source Support

**Extend beyond ChatGPT exports**:

**PDF highlights & annotations**:

``` python
import fitz  # PyMuPDF

def extract_pdf_highlights(pdf_path: Path) -> List[ChatTurn]:
    doc = fitz.open(pdf_path)
    contexts = []

    for page in doc:
        annotations = page.annots()
        for annot in annotations:
            if annot.type[0] == 8:  # Highlight
                text = page.get_textbox(annot.rect)
                # Treat highlight as user prompt
                # Surrounding text as context
                contexts.append(create_context_from_highlight(text, page))

    return contexts
```

**YouTube transcripts**:

``` python
from youtube_transcript_api import YouTubeTranscriptApi

def extract_youtube_concepts(video_id: str) -> List[ChatTurn]:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    # Chunk transcript into semantic segments
    # Extract Q&A-like exchanges
    # Score based on educational value
    pass
```

**Other sources**:

-   Apple Notes (via SQLite database at
    `~/Library/Group Containers/group.com.apple.notes/`)
-   Obsidian vaults (markdown with YAML frontmatter)
-   Notion exports (via API)
-   Email threads (IMAP parsing)
-   Slack conversations (via API)

### 17. Continuous Monitoring & Automation

**Watch mode for automatic processing**:

``` python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChatFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith('.md'):
            # Wait for file to be fully written
            time.sleep(2)

            # Process single file
            contexts = harvest_single_file(Path(event.src_path), args)
            if contexts:
                process_contexts_async(contexts)

observer = Observer()
observer.schedule(ChatFileHandler(), chat_root, recursive=True)
observer.start()
```

**Scheduled runs**:

``` bash
# Cron job for daily processing
0 2 * * * cd /path/to/aianki && python auto_anki_agent.py --unprocessed-only --quiet
```

**Smart rate limiting**:

-   Accumulate contexts throughout day
-   Process in batches to optimize API costs
-   Respect token budgets (max \$X per day)

## Prompt Engineering & Customization

### 18. Configurable Prompt Templates

**Current limitation**: Hardcoded prompt in `build_codex_prompt()`

**Proposal**: YAML-based prompt configuration

``` yaml
# prompts/default.yaml
version: 1.0
system_instructions: |
  You are operating as the decision layer of an autonomous
  spaced-repetition agent...

card_quality_rules:

  - minimum_information_principle
  - atomic_questions
  - clear_context_cues
  - optimal_wording

scoring_weights:
  question_like: 1.0
  definition_like: 0.45
  has_examples: 0.6
  hedging: -0.3

output_format:
  contract_version: 2.0
  required_fields: [context_id, deck, front, back, confidence]
  optional_fields: [tags, notes, card_style]

examples:

  - good:
      front: "What is the primary function of mitochondria?"
      back: "**Produce ATP** through cellular respiration"
    bad:
      front: "Tell me about mitochondria"
      back: "Mitochondria are organelles that produce energy..."
```

**Loading prompts**:

``` python
import yaml

class PromptTemplate:
    def __init__(self, template_path: Path):
        self.config = yaml.safe_load(template_path.read_text())

    def render(self, cards: List[Card], contexts: List[ChatTurn]) -> str:
        # Inject config values into prompt
        return self.config['system_instructions'] + \
               self._build_payload(cards, contexts)

    def get_scoring_weight(self, signal: str) -> float:
        return self.config['scoring_weights'].get(signal, 0.0)
```

### 19. Domain-Specific Agents

**Specialized prompts for different knowledge domains**:

**Math/Science**:

``` yaml
# prompts/math.yaml
domain: mathematics
system_instructions: |
  Focus on mathematical concepts, proofs, and formulas.
  ALWAYS use LaTeX for mathematical notation.
  Prefer theorem/proof style cards over procedural steps.

card_styles:

  - theorem_statement
  - proof_sketch
  - formula_derivation
  - worked_example

latex_guidelines: |

  - Inline: $x^2 + y^2 = z^2$
  - Display: $$\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$
  - Always define variables

scoring_boosts:
  has_latex: 0.8
  has_proof: 0.6
  references_theorem: 0.5
```

**Programming/CS**:

``` yaml
# prompts/programming.yaml
domain: computer_science
system_instructions: |
  Focus on concepts, algorithms, and design patterns.
  Avoid syntax-heavy cards - prioritize understanding over memorization.
  Include time/space complexity for algorithms.

card_styles:

  - concept_explanation
  - algorithm_analysis
  - pattern_recognition
  - tradeoff_comparison

anti_patterns:

  - code_syntax_without_concept
  - language_specific_details
  - procedural_how_to

scoring_boosts:
  mentions_complexity: 0.7
  compares_approaches: 0.6
  explains_tradeoff: 0.8
```

**Language Learning**:

``` yaml
# prompts/language.yaml
domain: language_learning
system_instructions: |
  Create vocabulary, grammar, and usage cards.
  Include pronunciation hints and example sentences.
  Tag by language, difficulty level, and category.

card_styles:

  - vocabulary (word → translation)
  - usage (sentence → correct form)
  - grammar_rule (concept → explanation)

required_fields:

  - target_language
  - difficulty_level (A1, A2, B1, B2, C1, C2)
  - part_of_speech

scoring_boosts:
  has_example_sentence: 0.8
  has_pronunciation: 0.5
  shows_usage_context: 0.7
```

### 20. Few-Shot Learning from User's Best Cards

**Auto-improve prompts using existing deck quality**:

``` python
def select_exemplar_cards(cards: List[Card], n=5) -> List[Card]:
    """Select best cards from deck to use as few-shot examples."""

    # Heuristics for "good" cards:
    # - Concise (front < 100 chars, back < 200 chars)
    # - Clear structure (question mark, bolded key terms)
    # - Appropriate tags

    scored_cards = []
    for card in cards:
        score = 0

        if len(card.front) < 100 and '?' in card.front:
            score += 1

        if 50 < len(card.back) < 200:
            score += 1

        if '**' in card.back:  # Has bold emphasis
            score += 0.5

        if len(card.tags) >= 2:
            score += 0.5

        scored_cards.append((score, card))

    # Return top N
    scored_cards.sort(reverse=True, key=lambda x: x[0])
    return [card for _, card in scored_cards[:n]]

def build_prompt_with_examples(cards, contexts, args):
    exemplars = select_exemplar_cards(cards, n=5)

    examples_section = "## Examples of High-Quality Cards from Your Deck\n\n"
    for ex in exemplars:
        examples_section += f"""
### Example

- **Front**: {ex.front}
- **Back**: {ex.back}
- **Tags**: {', '.join(ex.tags)}
- **Deck**: {ex.deck}

"""

    return base_instructions + examples_section + payload
```

## Advanced Features

### 21. Multi-Modal Card Generation

**Support for visual learning**:

**Extract images from conversations**:

``` python
def extract_conversation_images(chat_path: Path) -> List[Tuple[str, bytes]]:
    """Extract embedded images from markdown conversations."""
    text = chat_path.read_text()

    # Find markdown image references
    image_refs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text)

    images = []
    for alt_text, path in image_refs:
        image_path = chat_path.parent / path
        if image_path.exists():
            images.append((alt_text, image_path.read_bytes()))

    return images
```

**Generate visual mnemonics**:

``` python
def generate_mnemonic_image(concept: str) -> bytes:
    """Use DALL-E to create visual memory aids."""
    response = openai.Image.create(
        prompt=f"Simple, memorable illustration for learning: {concept}",
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    return requests.get(image_url).content
```

**Audio for pronunciation**:

``` python
def add_pronunciation_audio(card: Dict) -> Dict:
    """Generate TTS audio for language learning cards."""
    if card.get('target_language') and card.get('deck') == 'Vocabulary':
        audio = openai.Audio.create(
            input=card['front'],
            voice='alloy',
            model='tts-1'
        )
        card['audio_file'] = save_audio_for_anki(audio)
    return card
```

### 22. Concept Graph Visualization

**Build knowledge graph from cards**:

``` python
import networkx as nx
from pyvis.network import Network

class ConceptGraph:
    def __init__(self, cards: List[Card]):
        self.graph = nx.DiGraph()
        self._build_graph(cards)

    def _build_graph(self, cards):
        for card in cards:
            # Extract concepts from front and back
            concepts = self._extract_concepts(card.front + ' ' + card.back)

            # Add nodes
            for concept in concepts:
                self.graph.add_node(concept, deck=card.deck)

            # Add edges (concept relationships)
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    self.graph.add_edge(c1, c2, card_id=card.front[:50])

    def find_coverage_gaps(self) -> List[str]:
        """Identify isolated concepts or weak connections."""
        # Nodes with low degree centrality
        centrality = nx.degree_centrality(self.graph)
        gaps = [node for node, score in centrality.items() if score < 0.1]
        return gaps

    def suggest_learning_path(self, start_topic: str) -> List[str]:
        """Generate optimal learning sequence using topological sort."""
        subgraph = nx.ego_graph(self.graph, start_topic, radius=3)
        return list(nx.topological_sort(subgraph))

    def visualize(self, output_path: Path):
        """Create interactive HTML visualization."""
        net = Network(height='750px', width='100%', directed=True)
        net.from_nx(self.graph)
        net.save_graph(str(output_path))
```

**Use cases**:

-   Visual deck coverage analysis
-   Identify knowledge gaps
-   Suggest strategic learning sequences
-   Find redundant or orphaned concepts

### 23. Spaced Repetition Optimization

**Pre-tag cards with difficulty predictions**:

``` python
def predict_card_difficulty(card: Dict, context: ChatTurn) -> float:
    """Estimate how difficult a card will be to remember."""

    difficulty = 5.0  # Base difficulty (1-10 scale)

    # Complexity signals
    if len(card['back']) > 300:
        difficulty += 1.5

    if context.signals.get('code_blocks', 0) > 0:
        difficulty += 1.0

    # Math/formulas are harder
    if '$' in card['back'] or '$$' in card['back']:
        difficulty += 1.2

    # Lists are harder to remember
    if card['card_style'] == 'list':
        difficulty += 2.0

    # Definitional cards are easier
    if context.signals.get('definition_like'):
        difficulty -= 1.0

    # Clamp to 1-10
    return max(1.0, min(10.0, difficulty))

# Add to card metadata
card['predicted_difficulty'] = predict_card_difficulty(card, context)
card['suggested_ease_factor'] = 2.5 - (card['predicted_difficulty'] - 5) * 0.1
```

**Integration with SuperMemo algorithm**:

-   Set initial ease factor based on prediction
-   Adjust review intervals proactively
-   Flag "leeches" before they become problematic

### 24. Collaborative Features

**Share high-quality card templates**:

``` python
class CardTemplateRegistry:
    def __init__(self, registry_url='https://aianki.io/templates'):
        self.registry_url = registry_url

    def publish_template(self, prompt_config: Dict, stats: Dict):
        """Share successful prompt templates with community."""
        requests.post(f'{self.registry_url}/submit', json={
            'prompt_config': prompt_config,
            'stats': {
                'acceptance_rate': stats['accepted'] / stats['total'],
                'avg_confidence': stats['avg_confidence'],
                'domain': prompt_config['domain']
            }
        })

    def search_templates(self, domain: str, min_rating=4.0):
        """Find community-rated templates for your domain."""
        response = requests.get(
            f'{self.registry_url}/search',
            params={'domain': domain, 'min_rating': min_rating}
        )
        return response.json()['templates']
```

**Upvote/downvote for quality scoring**:

``` python
def submit_card_rating(card_id: str, rating: int, feedback: str):
    """Submit feedback on card quality (1-5 stars)."""
    # Aggregate ratings to improve future generation
    # Use as training data for quality predictor
    pass
```

## Code Architecture & Engineering

### 25. Plugin System

**Make the system extensible**:

``` python
# plugins/base.py
from abc import ABC, abstractmethod

class ScorerPlugin(ABC):
    @abstractmethod
    def score(self, user_text: str, assistant_text: str) -> Tuple[float, Dict]:
        """Return (score_delta, signals_dict)."""
        pass

class ParserPlugin(ABC):
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser handles the file."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> List[ChatTurn]:
        """Extract contexts from file."""
        pass

class ExporterPlugin(ABC):
    @abstractmethod
    def export(self, cards: List[Dict], output_path: Path):
        """Export cards to target format."""
        pass

# plugins/scorers/code_quality.py
class CodeQualityScorer(ScorerPlugin):
    def score(self, user_text, assistant_text):
        signals = {}
        score_delta = 0.0

        # Boost for code snippets with explanations
        code_blocks = assistant_text.count('```')
        explanation_words = len(assistant_text.split())

        if code_blocks > 0:
            ratio = explanation_words / max(1, code_blocks * 50)
            if ratio > 2:  # Good explanation-to-code ratio
                score_delta += 0.5
                signals['well_explained_code'] = True

        return score_delta, signals

# Usage
scorer_registry = PluginRegistry()
scorer_registry.register(CodeQualityScorer())
scorer_registry.register(MathNotationScorer())
scorer_registry.register(LanguageLearningScorer())

total_score = base_score
for scorer in scorer_registry.get_scorers():
    delta, signals = scorer.score(user_text, assistant_text)
    total_score += delta
    all_signals.update(signals)
```

### 26. Better Testing

**Comprehensive test suite**:

``` python
# tests/test_scoring.py
import pytest
from auto_anki_agent import detect_signals

def test_question_detection():
    score, signals = detect_signals(
        "What is gradient descent?",
        "Gradient descent is an optimization algorithm..."
    )
    assert signals['question_like'] == True
    assert score >= 1.0

def test_definition_detection():
    score, signals = detect_signals(
        "Define polymorphism",
        "Polymorphism is defined as the ability of..."
    )
    assert signals['definition_like'] == True

def test_hedging_penalty():
    score, signals = detect_signals(
        "How does X work?",
        "I'm not sure, but maybe it could be..."
    )
    assert signals['hedging'] == True
    # Should have lower score than confident answer

# tests/test_deduplication.py
def test_semantic_deduplication():
    card1 = Card(front="What is ML?", back="Machine learning...")
    card2 = Card(front="Define machine learning", back="ML is...")

    assert is_semantic_duplicate(card1, card2, threshold=0.85)

# tests/test_parsing.py
def test_chat_parsing():
    sample_md = """
# Test Conversation

- URL: https://example.com

---

[2025-01-15 10:30] user:
What is Python?

[2025-01-15 10:31] assistant:
Python is a programming language.
"""
    contexts = parse_chat_file(sample_md)
    assert len(contexts) == 1
    assert contexts[0].user_prompt == "What is Python?"

# Property-based testing with hypothesis
from hypothesis import given, strategies as st

@given(st.text(min_size=10, max_size=500))
def test_normalize_text_never_raises(text):
    # Should handle any text without crashing
    result = normalize_text(text)
    assert isinstance(result, str)
```

### 27. Configuration Management

**Separate config from code**:

``` yaml
# config/default.yaml
decks:
  glob_pattern: "*.html"
  cache_enabled: true
  cache_dir: ".deck_cache"

sources:
  chat_root: "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/chatgpt"
  formats:

    - markdown
    - pdf
    - youtube

processing:
  max_contexts: 24
  contexts_per_run: 8
  per_file_limit: 3
  min_score: 1.2

  parallel:
    enabled: true
    workers: 8

  deduplication:
    method: "semantic"  # or "string"
    threshold: 0.82
    use_embeddings: true
    embedding_model: "all-MiniLM-L6-v2"

codex:
  model: "gpt-5"
  reasoning_effort: "medium"
  max_retries: 3
  timeout_seconds: 120

  cost_limits:
    max_per_run: 5.0  # USD
    max_per_day: 20.0

output:
  format: "both"  # json, markdown, both
  directory: "auto_anki_runs"

  interactive_review:
    enabled: true
    auto_accept_threshold: 0.95  # Auto-accept high-confidence cards

state:
  file: ".auto_anki_agent_state.json"
  backend: "sqlite"  # json or sqlite
  database: ".auto_anki_agent.db"

profiles:
  conservative:
    min_score: 2.0
    similarity_threshold: 0.9
    max_contexts: 12

  aggressive:
    min_score: 0.8
    similarity_threshold: 0.75
    max_contexts: 48
```

**Load config with environment overrides**:

``` python
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        self._apply_env_overrides()

    def _load_config(self, config_path):
        if not config_path:
            config_path = Path(__file__).parent / 'config' / 'default.yaml'

        base_config = yaml.safe_load(config_path.read_text())

        # Check for profile
        profile = os.getenv('AIANKI_PROFILE', 'default')
        if profile != 'default' and profile in base_config.get('profiles', {}):
            # Merge profile settings
            profile_config = base_config['profiles'][profile]
            base_config['processing'].update(profile_config)

        return base_config

    def _apply_env_overrides(self):
        # Allow environment variable overrides
        # AIANKI_CODEX_MODEL=gpt-4 → config.codex.model
        for key, value in os.environ.items():
            if key.startswith('AIANKI_'):
                config_path = key[7:].lower().split('_')
                self._set_nested(self.config, config_path, value)

    def get(self, path: str, default=None):
        """Get config value by dot-notation path."""
        parts = path.split('.')
        value = self.config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default
        return value if value is not None else default

# Usage
config = Config()
model = config.get('codex.model')  # "gpt-5"
workers = config.get('processing.parallel.workers')  # 8
```

### 28. Observability & Metrics

**Structured logging**:

``` python
import structlog

logger = structlog.get_logger()

# Rich contextual logging
logger.info(
    "context_processed",
    context_id=context.context_id,
    score=context.score,
    file_path=context.source_path,
    signals=context.signals
)

logger.warning(
    "duplicate_detected",
    context_id=context.context_id,
    similar_card_id=card.front[:50],
    similarity=0.89
)

# Automatically includes timestamps, request IDs, etc.
```

**Metrics tracking**:

``` python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RunMetrics:
    run_id: str
    start_time: datetime
    end_time: datetime

    files_scanned: int
    contexts_extracted: int
    contexts_filtered: int
    contexts_sent_to_codex: int

    cards_generated: int
    cards_accepted: int
    cards_rejected: int

    codex_calls: int
    total_tokens: int
    total_cost_usd: float

    avg_card_confidence: float
    avg_context_score: float

    def cards_per_hour(self) -> float:
        duration = (self.end_time - self.start_time).total_seconds() / 3600
        return self.cards_generated / duration if duration > 0 else 0

    def cost_per_card(self) -> float:
        return self.total_cost_usd / self.cards_generated if self.cards_generated > 0 else 0

    def efficiency_ratio(self) -> float:
        """Ratio of cards generated to contexts processed."""
        return self.cards_generated / self.contexts_sent_to_codex if self.contexts_sent_to_codex > 0 else 0

# Save metrics
metrics_db.save_metrics(run_metrics)

# Generate reports
def print_metrics_report(metrics: RunMetrics):
    print(f"""
╭─ Run Metrics ─────────────────────────────────────╮
│ Run ID: {metrics.run_id}                          │
│ Duration: {metrics.end_time - metrics.start_time}  │
├───────────────────────────────────────────────────┤
│ Files scanned:        {metrics.files_scanned:>6} │
│ Contexts extracted:   {metrics.contexts_extracted:>6} │
│ Contexts filtered:    {metrics.contexts_filtered:>6} │
│ Contexts → Codex:     {metrics.contexts_sent_to_codex:>6} │
├───────────────────────────────────────────────────┤
│ Cards generated:      {metrics.cards_generated:>6} │
│ Cards accepted:       {metrics.cards_accepted:>6} │
│ Cards rejected:       {metrics.cards_rejected:>6} │
├───────────────────────────────────────────────────┤
│ Codex calls:          {metrics.codex_calls:>6} │
│ Total tokens:         {metrics.total_tokens:>6} │
│ Total cost:          ${metrics.total_cost_usd:>6.2f} │
├───────────────────────────────────────────────────┤
│ Efficiency:           {metrics.efficiency_ratio():>6.1%} │
│ Cards/hour:           {metrics.cards_per_hour():>6.1f} │
│ Cost/card:           ${metrics.cost_per_card():>6.3f} │
│ Avg confidence:       {metrics.avg_card_confidence:>6.2f} │
╰───────────────────────────────────────────────────╯
""")
```

**OpenTelemetry integration** (for production deployments):

``` python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer(__name__)

def harvest_chat_contexts(...):
    with tracer.start_as_current_span("harvest_contexts") as span:
        span.set_attribute("chat_root", str(chat_root))
        span.set_attribute("max_contexts", args.max_contexts)

        # ... processing ...

        span.set_attribute("contexts_found", len(contexts))
        return contexts
```

## Cost & Resource Optimization

### 29. Smart Batching & Prompt Caching

**Reduce token costs**:

``` python
def optimize_prompt_payload(cards: List[Card], contexts: List[ChatTurn]) -> str:
    """Compress payload while preserving information."""

    # Instead of full cards, send compact summaries
    card_summaries = [
        {
            'f': card.front[:100],  # Truncate to 100 chars
            'b': card.back[:150],
            'd': card.deck
        }
        for card in cards
    ]

    # Group similar contexts to reduce redundancy
    grouped_contexts = group_similar_contexts(contexts)

    # Use Anthropic's prompt caching for static parts
    cached_section = {
        "type": "text",
        "text": STATIC_INSTRUCTIONS,
        "cache_control": {"type": "ephemeral"}
    }

    return build_cached_prompt(cached_section, card_summaries, grouped_contexts)

def group_similar_contexts(contexts: List[ChatTurn]) -> List[Dict]:
    """Group contexts by topic to reduce duplication in prompt."""
    # Use embeddings to cluster
    clusters = cluster_by_topic(contexts, n_clusters=5)

    grouped = []
    for cluster in clusters:
        grouped.append({
            'topic': infer_topic(cluster),
            'contexts': cluster
        })

    return grouped
```

**Prompt caching with Anthropic** (70-90% cost reduction on repeated
sections):

``` python
def build_cached_prompt(cards, contexts):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": STATIC_INSTRUCTIONS,  # ~2000 tokens
                    "cache_control": {"type": "ephemeral"}  # Cache this!
                },
                {
                    "type": "text",
                    "text": json.dumps({"existing_cards": cards}),
                    "cache_control": {"type": "ephemeral"}  # Cache cards too
                },
                {
                    "type": "text",
                    "text": json.dumps({"contexts": contexts})  # Only this changes
                }
            ]
        }
    ]

    return messages
```

### 30. Cost Tracking & Budget Limits

**Monitor spending in real-time**:

``` python
class CostTracker:
    def __init__(self, max_per_run=5.0, max_per_day=20.0):
        self.max_per_run = max_per_run
        self.max_per_day = max_per_day
        self.current_run_cost = 0.0
        self.today_cost = self._load_today_cost()

    def estimate_cost(self, prompt: str, model: str) -> float:
        """Estimate cost before making API call."""
        tokens = len(prompt) // 4  # Rough estimate

        # Model pricing (per 1M tokens)
        rates = {
            'gpt-5': {'input': 10.0, 'output': 30.0},
            'gpt-4o': {'input': 2.5, 'output': 10.0},
            'claude-opus': {'input': 15.0, 'output': 75.0},
            'claude-sonnet': {'input': 3.0, 'output': 15.0}
        }

        rate = rates.get(model, rates['gpt-5'])
        input_cost = (tokens / 1_000_000) * rate['input']
        output_cost = (tokens * 0.5 / 1_000_000) * rate['output']  # Assume 0.5x output tokens

        return input_cost + output_cost

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if we're within budget."""
        if self.current_run_cost + estimated_cost > self.max_per_run:
            return False
        if self.today_cost + estimated_cost > self.max_per_day:
            return False
        return True

    def record_cost(self, actual_cost: float):
        """Record actual cost after API call."""
        self.current_run_cost += actual_cost
        self.today_cost += actual_cost
        self._save_today_cost()

    def get_budget_remaining(self) -> Dict[str, float]:
        return {
            'run': self.max_per_run - self.current_run_cost,
            'day': self.max_per_day - self.today_cost
        }

# Usage
cost_tracker = CostTracker(max_per_run=5.0, max_per_day=20.0)

for chunk in chunks:
    prompt = build_prompt(chunk)
    estimated = cost_tracker.estimate_cost(prompt, model='gpt-5')

    if not cost_tracker.can_afford(estimated):
        logger.warning(
            "budget_exceeded",
            estimated=estimated,
            remaining=cost_tracker.get_budget_remaining()
        )
        break  # Stop processing to stay within budget

    response = call_codex(prompt)
    cost_tracker.record_cost(response.usage.total_cost)
```

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

**High impact, low effort**

1.  [x] Parallel Stage 2 execution ✅ (3 concurrent workers)
2.  [x] Interactive review mode ✅ (Shiny UI)
3.  [ ] Better error recovery (robust JSON parsing)
4.  [ ] Cost tracking & budget limits
5.  [ ] Progress indicators with tqdm

### Phase 2: Intelligence (2-4 weeks)

**Improve card quality**

1.  [x] Two-stage LLM pipeline ✅ (cost reduction)
    - Enabled by default (`--two-stage`)
    - Stage 1: fast filter with full conversations, LLM judges quality directly
    - Stage 2: parallel card generation (3 workers)
    - Heuristics now optional (`--use-filter-heuristics`)
2.  [x] Semantic deduplication with embeddings ✅
    - SentenceTransformers + FAISS-based vector cache
3.  [ ] Enhanced scoring heuristics (now optional, via `--use-filter-heuristics`)
4.  [ ] Quality validation & auto-fix
5.  [ ] Configurable prompt templates

### Phase 3: Integration (2-3 weeks)

**Reduce friction**

1.  [x] AnkiConnect integration ✅ DONE
2.  [ ] SQLite state backend
3.  [ ] Multi-source support (PDFs, YouTube)
4.  [ ] Watch mode for continuous processing
5.  [x] Web dashboard for progress tracking ✅ DONE (Shiny UI)

### Phase 4: Advanced Features (4-6 weeks)

**Next-generation capabilities**

1.  [ ] Active learning from feedback
2.  [ ] Concept graph visualization
3.  [x] Context clustering & topic modeling ✅ PARTIAL (conversation-level grouping)
4.  [ ] Multi-modal cards (images, audio)
5.  [ ] Plugin architecture

### Phase 5: Polish & Scale (2-3 weeks)

**Production-ready**

1.  [ ] Comprehensive test suite
2.  [ ] Documentation & examples
3.  [ ] Docker containerization
4.  [ ] Observability & metrics
5.  [ ] Community template registry

## Open Questions & Research Directions

### Research Questions

1.  **Optimal scoring function**: Can we learn the perfect weights for
    heuristics using supervised learning on user feedback?

2.  **Semantic chunking**: What's the best way to segment long
    conversations into coherent learning units?
    - ✅ ADDRESSED: Implemented `split_conversation_by_topic()` using term overlap
      and explicit markers. Configurable via `--conversation-max-turns` and
      `--conversation-max-chars`.

3.  **Card difficulty prediction**: Can we accurately predict SuperMemo
    ease factors from content analysis?

4.  **Knowledge graph construction**: How to automatically build concept
    dependency graphs from unstructured text?

5.  **Multi-modal fusion**: When/how should we augment text cards with
    images or audio?

### Technical Challenges

1.  **Scalability**: How to handle 100k+ cards and 10k+ daily contexts
    efficiently?

2.  **Quality consistency**: How to ensure LLM-generated cards maintain
    consistent quality across runs?

3.  **Personalization**: How to adapt to individual learning styles and
    preferences automatically?

4.  **Real-time processing**: Can we reduce latency to \<5 seconds for
    interactive use?

5.  **Offline mode**: How to support local LLMs for
    privacy/cost-sensitive users?

### Design Decisions

1.  **Stateful vs stateless**: Should the agent maintain long-term
    context about user preferences, or operate statelessly?

2.  **Batch vs streaming**: Trade-offs between batch processing
    (efficient) vs streaming (responsive)?

3.  **Centralized vs distributed**: Should card generation happen
    locally or on a server?

4.  **Opinionated vs configurable**: How much flexibility to expose vs
    sensible defaults?

5.  **Monolith vs microservices**: Keep as single script or split into
    specialized services?

## Metrics for Success

### Quality Metrics

-   **Acceptance rate**: % of generated cards that user imports
-   **Retention rate**: % of cards still active after 30/90 days
-   **Review accuracy**: % correct on first review
-   **Card edit rate**: % of cards that need manual editing

### Efficiency Metrics

-   **Cards per hour**: Throughput of card generation
-   **Cost per card**: USD spent per useful card
-   **Dedup accuracy**: Precision/recall of duplicate detection
-   **Context utilization**: % of harvested contexts that yield cards

### User Experience Metrics

-   **Time to first card**: Latency from run to reviewable output
-   **Friction points**: Where do users abandon the workflow?
-   **Learning curve**: Time to proficiency with tool
-   **Satisfaction score**: NPS or similar user rating

## Conclusion

The Auto Anki Agent has evolved from a batch processor into a production-ready
intelligent learning companion. The highest-leverage improvements are now complete:

1.  ~~**Semantic deduplication**~~ ✅ **DONE** - FAISS + embeddings with caching
2.  ~~**Interactive review**~~ ✅ **DONE** - Shiny web app with keyboard shortcuts
3.  ~~**Two-stage pipeline**~~ ✅ **DONE** - Fast filter + slow generation
4.  ~~**AnkiConnect**~~ ✅ **DONE** - One-click import, 30-60x faster
5.  ~~**Conversation-level processing**~~ ✅ **DONE** - LLM sees full learning journey
6.  ~~**Parallel Stage 2**~~ ✅ **DONE** - 3 concurrent workers for card generation
7.  ~~**Full context to Stage 1**~~ ✅ **DONE** - LLM judges quality directly
8.  **Active learning** - Biggest remaining potential

**Recent enhancements:**
-   Heuristics now **optional** (`--use-filter-heuristics`, off by default)
-   Stage 1 receives **full conversations** (no truncation)
-   Stage 2 runs with **3 parallel workers** via ThreadPoolExecutor

**Remaining opportunities:**
-   Plugin architecture for custom scorers/parsers
-   Multi-source support (PDFs, YouTube, etc.)
-   Active learning from user feedback
-   SQLite state backend for scale

The metrics and observability infrastructure ensure continuous improvement based on
data.

Long-term vision: An autonomous agent that discovers your knowledge
gaps, curates optimal learning material from all your information
sources, generates perfect cards, and adapts to your learning
patterns---all while staying within budget and respecting your time.
