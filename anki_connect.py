#!/usr/bin/env python3
"""
AnkiConnect client for Auto Anki Agent.

Provides programmatic access to Anki via the AnkiConnect plugin.
AnkiConnect must be installed and Anki must be running for this to work.

Installation:
1. Install AnkiConnect plugin in Anki (code: 2055492159)
2. Restart Anki
3. Ensure Anki is running when using these functions

Documentation: https://foosoft.net/projects/anki-connect/
"""

import json
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any


def markdown_to_anki_html(text: str) -> str:
    """Convert Markdown formatting to Anki-compatible HTML.

    Handles:
    - **bold** and __bold__ → <b>bold</b>
    - *italic* and _italic_ → <i>italic</i>
    - `inline code` → <code>inline code</code>
    - ```code blocks``` → <pre>code blocks</pre>
    - Unordered lists (- item) → <ul><li>item</li></ul>
    - Ordered lists (1. item) → <ol><li>item</li></ol>
    - Line breaks → <br>

    LaTeX notation (\\(...\\) and \\[...\\]) is preserved for MathJax.
    """
    if not text:
        return text

    # Protect LaTeX from being processed
    latex_placeholders = []

    def protect_latex(match):
        latex_placeholders.append(match.group(0))
        return f"__LATEX_{len(latex_placeholders) - 1}__"

    # Protect display math \[...\] and inline math \(...\)
    text = re.sub(r'\\\[.*?\\\]', protect_latex, text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', protect_latex, text, flags=re.DOTALL)
    # Also protect $...$ style if present
    text = re.sub(r'\$\$.*?\$\$', protect_latex, text, flags=re.DOTALL)
    text = re.sub(r'\$[^$\n]+\$', protect_latex, text)

    # Code blocks (must be before inline code)
    text = re.sub(r'```(\w*)\n?(.*?)```', r'<pre>\2</pre>', text, flags=re.DOTALL)

    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # Bold (**text**) - only asterisk style to avoid matching underscores in code/math
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)

    # Italic (*text*) - only asterisk style, be careful not to match inside words
    text = re.sub(r'(?<![*\w])\*([^*]+)\*(?![*\w])', r'<i>\1</i>', text)

    # Blockquotes
    text = re.sub(r'^>\s*(.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)

    # Process lists (unordered and ordered)
    lines = text.split('\n')
    result_lines = []
    in_ul = False
    in_ol = False

    for line in lines:
        stripped = line.strip()

        # Check for unordered list item
        ul_match = re.match(r'^[-*+]\s+(.+)$', stripped)
        # Check for ordered list item
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)

        if ul_match:
            if not in_ul:
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                result_lines.append('<ul>')
                in_ul = True
            result_lines.append(f'<li>{ul_match.group(1)}</li>')
        elif ol_match:
            if not in_ol:
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                result_lines.append('<ol>')
                in_ol = True
            result_lines.append(f'<li>{ol_match.group(1)}</li>')
        else:
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            # Add <br> for non-list lines (except empty lines which become paragraph breaks)
            if stripped:
                result_lines.append(line + '<br>')
            else:
                result_lines.append('')  # Keep empty lines for paragraph detection

    if in_ul:
        result_lines.append('</ul>')
    if in_ol:
        result_lines.append('</ol>')

    text = '\n'.join(result_lines)

    # Convert double+ newlines to paragraph breaks
    text = re.sub(r'\n\n+', '</p><p>', text)
    # Remove remaining newlines (they're just formatting artifacts now)
    text = re.sub(r'\n', '', text)
    # Clean up trailing <br> before block elements
    text = re.sub(r'<br>(</?(?:ul|ol|li|p|pre|blockquote)>)', r'\1', text)
    # Clean up <br> at very end
    text = re.sub(r'<br>$', '', text)

    # Wrap in paragraph if we added paragraph breaks
    if '</p><p>' in text:
        text = '<p>' + text + '</p>'

    # Restore LaTeX
    for i, latex in enumerate(latex_placeholders):
        text = text.replace(f"__LATEX_{i}__", latex)

    return text


class AnkiConnectError(Exception):
    """Raised when AnkiConnect API returns an error."""
    pass


class AnkiConnectClient:
    """Client for interacting with Anki via AnkiConnect plugin."""

    def __init__(self, url: str = "http://localhost:8765"):
        """
        Initialize AnkiConnect client.

        Args:
            url: AnkiConnect API endpoint (default: http://localhost:8765)
        """
        self.url = url
        self.version = 6

    def _invoke(self, action: str, **params) -> Any:
        """
        Invoke an AnkiConnect API action.

        Args:
            action: API action name
            **params: Parameters for the action

        Returns:
            API response result

        Raises:
            AnkiConnectError: If the API returns an error
            ConnectionError: If cannot connect to Anki
        """
        request_data = {
            'action': action,
            'version': self.version,
            'params': params
        }

        try:
            request = urllib.request.Request(
                self.url,
                data=json.dumps(request_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                response_data = json.loads(response.read().decode('utf-8'))

            if len(response_data) != 2:
                raise AnkiConnectError(f'Invalid response format: {response_data}')

            if response_data['error'] is not None:
                raise AnkiConnectError(response_data['error'])

            return response_data['result']

        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to Anki. "
                f"Make sure Anki is running and AnkiConnect is installed. "
                f"Error: {e}"
            )

    def check_connection(self) -> bool:
        """
        Check if AnkiConnect is available.

        Returns:
            True if connected, False otherwise
        """
        try:
            self._invoke('version')
            return True
        except (AnkiConnectError, ConnectionError):
            return False

    def get_version(self) -> int:
        """Get AnkiConnect version."""
        return self._invoke('version')

    # ========================================================================
    # Deck Operations
    # ========================================================================

    def get_deck_names(self) -> List[str]:
        """Get list of all deck names."""
        return self._invoke('deckNames')

    def get_deck_stats(self, deck: str) -> Dict:
        """
        Get statistics for a deck.

        Args:
            deck: Deck name

        Returns:
            Dict with deck statistics
        """
        return self._invoke('getDeckStats', deck=deck)

    def create_deck(self, deck: str) -> int:
        """
        Create a new deck.

        Args:
            deck: Deck name (can use :: for nested, e.g. "Parent::Child")

        Returns:
            Deck ID
        """
        return self._invoke('createDeck', deck=deck)

    # ========================================================================
    # Note/Card Operations
    # ========================================================================

    def add_note(
        self,
        deck: str,
        front: str,
        back: str,
        tags: Optional[List[str]] = None,
        model: str = "Basic",
        allow_duplicate: bool = False
    ) -> Optional[int]:
        """
        Add a note (card) to Anki.

        Args:
            deck: Deck name
            front: Front of card
            back: Back of card
            tags: List of tags
            model: Note type (default: "Basic")
            allow_duplicate: Whether to allow duplicate notes

        Returns:
            Note ID if successful, None if duplicate and not allowed

        Raises:
            AnkiConnectError: If the operation fails
        """
        tags = tags or []

        note = {
            'deckName': deck,
            'modelName': model,
            'fields': {
                'Front': front,
                'Back': back
            },
            'tags': tags,
            'options': {
                'allowDuplicate': allow_duplicate
            }
        }

        try:
            return self._invoke('addNote', note=note)
        except AnkiConnectError as e:
            if 'duplicate' in str(e).lower():
                return None  # Duplicate note
            raise

    def add_notes(self, notes: List[Dict]) -> List[Optional[int]]:
        """
        Add multiple notes at once.

        Args:
            notes: List of note dictionaries in AnkiConnect format

        Returns:
            List of note IDs (None for failed/duplicate notes)
        """
        return self._invoke('addNotes', notes=notes)

    def find_notes(self, query: str) -> List[int]:
        """
        Find notes matching a query.

        Args:
            query: Anki search query (e.g., "deck:MyDeck")

        Returns:
            List of note IDs
        """
        return self._invoke('findNotes', query=query)

    def get_notes_info(self, note_ids: List[int]) -> List[Dict]:
        """
        Get information about notes.

        Args:
            note_ids: List of note IDs

        Returns:
            List of note info dictionaries
        """
        return self._invoke('notesInfo', notes=note_ids)

    # ========================================================================
    # Model (Note Type) Operations
    # ========================================================================

    def get_model_names(self) -> List[str]:
        """Get list of all note type names."""
        return self._invoke('modelNames')

    def get_model_field_names(self, model: str) -> List[str]:
        """
        Get field names for a note type.

        Args:
            model: Note type name

        Returns:
            List of field names
        """
        return self._invoke('modelFieldNames', modelName=model)

    # ========================================================================
    # Convenience Methods for Auto Anki Agent
    # ========================================================================

    @staticmethod
    def _has_cloze_syntax(text: str) -> bool:
        """Check if text contains Anki cloze deletion syntax {{c1::...}}."""
        return bool(re.search(r'\{\{c\d+::', text))

    def import_card(
        self,
        card: Dict,
        allow_duplicate: bool = False,
        convert_markdown: bool = True
    ) -> Optional[int]:
        """
        Import a card from Auto Anki Agent format.

        Automatically detects cloze syntax and uses appropriate note type.

        Args:
            card: Card dictionary with keys: deck, front, back, tags
            allow_duplicate: Whether to allow duplicate cards
            convert_markdown: Whether to convert Markdown to HTML (default: True)

        Returns:
            Note ID if successful, None if duplicate

        Raises:
            AnkiConnectError: If the operation fails
        """
        front = card['front']
        back = card['back']

        if convert_markdown:
            front = markdown_to_anki_html(front)
            back = markdown_to_anki_html(back)

        # Detect cloze syntax and use appropriate note type
        if self._has_cloze_syntax(front) or self._has_cloze_syntax(back):
            # Use Cloze note type - combine front and back into Text field
            cloze_text = front
            if back and back.strip():
                cloze_text += "<br><br>" + back
            return self._add_cloze_note(
                deck=card.get('deck', 'Default'),
                text=cloze_text,
                tags=card.get('tags', []),
                allow_duplicate=allow_duplicate
            )
        else:
            return self.add_note(
                deck=card.get('deck', 'Default'),
                front=front,
                back=back,
                tags=card.get('tags', []),
                allow_duplicate=allow_duplicate
            )

    def _add_cloze_note(
        self,
        deck: str,
        text: str,
        tags: Optional[List[str]] = None,
        allow_duplicate: bool = False
    ) -> Optional[int]:
        """Add a cloze deletion note to Anki."""
        tags = tags or []

        note = {
            'deckName': deck,
            'modelName': 'Cloze',
            'fields': {
                'Text': text,
                'Extra': ''
            },
            'tags': tags,
            'options': {
                'allowDuplicate': allow_duplicate
            }
        }

        try:
            return self._invoke('addNote', note=note)
        except AnkiConnectError as e:
            if 'duplicate' in str(e).lower():
                return None
            raise

    def import_cards_batch(
        self,
        cards: List[Dict],
        allow_duplicates: bool = False,
        create_decks: bool = True,
        convert_markdown: bool = True
    ) -> Dict[str, int]:
        """
        Import multiple cards at once.

        Automatically detects cloze syntax and uses appropriate note type.

        Args:
            cards: List of card dictionaries
            allow_duplicates: Whether to allow duplicate cards
            create_decks: Whether to create decks if they don't exist
            convert_markdown: Whether to convert Markdown to HTML (default: True)

        Returns:
            Dict with 'imported', 'duplicates', 'failed' counts
        """
        # Get existing decks
        existing_decks = set(self.get_deck_names())

        # Create missing decks if requested
        if create_decks:
            needed_decks = set(card.get('deck', 'Default') for card in cards)
            for deck in needed_decks:
                if deck not in existing_decks:
                    self.create_deck(deck)

        # Separate basic and cloze cards
        basic_notes = []
        cloze_notes = []

        for card in cards:
            front = card['front']
            back = card['back']

            if convert_markdown:
                front = markdown_to_anki_html(front)
                back = markdown_to_anki_html(back)

            if self._has_cloze_syntax(front) or self._has_cloze_syntax(back):
                # Cloze note - combine front and back into Text field
                cloze_text = front
                if back and back.strip():
                    cloze_text += "<br><br>" + back
                cloze_notes.append({
                    'deckName': card.get('deck', 'Default'),
                    'modelName': 'Cloze',
                    'fields': {
                        'Text': cloze_text,
                        'Extra': ''
                    },
                    'tags': card.get('tags', []),
                    'options': {
                        'allowDuplicate': allow_duplicates
                    }
                })
            else:
                # Basic note
                basic_notes.append({
                    'deckName': card.get('deck', 'Default'),
                    'modelName': 'Basic',
                    'fields': {
                        'Front': front,
                        'Back': back
                    },
                    'tags': card.get('tags', []),
                    'options': {
                        'allowDuplicate': allow_duplicates
                    }
                })

        # Add notes in batches
        all_result_ids = []

        if basic_notes:
            basic_ids = self.add_notes(basic_notes)
            all_result_ids.extend(basic_ids)

        if cloze_notes:
            cloze_ids = self.add_notes(cloze_notes)
            all_result_ids.extend(cloze_ids)

        # Count results
        imported = sum(1 for id in all_result_ids if id is not None)
        duplicates = sum(1 for id in all_result_ids if id is None)
        failed = len(all_result_ids) - imported - duplicates

        return {
            'total': len(cards),
            'imported': imported,
            'duplicates': duplicates,
            'failed': failed
        }

    def get_deck_card_count(self, deck: str) -> int:
        """
        Get number of cards in a deck.

        Args:
            deck: Deck name

        Returns:
            Number of cards
        """
        note_ids = self.find_notes(f'deck:"{deck}"')
        return len(note_ids)

    def get_existing_cards_for_dedup(self, deck: str) -> List[Dict]:
        """
        Get existing cards from a deck for deduplication.

        Args:
            deck: Deck name

        Returns:
            List of card dictionaries with 'front' and 'back' keys
        """
        # Find notes in deck
        note_ids = self.find_notes(f'deck:"{deck}"')

        if not note_ids:
            return []

        # Get note info
        notes_info = self.get_notes_info(note_ids)

        # Extract front/back
        cards = []
        for note in notes_info:
            fields = note.get('fields', {})
            front = fields.get('Front', {}).get('value', '')
            back = fields.get('Back', {}).get('value', '')

            if front or back:
                cards.append({
                    'front': front,
                    'back': back,
                    'deck': deck,
                    'tags': note.get('tags', [])
                })

        return cards


# ============================================================================
# Standalone Testing
# ============================================================================

def test_connection():
    """Test AnkiConnect connection and display info."""
    client = AnkiConnectClient()

    print("Testing AnkiConnect...")
    print("=" * 60)

    try:
        # Check connection
        if not client.check_connection():
            print("✗ Cannot connect to Anki")
            print("\nMake sure:")
            print("1. Anki is running")
            print("2. AnkiConnect plugin is installed (code: 2055492159)")
            return False

        print("✓ Connected to Anki")

        # Get version
        version = client.get_version()
        print(f"✓ AnkiConnect version: {version}")

        # Get decks
        decks = client.get_deck_names()
        print(f"✓ Found {len(decks)} decks:")
        for deck in decks[:5]:  # Show first 5
            count = client.get_deck_card_count(deck)
            print(f"  - {deck}: {count} cards")
        if len(decks) > 5:
            print(f"  ... and {len(decks) - 5} more")

        # Get note types
        models = client.get_model_names()
        print(f"✓ Available note types: {', '.join(models[:3])}")

        print("\n" + "=" * 60)
        print("✓ AnkiConnect is working correctly!")
        return True

    except ConnectionError as e:
        print(f"✗ Connection error: {e}")
        return False
    except AnkiConnectError as e:
        print(f"✗ API error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    test_connection()
