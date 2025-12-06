#!/bin/bash
# Launch script for Auto Anki Review UI with AnkiConnect Integration

echo "🎓 Auto Anki Card Review UI"
echo "================================================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Test AnkiConnect connection
echo "🔌 Testing AnkiConnect..."
if python3 -c "from anki_connect import AnkiConnectClient; client = AnkiConnectClient(); exit(0 if client.check_connection() else 1)" 2>/dev/null; then
    echo "✓ AnkiConnect is available - direct import enabled!"
else
    echo "⚠ AnkiConnect not connected"
    echo "  You can still review cards and export to JSON"
    echo "  For direct import: start Anki and install AnkiConnect (code: 2055492159)"
fi

echo ""
echo "🚀 Starting Shiny app..."
echo "   The UI will open automatically in your browser"
echo "   Press Ctrl+C to stop the server"
echo "================================================================="
echo ""

# Launch Shiny app
shiny run anki_review_ui.py --reload
