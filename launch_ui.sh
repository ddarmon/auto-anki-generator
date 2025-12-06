#!/bin/bash
# Launch script for Auto Anki Review UI

echo "🎓 Auto Anki Card Review UI"
echo "=" "================================================================="
echo ""
echo "Starting Shiny app..."
echo "The UI will open automatically in your browser."
echo ""
echo "To stop the server, press Ctrl+C"
echo "=" "================================================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Launch Shiny app
shiny run anki_review_ui.py --reload
