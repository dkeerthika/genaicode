#!/bin/bash

echo "Setting up RAG Document Assistant..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Create necessary directories
mkdir -p data
mkdir -p chroma_db

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt')" 2>/dev/null || echo "NLTK punkt tokenizer already available"

echo "Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run: streamlit run main.py"
echo ""
echo "For Ollama models, make sure Ollama is running: ollama serve"
echo "For OpenAI models, set your OPENAI_API_KEY in .env file"