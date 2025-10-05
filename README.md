# RAG Document Assistant 📚

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and ChromaDB. Upload your documents and ask questions about their content using advanced natural language processing.

## Features

- **Multiple File Format Support**: Upload PDF, DOCX, and TXT files
- **Intelligent Text Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Multiple Embedding Models**: Choose from various sentence transformer models
- **Flexible LLM Support**: Use local Ollama models or OpenAI's GPT models
- **Interactive Chat Interface**: Ask questions and get contextual answers
- **Source Document Display**: See which parts of your documents were used for each answer
- **Persistent Vector Storage**: Documents are stored in a local ChromaDB database
- **Chat History**: Keep track of your conversations

## Installation

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) (for local models) - *optional*
- OpenAI API key (for OpenAI models) - *optional*

### Setup

1. **Clone and navigate to the project directory:**
   ```bash
   cd /path/to/rag-project
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Or manually install:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install dependencies
   pip install -e .

   # Create necessary directories
   mkdir -p data chroma_db
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

## Usage

### Running the Application

1. **Start the application:**
   ```bash
   source venv/bin/activate
   streamlit run main.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

### Using the Application

1. **Upload Documents:**
   - Click on the file uploader in the left column
   - Select PDF, DOCX, or TXT files
   - Click "Process Documents" to create the vector database

2. **Configure Settings (Sidebar):**
   - Choose your preferred LLM (Ollama or OpenAI)
   - Select embedding model
   - Adjust chunk size and overlap as needed

3. **Ask Questions:**
   - Type your question in the chat interface
   - Click "Get Answer" to receive a response
   - View source documents used for the answer

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
CHROMA_DB_PATH=./chroma_db
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200

# Ollama Configuration
OLLAMA_MODEL=mistral
```

### Model Options

#### Embedding Models
- `all-MiniLM-L6-v2` (default) - Fast and good performance
- `all-mpnet-base-v2` - Higher quality but slower
- `paraphrase-MiniLM-L3-v2` - Optimized for paraphrase detection

#### Language Models
- **Ollama (Local):** mistral, llama2, codellama, etc.
- **OpenAI:** gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview

## Project Structure

```
rag-project/
├── main.py              # Main Streamlit application
├── pyproject.toml       # Project dependencies
├── setup.sh            # Setup script
├── .env.example        # Environment configuration template
├── README.md           # This file
├── data/               # Directory for additional documents
├── chroma_db/          # Vector database storage
└── venv/               # Virtual environment
```

## How It Works

1. **Document Processing:** Files are parsed and text is extracted
2. **Text Chunking:** Documents are split into manageable chunks
3. **Vector Embedding:** Text chunks are converted to vectors using sentence transformers
4. **Vector Storage:** Embeddings are stored in ChromaDB for fast retrieval
5. **Query Processing:** User questions are embedded and used to find relevant chunks
6. **Answer Generation:** LLM generates answers using retrieved context

## Tips for Best Results

- **Chunk Size:** Larger chunks (1000-1500) work well for most documents
- **Chunk Overlap:** 200-300 overlap helps maintain context between chunks
- **Embedding Model:** Start with `all-MiniLM-L6-v2` for good balance of speed and quality
- **Document Types:** Ensure your documents have clean, readable text
- **Question Clarity:** Ask specific, clear questions for better answers

## Troubleshooting

### Common Issues

1. **Ollama Connection Error:**
   - Ensure Ollama is installed and running: `ollama serve`
   - Check that the model is available: `ollama pull mistral`

2. **Memory Issues:**
   - Reduce chunk size for large documents
   - Use a smaller embedding model
   - Process documents in smaller batches

3. **Slow Performance:**
   - Use a smaller embedding model
   - Reduce the number of retrieved chunks (k parameter)
   - Consider using a more powerful machine

4. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -e .`
   - Check Python version (3.11+ required)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review open issues on GitHub
3. Create a new issue with detailed information

---

*Built with ❤️ using LangChain, ChromaDB, Streamlit, and modern NLP techniques*