# CRAG + Self-Reflective RAG Implementation

Minimal FastAPI application demonstrating **Corrective RAG (CRAG)** and **Self-Reflective RAG** patterns without LangChain.

## Features

- **Document Processing**: Upload PDF, MD, TXT, JSON files
- **Docling Integration**: HybridChunker for intelligent document chunking
- **Vector Storage**: Qdrant for efficient similarity search
- **CRAG**: Pre-generation relevance evaluation with web search fallback
- **Self-Reflective RAG**: Post-generation grounding validation with iterative refinement
- **Comparison Mode**: Side-by-side evaluation of all approaches

## Architecture

### Corrective RAG (CRAG)
1. Retrieve documents from vector store
2. **Evaluate relevance** before generation (LLM-based grader)
3. Route based on evaluation:
   - **Relevant**: Use retrieved documents
   - **Ambiguous**: Augment with web search (Tavily)
   - **Irrelevant**: Replace with web search only
4. Generate answer with optimal context

### Self-Reflective RAG
1. Retrieve documents from vector store
2. Generate initial answer
3. **Reflect on answer** to check grounding
4. If not grounded:
   - Refine query based on reflection
   - Re-retrieve with refined query
   - Generate again (up to max iterations)
5. Return best grounded answer

---

## 🚀 Quick Start with uv

This project uses [**uv**](https://docs.astral.sh/uv/) - the extremely fast Python package manager written in Rust.

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Docker (for Qdrant)
- OpenAI API key
- Tavily API key

### Installation & Setup

1. **Clone and navigate to the project**:
```bash
cd crag-reflective-rag
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - TAVILY_API_KEY
```

3. **Start Qdrant** (in a separate terminal):
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4. **Install dependencies and run**:
```bash
# uv automatically creates .venv and installs dependencies
uv sync

# Run the application
uv run python -m app.main
```

The API will be available at: http://localhost:8000

API docs at: http://localhost:8000/docs

---

## 📦 uv Commands Reference

### Development Workflow

```bash
# Install all dependencies (including dev)
uv sync

# Install only production dependencies
uv sync --no-dev

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Remove a dependency
uv remove <package-name>

# Update lockfile after manual pyproject.toml changes
uv lock

# Run a command in the project environment
uv run <command>

# Run the FastAPI app with auto-reload
uv run uvicorn app.main:app --reload

# Run with specific host/port
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running Scripts & Tests

```bash
# Run Python script
uv run python script.py

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run linting
uv run ruff check .

# Run type checking
uv run mypy app/
```

### Managing Python Versions

```bash
# Pin Python version (creates .python-version file)
uv python pin 3.12

# List available Python versions
uv python list

# Install a specific Python version
uv python install 3.12
```

### Virtual Environment

uv automatically manages the virtual environment in `.venv/`. You don't need to activate it manually when using `uv run`, but you can if needed:

```bash
# Activate (traditional way)
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Deactivate
deactivate
```

---

## 📡 API Usage

### 1. Upload Document

```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@path/to/document.pdf"
```

Response:
```json
{
  "file_id": "uuid-here",
  "filename": "document.pdf",
  "file_type": "pdf",
  "chunks_created": 45,
  "status": "success",
  "message": "Document processed successfully with 45 chunks"
}
```

### 2. Query - Standard RAG

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "mode": "standard",
    "top_k": 5
  }'
```

### 3. Query - CRAG Mode

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "mode": "crag",
    "top_k": 5
  }'
```

Response includes:
- Answer
- Retrieved chunks
- **CRAG evaluation** (relevance_score, routing decision)
- Web search results (if triggered)

### 4. Query - Self-Reflective Mode

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the methodology",
    "mode": "self_reflective",
    "top_k": 5
  }'
```

Response includes:
- Final answer (after reflection)
- **Reflection details** (grounding score, iterations)
- Sources used

### 5. Query - Both CRAG + Self-Reflective

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main conclusion?",
    "mode": "both",
    "top_k": 5
  }'
```

Combines both approaches for maximum quality.

### 6. Compare All Modes

```bash
curl "http://localhost:8000/query/compare?query=What%20are%20the%20key%20findings&top_k=5"
```

Returns side-by-side comparison of:
- Standard RAG
- CRAG
- Self-Reflective RAG

---

## 🧪 Testing Strategy

### Test CRAG Routing

**Test with relevant documents**:
```bash
# Upload a document about "machine learning"
curl -X POST "http://localhost:8000/upload/" -F "file=@ml_paper.pdf"

# Query about machine learning (should be "relevant")
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gradient descent?", "mode": "crag"}'
```
Expected: `relevance_label: "relevant"`, no web search

**Test with irrelevant query**:
```bash
# Query about something NOT in the document
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather today?", "mode": "crag"}'
```
Expected: `relevance_label: "irrelevant"`, triggers Tavily web search

### Test Self-Reflective Refinement

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about the results", "mode": "self_reflective"}'
```
Expected: Multiple iterations, query refinement, improved final answer

### Run Automated Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_crag.py
```

---

## 📁 Project Structure

```
crag-reflective-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Pydantic settings
│   ├── models.py               # Pydantic schemas
│   ├── api/
│   │   ├── __init__.py
│   │   ├── upload.py           # Document upload endpoints
│   │   └── query.py            # Query endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── retrieval.py        # Retrieval service
│   └── services/
│       ├── __init__.py
│       ├── llm_service.py      # OpenAI LLM
│       ├── embedding_service.py # OpenAI embeddings
│       ├── vector_store.py     # Qdrant operations
│       ├── document_processor.py # Docling chunking
│       ├── web_search.py       # Tavily integration
│       ├── crag.py             # CRAG logic
│       └── self_reflective.py  # Self-Reflective RAG logic
├── uploads/                    # File storage (gitignored)
├── .venv/                      # Virtual environment (gitignored, managed by uv)
├── .python-version             # Python version pin (3.12)
├── pyproject.toml             # Project config & dependencies
├── uv.lock                    # Lockfile (generated by uv)
├── .env                       # Environment variables (gitignored)
├── .env.example               # Environment template
├── .gitignore
└── README.md                  # This file
```

---

## ⚙️ Key Implementation Details

### Metadata Richness

Every chunk includes:
- Document structure (page, heading, hierarchy)
- Content metrics (tokens, characters)
- Processing info (chunk method, timestamps)
- Extracted keywords

### CRAG Evaluation

Uses LLM-as-judge with structured JSON output:
```python
{
  "relevance_score": 0.87,
  "relevance_label": "relevant",  # relevant/ambiguous/irrelevant
  "confidence": 0.92
}
```

Thresholds (configurable in `.env`):
- `relevant`: score >= 0.7
- `ambiguous`: 0.4 <= score < 0.7
- `irrelevant`: score < 0.4

### Self-Reflective Validation

Post-generation grounding check:
```python
{
  "answer_grounded": true,
  "hallucination_detected": false,
  "reflection_score": 0.95,
  "sources_cited": ["chunk_005", "chunk_007"]
}
```

If `reflection_score < 0.8`, refine query and retry (up to 2 iterations).

---

## 🛠️ Troubleshooting

### Qdrant Connection Error
```
Error: Connection refused to localhost:6333
```
Solution: Start Qdrant with Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### OpenAI API Error
```
Error: Invalid API key
```
Solution: Check `.env` file has correct `OPENAI_API_KEY`

### uv not found
```
command not found: uv
```
Solution: Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Empty Retrieval Results
```
404: No relevant documents found
```
Solution: Upload documents first using `/upload/` endpoint

---

## 📝 Next Steps

This implementation provides a solid foundation. Consider adding:

1. **Evaluation Framework**: Integrate RAGAS/DeepEval for systematic evaluation
2. **Hybrid Search**: Combine dense + sparse (BM25) retrieval
3. **Reranking**: Add cross-encoder reranking post-retrieval
4. **Streaming**: Add streaming support for real-time responses
5. **Caching**: Implement prompt caching for repeat queries
6. **Authentication**: Add API key authentication
7. **Database**: Persist metadata in PostgreSQL alongside Qdrant

---

## 📄 License

MIT License - feel free to use for educational purposes!

---

Built for demonstrating CRAG and Self-Reflective RAG patterns in production-ready architecture.

**Powered by [uv](https://docs.astral.sh/uv/) ⚡ - The fast Python package manager**
