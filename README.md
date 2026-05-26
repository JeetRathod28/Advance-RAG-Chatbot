# Advanced RAG Chatbot

A production-grade Retrieval-Augmented Generation chatbot with hybrid retrieval, agentic reasoning, streaming responses, and a modern React UI.

## Architecture

```
User Query
    │
Query Understanding → Query Rewriting → Multi-Query Generation
    │
Router Decision (LLM)
    ├── vector_only → FAISS + BM25 (Hybrid RRF)
    ├── web_only    → Tavily Web Search
    ├── both        → Hybrid + Web
    └── direct      → LLM direct answer
    │
Reranking (BGE-reranker-large)  top-20 → top-5
    │
Context Compression
    │
Answer Generation (Groq llama-3.3-70b-versatile)  ← streamed
    │
Validation → Memory Update → SSE Response
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim) |
| Reranker | `BAAI/bge-reranker-large` |
| Vector Store | FAISS (cosine similarity) |
| Sparse Search | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF k=60) |
| Web Search | Tavily API |
| Agent | LangGraph `StateGraph` |
| Backend | FastAPI + SSE streaming |
| Frontend | React 18 + Vite + Tailwind CSS |

## Setup

### 1. Prerequisites

- Python 3.11+
- Node.js 20+
- Groq API key — [console.groq.com](https://console.groq.com)
- Tavily API key — [app.tavily.com](https://app.tavily.com)

### 2. Clone and configure

```bash
cd "RAG chatbot"
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Backend

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

The API is now at `http://localhost:8000`  
Swagger UI: `http://localhost:8000/api/docs`

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

### 5. Docker (full stack)

```bash
cp .env.example .env
# Add API keys to .env

docker-compose up --build
```

Open `http://localhost:8000`

## API Reference

### `POST /api/chat/stream`
Stream a response via SSE.

```json
{
  "session_id": "user-123",
  "message": "What does this document say about pricing?",
  "use_web_search": null
}
```

SSE events:
```
data: {"type": "status",  "content": "Retrieved 5 passages"}
data: {"type": "token",   "content": "Based on the..."}
data: {"type": "sources", "content": [{...}]}
data: {"type": "done",    "content": ""}
```

### `POST /api/chat`
Non-streaming JSON response.

### `POST /api/upload`
Upload a document (PDF, TXT, DOCX, MD). Max 50 MB.

### `GET /api/health`
Returns system status, vector store size, and model loading state.

### `GET /api/sources`
Lists all indexed documents.

## Usage

1. **Upload documents** — Click "Upload" and drag/drop PDF, TXT, or DOCX files
2. **Ask questions** — Type your question and press Enter
3. **Choose search mode**:
   - **Documents** — query only your uploaded files
   - **Auto** — let the router decide (recommended)
   - **Web Search** — always include Tavily results
4. **View sources** — Click the sources section below any AI response

## Project Structure

```
├── app/
│   ├── agents/          # LangGraph stateful agent
│   ├── retrieval/       # Dense (FAISS), sparse (BM25), hybrid (RRF)
│   ├── rerank/          # BGE cross-encoder reranker
│   ├── memory/          # Per-session conversation memory
│   ├── ingestion/       # Document loader + chunker + pipeline
│   ├── prompts/         # All prompt templates
│   ├── tools/           # Tavily search, query rewriter
│   ├── api/             # FastAPI routes
│   ├── config/          # Pydantic settings
│   ├── models/          # Request/response schemas
│   ├── services/        # Embedding, LLM, compression services
│   └── utils/           # Logger, helpers
├── frontend/            # React + Vite + Tailwind
├── vector_store/        # FAISS index (auto-created)
├── data/                # Uploaded documents
├── logs/                # Application logs
└── tests/               # pytest test suite
```

## Performance Notes

- First startup downloads embedding model (~1.5 GB) and reranker (~1 GB) from HuggingFace
- Models are cached in `~/.cache/huggingface/` after first download
- Use `--workers 1` with LangGraph (stateful graph is not multi-process safe)
- For high traffic, use async FAISS + multiple Gunicorn processes with Redis checkpointer

## Security

- API keys are only read from environment variables
- File uploads are validated by extension and size
- CORS is restricted to configured origins
- No database SQL surface — FAISS is file-based
