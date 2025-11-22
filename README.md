# 🏰 Disney Reviews RAG System

## The Challenge

This project implements a Retrieval-Augmented Generation (RAG) system to enable semantic search and question-answering over 42,000+ Disneyland reviews from three locations (California, Paris, Hong Kong). The system transforms natural language questions into accurate, source-grounded answers using OpenAI embeddings, FAISS vector search, and GPT-4.

---

## 📚 Documentation

- **[SYSTEM_DESIGN.md](./SYSTEM_DESIGN.md)** 
  - Complete system architecture, RAG flow diagrams, component deep-dive (tiktoken, FAISS, OpenAI), FastAPI server flow, performance metrics, and design trade-offs

- **[QUICKSTART.md](./QUICKSTART.md)** 
  - Step-by-step tutorial to get started quickly, from installation to running your first query

- **[API_README.md](./API_README.md)** - Detailed API documentation with endpoints, request/response schemas, and usage examples

---

## 🔄 RAG Pipeline Architecture

FastAPI orchestrates both the indexing flow (build FAISS index) and query flow (answer user questions):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI WEB SERVER                                   │
│                     http://localhost:8000                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          STARTUP (One-time)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📦 Application Lifespan (app/main.py)                                      │
│  ├─ Load configuration from .env                                            │
│  ├─ Initialize RAGBuilder                                                   │
│  │   └─ Load/Build FAISS index (rag_index/faiss_*.index)                    │
│  │   └─ Load metadata (rag_index/meta_*.jsonl)                              │
│  ├─ Initialize RAGQueryHandler                                              │
│  │   └─ Ready to handle queries                                             │
│  ├─ Mount Gradio UI at /ui                                                  │
│  └─ Start server ✅                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RUNTIME (Handle Requests)                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│   INDEXING FLOW (Admin/Setup)    │    │   QUERY FLOW (User Requests)     │
│   Offline / Build Phase          │    │   Online / Production            │
└──────────────────────────────────┘    └──────────────────────────────────┘

        Not exposed via API                      HTTP POST /query
        (Run via notebooks/scripts)              {"query": "...", "k": 5}
                │                                         │
                ▼                                         ▼
┌────────────────────────────────┐      ┌────────────────────────────────┐
│  📄 Load Reviews               │      │  🌐 FastAPI Route Handler      │
│  data/DisneylandReviews.csv    │      │  app/api/routes.py             │
│  42,656 reviews                │      └────────────┬───────────────────┘
└────────────┬───────────────────┘                   │
             │                                        ▼
             ▼                            ┌────────────────────────────────┐
┌────────────────────────────────┐        │  ✅ Validate Request           │
│  ✂️ Chunk Text                 │        │  Pydantic: QueryRequest        │
│  RAGBuilder.chunk_texts()      │        │  - query (string)              │
│  tiktoken: 500 tokens/chunk    │        │  - k (1-20, default: 5)        │
└────────────┬───────────────────┘        │  - temperature (0-2)           │
             │                            └────────────┬───────────────────┘
             ▼                                         │
┌────────────────────────────────┐                     ▼
│  🧮 Generate Embeddings        │      ┌────────────────────────────────┐
│  RAGBuilder                    │      │  🔍 RAGQueryHandler.query()    │
│  .get_embeddings_batch()       │      │  app/services/rag_query.py     │
│  OpenAI text-embedding-3       │      ├────────────────────────────────┤
└────────────┬───────────────────┘      │  Step 1: Embed Query           │
             │                          │  • OpenAI API (~80ms)          │
             ▼                          │                                │
┌────────────────────────────────┐      │  Step 2: Search FAISS          │
│  💾 Build FAISS Index          │      │  • index.search(query, k)      │
│  RAGBuilder                    │      │  • <1ms for 45K vectors        │
│  .build_faiss_index()          │      │                                │
│  IndexFlatL2 (L2 distance)     │      │  Step 3: Retrieve Metadata     │
│  Save: faiss_*.index           │      │  • Load from meta_*.jsonl      │
└────────────┬───────────────────┘      │                                │
             │                          │  Step 4: Build Prompt          │
             ▼                          │  • System + Context + Query    │
┌────────────────────────────────┐      │                                │
│  📝 Save Metadata              │      │  Step 5: Generate Answer       │
│  RAGBuilder.save_artifacts()   │      │  • OpenAI GPT-4o-mini          │
│  Format: JSONL                 │      │  • Temperature: 0.2            │
│  Save: meta_*.jsonl            │      │  • (~500ms)                    │
└────────────────────────────────┘      └────────────┬───────────────────┘
                                                     │
         ✅ READY FOR QUERIES                        ▼
         (Index loaded in memory)       ┌────────────────────────────────┐
                                        │  📊 Record Metrics             │
                                        │  • Latency: ~600ms             │
                                        │  • Retrieval quality           │
                                        │  • Cost: ~$0.0001              │
                                        └────────────┬───────────────────┘
                                                     │
                                                     ▼
                                        ┌────────────────────────────────┐
                                        │  📤 Return Response            │
                                        │  JSON: {                       │
                                        │    "query": "...",             │
                                        │    "answer": "...",            │
                                        │    "retrieval_results": [...]  │
                                        │  }                             │
                                        └────────────────────────────────┘
                                                     │
                                                     ▼
                                              Client receives
                                              answer + sources

┌─────────────────────────────────────────────────────────────────────────────┐
│                           API ENDPOINTS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  🔹 POST /query         → RAG query with context retrieval + LLM generation │
│  🔹 GET  /ui            → Gradio web interface                              │
│  🔹 GET  /health        → System health check                               │
│  🔹 GET  /metrics       → Performance metrics                               │
│  🔹 GET  /docs          → Swagger API documentation                         │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **Indexing Flow**: Run offline via notebooks/scripts to build FAISS index and metadata
- **Query Flow**: FastAPI handles HTTP requests, orchestrates RAG pipeline (retrieve → prompt → generate)
- **FAISS Index**: Loaded into memory at startup for fast <1ms vector search
- **OpenAI API**: Used for embeddings (indexing + queries) and LLM generation (queries only)
- **Total Query Latency**: ~600ms (p50), ~1200ms (p95)

---

## 📁 Project Structure

```
disney_reviews/
├── app/                          # Main application package
│   ├── main.py                   # FastAPI application entry point & lifespan
│   ├── api/
│   │   └── routes.py             # API endpoints (/query, /ui, /health, /metrics)
│   ├── services/
│   │   ├── rag_builder.py        # RAG indexing: build/load FAISS indices
│   │   └── rag_query.py          # RAG retrieval: query processing & LLM
│   ├── models/
│   │   └── schemas.py            # Pydantic data models for requests/responses
│   ├── ui/
│   │   └── gradio_interface.py   # Gradio web UI integration
│   ├── core/
│   │   └── config.py             # Configuration & settings management
│   └── utils/
│       ├── metrics.py            # Metrics collection & statistics
│       └── logging_config.py     # Structured logging configuration
├── data/
│   └── DisneylandReviews.csv     # Source dataset (42,656 reviews)
├── rag_index/                    # Persisted vector indices & metadata
│   ├── faiss_*.index             # FAISS vector indices (by sample size)
│   ├── meta_*.jsonl              # Metadata files (by sample size)
│   └── embeddings_*.npy          # Cached embeddings (optional)
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── analyze_dysney_reviews.ipynb
│   ├── rag_flow_query_7.ipynb
│   └── rag_flow_query_9.ipynb
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests (schemas, metrics)
│   ├── integration/              # Integration tests (API routes)
│   └── conftest.py               # Pytest fixtures & configuration
├── logs/                         # Application logs (auto-generated)
├── .env                          # Environment variables (API keys)
├── pyproject.toml                # Project dependencies (uv/pip)
├── requirements.txt              # Pinned dependencies
└── README.md                     # This file
```


## 🧩 Main Classes & Modules

### Core Services

#### `RAGBuilder` (`app/services/rag_builder.py`)
Handles the **indexing phase** of the RAG pipeline:
- **`load_data()`**: Loads review CSV and samples N reviews
- **`chunk_texts()`**: Splits reviews into 500-token chunks with overlap
- **`get_embeddings_batch()`**: Generates embeddings via OpenAI API (with caching)
- **`build_faiss_index()`**: Creates FAISS IndexFlatL2 and saves to disk
- **`build_or_load()`**: Smart loader—builds if missing, else loads from cache
- **Purpose**: Prepares the searchable knowledge base

#### `RAGQueryHandler` (`app/services/rag_query.py`)
Handles the **retrieval phase** of the RAG pipeline:
- **`retrieve_context()`**: Performs FAISS similarity search for query
- **`build_prompt()`**: Constructs LLM prompt with retrieved context
- **`generate_answer()`**: Calls OpenAI GPT-4 to generate grounded answer
- **`query()`**: Orchestrates full query pipeline (retrieve → prompt → generate)
- **Purpose**: Answers user questions using retrieved context

### API Layer

#### `routes.py` (`app/api/routes.py`)
Defines FastAPI endpoints:
- **`POST /query`**: Main RAG query endpoint (accepts QueryRequest, returns QueryResponse)
- **`GET /ui`**: Gradio web interface for interactive queries
- **`GET /health`**: Health check (returns index status, vector count)
- **`GET /metrics`**: System metrics (throughput, latency, retrieval quality, costs)
- **`GET /`**: Root endpoint (API welcome message)
- **Purpose**: Exposes RAG system via REST API and web UI

#### `main.py` (`app/main.py`)
Application entry point:
- **Lifespan management**: Initializes RAGBuilder and RAGQueryHandler on startup
- **CORS middleware**: Enables cross-origin requests
- **Gradio integration**: Mounts Gradio UI at `/ui`
- **Dependency injection**: Provides query handler to routes
- **Purpose**: Orchestrates application lifecycle

### Data Models

#### `schemas.py` (`app/models/schemas.py`)
Pydantic models for type safety and validation:
- **`QueryRequest`**: User query input (query, k, temperature, model)
- **`RetrievalResult`**: Single search result (rank, distance, branch, rating, snippet)
- **`QueryResponse`**: Complete response (answer + retrieval results + metadata)
- **`HealthResponse`**: Health check response (status, index info)
- **Purpose**: Ensures data consistency and auto-generates API docs

### UI

#### `gradio_interface.py` (`app/ui/gradio_interface.py`)
Web interface for non-technical users:
- **Interactive chat interface**: Text input for queries
- **Parameter controls**: Sliders for k and temperature
- **Results display**: Answer + source citations with metadata
- **Purpose**: Makes RAG system accessible via web UI

### Utilities

#### `metrics.py` (`app/utils/metrics.py`)
Performance tracking:
- **`MetricsCollector`**: Singleton class for collecting metrics
- **Tracks**: Request counts, latency, retrieval distances, model usage, costs
- **Methods**: `record_request()`, `get_stats()`, `get_detailed_stats()`
- **Purpose**: Monitors system health and quality

#### `config.py` (`app/core/config.py`)
Centralized configuration:
- **Settings class**: Pydantic BaseSettings with environment variable loading
- **Parameters**: OpenAI key, data paths, index paths, model settings
- **Defaults**: NUM_SAMPLES=1000, EMBED_MODEL="text-embedding-3-small"
- **Purpose**: Single source of truth for configuration

#### `logging_config.py` (`app/utils/logging_config.py`)
Structured logging:
- **File logging**: Rotating logs in `logs/` directory
- **Console logging**: Colored output for development
- **Log format**: Timestamp, level, module, message
- **Purpose**: Facilitates debugging and monitoring

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only

# Generate coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

**Test Coverage:**
- **Unit tests** (`tests/unit/`): Schemas, metrics, utilities
- **Integration tests** (`tests/integration/`): API endpoints, RAG flow
- **Fixtures** (`tests/conftest.py`): Mock dependencies for isolated testing

---

## 📊 Key Metrics

Access at: `http://localhost:8000/metrics`

```json
{
  "metrics": {
    "system": {
      "index_status": "loaded",
      "total_vectors": 45610
    },
    "throughput": {
      "total_requests": 150,
      "successful_requests": 148
    },
    "latency": {
      "average_seconds": 0.623,
      "p50_seconds": 0.580,
      "p95_seconds": 1.234
    },
    "retrieval_quality": {
      "average_distance": 0.387,
      "poor_retrieval_rate": 0.02
    }
  }
}
```

---

## 🛠️ Project Configuration

Edit `.env` file to customize:

```bash
# OpenAI API

# Dataset size (pre-built indices available)
NUM_SAMPLES=1000  # Options: 100, 200, 300, 500, 1000, 5000, 10000, 50000, 100000

# Embedding model
EMBED_MODEL=text-embedding-3-small

# LLM model
LLM_MODEL=gpt-4o-mini

# Chunking parameters
MAX_TOKENS=500
OVERLAP=50

# Query defaults
DEFAULT_K=5
DEFAULT_TEMPERATURE=0.2
```
- **[API_README.md](./API_README.md)** - Detailed API documentation with endpoints, request/response schemas, and usage examples

---

## 🎯 Example Queries

```python
# Customer insights
"What do visitors like about Disneyland Paris?"

# Comparative analysis
"How does food quality compare between the three parks?"

# Sentiment analysis
"What are common complaints about Disneyland Hong Kong?"

# Feature discovery
"Which park has the best attractions for young children?"
```

