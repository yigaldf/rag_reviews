# System Design & Architecture
## Disney Reviews RAG System

---

## 1. RAG Solution Overview

**The Challenge**: Enable customer experience teams to query 42,000+ Disneyland reviews using natural language and receive accurate, source-grounded answers.

**The Solution**: A Retrieval-Augmented Generation (RAG) system that combines vector similarity search (retrieval) with large language models (generation) to provide contextually accurate answers backed by actual customer reviews.

**One-Line RAG Flow Description**:  
*"Transform reviews into searchable vectors using tiktoken chunking and OpenAI embeddings, store in FAISS index, then retrieve relevant chunks for any query and feed them to GPT-4 to generate grounded answers."*

---

## 2. RAG Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG SYSTEM ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│   PHASE 1: INDEXING (Offline)  │      │  PHASE 2: RETRIEVAL (Runtime)   │
│         Build Once              │      │      Query Many Times           │
└─────────────────────────────────┘      └─────────────────────────────────┘

┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│                                 │      │                                 │
│  📄 Disney Reviews CSV          │      │  💬 User Query                  │
│  42,656 reviews                 │      │  "What do visitors say          │
│  3 locations (CA, Paris, HK)    │      │   about Hong Kong park?"        │
│                                 │      │                                 │
└────────────┬────────────────────┘      └────────────┬────────────────────┘
             │                                        │
             ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  ✂️ TEXT CHUNKING               │      │  🔢 QUERY EMBEDDING             │
│  Tool: tiktoken                 │      │  Tool: OpenAI API               │
│  • cl100k_base encoding         │      │  • text-embedding-3-small       │
│  • 500 tokens per chunk         │      │  • Same model as indexing       │
│  • 50 token overlap             │      │  • Output: 1536-dim vector      │
│  Output: ~45,610 chunks         │      │  Latency: ~80ms                 │
└────────────┬────────────────────┘      └────────────┬────────────────────┘
             │                                        │
             ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  🧮 EMBEDDING GENERATION        │      │  🔍 VECTOR SEARCH               │
│  Tool: OpenAI API               │      │  Tool: FAISS                    │
│  • text-embedding-3-small       │      │  • IndexFlatL2 (L2 distance)    │
│  • Batch: 128-1800 chunks       │      │  • Search 45K+ vectors          │
│  • Output: [N × 1536] matrix    │      │  • Return top-K nearest         │
│  • Cache: embeddings_N.npy      │      │  • K=5 (default)                │
│  Time: ~2-10 min (one-time)     │      │  Latency: <1ms                  │
└────────────┬────────────────────┘      └────────────┬────────────────────┘
             │                                        │
             ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  💾 FAISS INDEX BUILD           │      │  📋 METADATA RETRIEVAL          │
│  Tool: FAISS (Facebook AI)      │      │  Tool: JSONL file               │
│  • IndexFlatL2 creation         │      │  • Load metadata for top-K      │
│  • Add all embeddings           │      │  • Branch, rating, location     │
│  • Save: faiss_N.index          │      │  • Review text chunks           │
│  • Size: ~265 MB (45K vectors)  │      │  Latency: <1ms                  │
│  Build time: <1 second          │      │                                 │
└────────────┬────────────────────┘      └────────────┬────────────────────┘
             │                                        │
             ▼                                        ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  📝 METADATA STORAGE            │      │  📝 PROMPT CONSTRUCTION         │
│  Format: JSONL                  │      │  Tool: Python string formatting │
│  • One JSON per line            │      │  • System instructions          │
│  • Aligned with FAISS index     │      │  • User query insertion         │
│  • Fields: review_id, branch,   │      │  • Retrieved context (top-K)    │
│    rating, location, chunk_text │      │  • Grounding rules              │
│  • Save: meta_N.jsonl           │      │  Latency: <1ms                  │
└─────────────────────────────────┘      └────────────┬────────────────────┘
                                                      │
   ┌──────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  🤖 LLM GENERATION                                                           │
│  Tool: OpenAI Chat Completions API                                          │
│  • Model: gpt-4o-mini                                                       │
│  • Temperature: 0.2 (low for consistency)                                   │
│  • Input: Prompt + Context (top-K chunks)                                   │
│  • Output: Grounded answer                                                  │
│  Latency: ~500ms                                                            │
│  Cost: ~$0.0001 per query                                                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │  ✅ FINAL RESPONSE   │
                      │  • Answer text       │
                      │  • Source citations  │
                      │  • Metadata          │
                      └─────────────────────┘
```

---

## 3. FastAPI Server Flow Diagram

This diagram shows how the FastAPI server orchestrates the RAG pipeline for incoming requests.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVER REQUEST FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

                           ┌──────────────────┐
                           │   CLIENT         │
                           │  (Browser/API)   │
                           └────────┬─────────┘
                                    │ HTTP POST /query
                                    │ {"query": "...", "k": 5}
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FASTAPI SERVER                                    │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  1. API ROUTE HANDLER (/query)                                     │    │
│  │     app/api/routes.py                                              │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Receive HTTP request                                            │    │
│  │  • Parse JSON body                                                 │    │
│  │  • Extract: query, k, temperature, model                           │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  2. REQUEST VALIDATION                                             │    │
│  │     Pydantic Schema (QueryRequest)                                 │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Validate query: string, max 500 chars                           │    │
│  │  • Validate k: int, range 1-20, default=5                          │    │
│  │  • Validate temperature: float, range 0.0-2.0, default=0.2         │    │
│  │  • Validate model: str, default="gpt-4o-mini"                      │    │
│  │  ❌ If invalid → Return 422 Error                                  │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  3. METRICS TRACKING (Start)                                       │    │
│  │     app/utils/metrics.py                                           │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Start latency timer                                             │    │
│  │  • Log request metadata                                            │    │
│  │  • Record request count                                            │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  4. DEPENDENCY INJECTION                                           │    │
│  │     FastAPI Depends()                                              │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Inject: query_handler (RAGQueryHandler)                         │    │
│  │  • Inject: settings (Config)                                       │    │
│  │  • Inject: metrics (MetricsCollector)                              │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
└───────────────────────────┼──────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RAG QUERY HANDLER                                        │
│                     app/services/rag_query.py                                │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  5. RETRIEVE CONTEXT (retrieve_context)                            │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  Step 5a: Embed Query                                              │    │
│  │    • Call OpenAI Embeddings API                                    │    │
│  │    • Model: text-embedding-3-small                                 │    │
│  │    • Output: [1 × 1536] vector                                     │    │
│  │    • Time: ~80ms                                                   │    │
│  │                                                                     │    │
│  │  Step 5b: Search FAISS Index                                       │    │
│  │    • Load: self.index (FAISS IndexFlatL2)                          │    │
│  │    • Search: index.search(query_vector, k)                         │    │
│  │    • Output: indices=[10, 25, 42], distances=[0.23, 0.31, 0.42]    │    │
│  │    • Time: <1ms                                                    │    │
│  │                                                                     │    │
│  │  Step 5c: Load Metadata                                            │    │
│  │    • Load: self.metadata (from JSONL)                              │    │
│  │    • Extract: metadata[indices]                                    │    │
│  │    • Output: List of review chunks with branch, rating, etc.       │    │
│  │    • Time: <1ms                                                    │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  6. BUILD PROMPT (build_prompt)                                    │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Template: System instruction                                    │    │
│  │  • Insert: User query                                              │    │
│  │  • Insert: Retrieved context (top-K chunks)                        │    │
│  │  • Add: Grounding instructions                                     │    │
│  │  • Output: Complete prompt string (~800-1200 tokens)               │    │
│  │  • Time: <1ms                                                      │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  7. GENERATE ANSWER (generate_answer)                              │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Call OpenAI Chat Completions API                                │    │
│  │  • Model: gpt-4o-mini (or user-specified)                          │    │
│  │  • Temperature: 0.2 (or user-specified)                            │    │
│  │  • Input: Prompt with context                                      │    │
│  │  • Output: Generated answer text                                   │    │
│  │  • Time: ~500ms                                                    │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  8. PACKAGE RESPONSE (query)                                       │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Combine: answer + retrieval_results                             │    │
│  │  • Add metadata: k, model, temperature                             │    │
│  │  • Format: QueryResponse (Pydantic model)                          │    │
│  │  • Time: <1ms                                                      │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
└───────────────────────────┼──────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACK TO FASTAPI SERVER                                   │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  9. METRICS TRACKING (End)                                         │    │
│  │     app/utils/metrics.py                                           │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Stop latency timer                                              │    │
│  │  • Calculate total latency: ~600ms                                 │    │
│  │  • Record retrieval distance avg: 0.32                             │    │
│  │  • Update metrics: success count, latency histogram                │    │
│  │  • Log completion                                                  │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  10. RESPONSE SERIALIZATION                                        │    │
│  │      Pydantic → JSON                                               │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │  • Convert QueryResponse to JSON                                   │    │
│  │  • Add HTTP headers (Content-Type: application/json)               │    │
│  │  • Add CORS headers (if enabled)                                   │    │
│  │  • Time: <1ms                                                      │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                  │
└───────────────────────────┼──────────────────────────────────────────────────┘
                            │ HTTP 200 OK
                            │ JSON Response
                            ▼
                     ┌──────────────────┐
                     │   CLIENT         │
                     │  (Browser/API)   │
                     └──────────────────┘


### Key FastAPI Features Used

| Feature | Purpose | Implementation |
|---------|---------|----------------|
| **Pydantic Models** | Request/Response validation | `QueryRequest`, `QueryResponse` schemas |
| **Dependency Injection** | Share components across routes | `Depends(get_query_handler)` |
| **Lifespan Events** | Load FAISS index on startup | `@asynccontextmanager` for app lifespan |
| **CORS Middleware** | Enable cross-origin requests | `app.add_middleware(CORSMiddleware)` |
| **Automatic Docs** | Interactive API documentation | Swagger UI at `/docs` |
| **Async/Await** | Non-blocking I/O for OpenAI calls | `async def query_endpoint()` |

### Error Handling Flow

```
Request → Validation Error (422)
       → OpenAI API Error (503)
       → FAISS Index Not Loaded (503)
       → Internal Server Error (500)
       → Success (200)
```

## 4. RAG Components Deep Dive

### 4.1 Text Chunking (tiktoken)

**Purpose**: Split long reviews into manageable, semantically coherent pieces.

**Tool**: `tiktoken` - OpenAI's fast tokenizer library

### 4.2 Embeddings (OpenAI API)

**Purpose**: Convert text chunks into high-dimensional vectors that capture semantic meaning.

**Tool**: OpenAI Embeddings API
- **Model**: `text-embedding-3-small`
- **Dimensions**: 1,536 per vector
- **Quality**: State-of-the-art semantic understanding
- **Cost**: $0.02 per 1M tokens (~$0.42 for full dataset)

**Why text-embedding-3-small**:
- ✅ Best quality-to-cost ratio
- ✅ High semantic accuracy
- ✅ Consistent with OpenAI's LLM ecosystem
- ✅ 1,536 dimensions (optimal for FAISS)


**Optimization**:
- Caching: Save to `.npy` files to avoid regenerating
- Parallel processing: Use async/threading for 4-8x speedup
- Rate limiting: Respect OpenAI's 1M tokens/minute limit

**Output**: NumPy array of shape `[45,610, 1,536]`

### 4.3 Vector Index (FAISS)

**Purpose**: Enable fast similarity search over thousands of embedding vectors.

**Tool**: FAISS (Facebook AI Similarity Search)
- **Index Type**: `IndexFlatL2` (exact L2 distance search)
- **Distance Metric**: Euclidean (L2) distance
- **Why FAISS**: Ultra-fast (<1ms), no external dependencies, handles millions of vectors

**IndexFlatL2 Characteristics**:
- ✅ **Exact search**: 100% recall accuracy
- ✅ **Fast**: <1ms for 45K vectors, <10ms for 1M vectors
- ✅ **Simple**: No training or tuning required
- ❌ **Memory**: ~265 MB for 45K vectors (6 bytes per dimension)

**Alternative Index Types** (for future scaling):
- `IndexIVFFlat`: Approximate search, 10-100x faster for millions of vectors
- `IndexHNSWFlat`: Graph-based, excellent for high-dimensional data

**Search Performance**:
```python
# Query: Find top-K most similar chunks


### 4.5 LLM Generation (OpenAI Chat Completions)

**Purpose**: Generate natural language answers grounded in retrieved context.

**Tool**: OpenAI Chat Completions API
- **Model**: `gpt-4o-mini`
- **Why gpt-4o-mini**: Best cost/performance balance ($0.15/1M input tokens vs $5/1M for GPT-4)
- **Temperature**: 0.2 (low for factual consistency)
- **Max tokens**: 500-1000 (configurable)


## 5. Complete RAG Flow (Step-by-Step)

### Phase 1: Indexing (One-time setup, ~10 minutes)

```
Step 1: Load Data
├─ Read: data/DisneylandReviews.csv
├─ Sample: 10,000 reviews (configurable)
└─ Output: DataFrame with 10,000 rows

Step 2: Chunk Text
├─ Tool: tiktoken (cl100k_base)
├─ Process: 10,000 reviews → 10,700 chunks
├─ Params: 500 tokens/chunk, 50 token overlap
└─ Output: List of 10,700 text chunks

Step 3: Generate Embeddings
├─ Tool: OpenAI text-embedding-3-small
├─ Batch: 128-1800 chunks per API call
├─ Process: 10,700 chunks → 10,700 vectors
├─ Output: [10,700 × 1,536] NumPy array
├─ Time: ~2-10 minutes
└─ Cost: ~$0.05

Step 4: Build FAISS Index
├─ Tool: FAISS IndexFlatL2
├─ Add: 10,700 vectors to index
├─ Save: rag_index/faiss_10000.index
├─ Time: <1 second
└─ Size: ~62 MB

Step 5: Save Metadata
├─ Format: JSONL
├─ Save: rag_index/meta_10000.jsonl
├─ Fields: review_id, branch, rating, location, year_month, chunk
└─ Size: ~15 MB
```

### Phase 2: Retrieval (Every query, ~600ms)

```
Step 1: Receive Query
├─ Input: "What do visitors like about Hong Kong park?"
├─ Validation: Check query length, sanitize
└─ Time: <1ms

Step 2: Embed Query
├─ Tool: OpenAI text-embedding-3-small
├─ Process: Query text → 1536-dim vector
├─ Output: [1 × 1,536] vector
├─ Time: ~80ms
└─ Cost: ~$0.000001

Step 3: Search FAISS Index
├─ Tool: FAISS IndexFlatL2
├─ Input: Query vector + k=5
├─ Process: Compare with 10,700 vectors
├─ Output: 5 nearest neighbor indices + distances
├─ Example: indices=[42, 156, 891, 1203, 3456]
│           distances=[0.23, 0.31, 0.42, 0.48, 0.52]
└─ Time: <1ms

Step 4: Retrieve Metadata
├─ Tool: JSONL file reading
├─ Load: Metadata for indices [42, 156, 891, 1203, 3456]
├─ Output: 5 review chunks with branch, rating, location
└─ Time: <1ms

Step 5: Build Prompt
├─ Template: System instruction + Query + Context
├─ Insert: User query
├─ Insert: Top-5 review chunks with metadata
├─ Output: Complete prompt (~800-1200 tokens)
└─ Time: <1ms

Step 6: Generate Answer
├─ Tool: OpenAI gpt-4o-mini
├─ Input: Prompt with context
├─ Params: temperature=0.2, max_tokens=500
├─ Process: LLM generates grounded answer
├─ Output: Answer text (~100-300 words)
├─ Time: ~500ms
└─ Cost: ~$0.0001

Step 7: Return Response
├─ Package: Answer + sources + metadata
├─ Format: JSON response
├─ Fields: query, answer, retrieval_results, model, k, temperature
└─ Time: <1ms

Total Query Latency: ~600ms (p50), ~1200ms (p95)
Total Query Cost: ~$0.0001
``

## 6. RAG Performance Metrics

### Indexing Metrics (One-time)

| Metric | Value | Notes |
|--------|-------|-------|
| **Input reviews** | 42,656 | Full dataset |
| **Output chunks** | 45,610 | After tiktoken chunking |
| **Embedding dimensions** | 1,536 | per chunk |
| **Index size** | 265 MB | FAISS + metadata |
| **Build time** | 8-12 min | With embedding generation |
| **Cost** | $0.42 | OpenAI embeddings |

### Query Metrics (Per request)

| Component | Latency | Cost | Tool |
|-----------|---------|------|------|
| **Query embedding** | ~80ms | $0.000001 | OpenAI API |
| **FAISS search** | <1ms | $0 | Local |
| **Metadata retrieval** | <1ms | $0 | Local |
| **LLM generation** | ~500ms | $0.0001 | OpenAI API |
| **Total (p50)** | **~600ms** | **$0.0001** | - |
| **Total (p95)** | **~1200ms** | **$0.0001** | - |

### Quality Metrics

| Metric | Good | Fair | Poor |
|--------|------|------|------|
| **Retrieval distance** | <0.5 | 0.5-0.7 | >0.7 |
| **Answer relevance** | High | Medium | Low |
| **Source citations** | ✅ Always | ✅ Always | ✅ Always |

---

## 7. Technology Stack Summary

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Tokenization** | tiktoken | latest | Chunk text into tokens |
| **Embeddings** | OpenAI API | text-embedding-3-small | Convert text to vectors |
| **Vector Search** | FAISS | 1.7.4+ | Fast similarity search |
| **Metadata** | JSONL | - | Store review metadata |
| **LLM** | OpenAI API | gpt-4o-mini | Generate answers |
| **API Framework** | FastAPI | 0.119+ | Serve HTTP endpoints |
| **UI** | Gradio | 4.0+ | Interactive web interface |
| **Language** | Python | 3.10+ | Implementation language |


## 8. RAG Trade-offs & Design Decisions

### Why Exact Search (IndexFlatL2) vs Approximate?

**Decision**: Use `IndexFlatL2` (exact search)

**Rationale**:
- ✅ **100% recall**: No missed relevant results
- ✅ **Simple**: No training or parameter tuning
- ✅ **Fast enough**: <1ms for 45K vectors
- ✅ **Quality first**: Accuracy more important than speed at this scale

**Trade-offs**:
- ❌ Slower for >1M vectors (but still <10ms)
- ✅ Can switch to `IndexIVFFlat` later if needed

---

### Why 500-token chunks with 50-token overlap?

**Decision**: 500 tokens per chunk, 50-token overlap

**Rationale**:
- ✅ **Semantic coherence**: 500 tokens = ~1-2 paragraphs (enough context)
- ✅ **Embedding quality**: Within sweet spot for text-embedding-3-small
- ✅ **Overlap**: Prevents losing information at boundaries
- ✅ **Retrieval precision**: Smaller chunks = more precise matching

**Trade-offs**:
- Larger chunks (1000 tokens): More context but less precise
- Smaller chunks (200 tokens): More precise but fragmented context
- More overlap (100 tokens): Better continuity but more redundancy

---

### Why gpt-4o-mini vs GPT-4?

**Decision**: Use `gpt-4o-mini` as default LLM

**Rationale**:
- ✅ **Cost**: $0.15/1M input tokens vs $5/1M for GPT-4 (33x cheaper)
- ✅ **Speed**: ~500ms vs ~1-2s for GPT-4
- ✅ **Quality**: Sufficient for factual QA with provided context
- ✅ **Flexibility**: Users can override with GPT-4 if needed

**Trade-offs**:
- GPT-4 has slightly better reasoning, but the difference is minimal for RAG tasks where context is provided

---

### Why FAISS vs Vector Databases (Pinecone, Weaviate)?

**Decision**: Use FAISS (local, in-memory)

**Rationale**:
- ✅ **No external dependencies**: Works offline, no API costs
- ✅ **Fast**: <1ms search latency (in-memory)
- ✅ **Simple deployment**: Just load index file
- ✅ **Cost**: Free (vs $70-100/month for managed vector DBs)
- ✅ **Proven**: Battle-tested by Facebook AI Research

**Trade-offs**:
- Vector DBs offer features like filtering, updates, distributed search
- For 45K-1M vectors, FAISS is optimal
- Can migrate to vector DB if scale requires it (>10M vectors)

---

**Document Version**: 1.0  
**Last Updated**: October 29, 2025  
**Focus**: RAG Solution Architecture

