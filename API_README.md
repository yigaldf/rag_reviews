# Disneyland Reviews RAG API

A FastAPI application that provides a Retrieval-Augmented Generation (RAG) system for querying Disneyland reviews. The API uses OpenAI embeddings and FAISS for semantic search, combined with GPT models for generating natural language answers.

## Features

- 🔍 Semantic search across Disneyland reviews using FAISS vector search
- 🤖 Natural language answers generated using OpenAI GPT models
- 📊 Retrieves relevant context from 50,000+ reviews
- 🚀 Fast and efficient with pre-built FAISS index
- 📝 Comprehensive logging for debugging and monitoring
- 🔧 Configurable parameters (k, temperature, model)

## Architecture

```
app/
├── main.py              # FastAPI application entry point
├── core/
│   └── config.py        # Configuration and settings
├── models/
│   └── schemas.py       # Pydantic models for requests/responses
├── services/
│   └── rag_service.py   # RAG business logic
├── api/
│   └── routes.py        # API endpoint handlers
└── utils/
    └── logging_config.py # Logging configuration
```

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Pre-built FAISS index and metadata files (in `rag_index/` directory)

### Setup

1. **Clone the repository** (if not already done)

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Ensure FAISS index files exist**:
   Make sure you have the following files in the `rag_index/` directory:
   - `faiss_50000.index`
   - `meta_50000.jsonl`

## Running the API

### Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Python directly

```bash
python -m app.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
```http
GET /
```

Returns basic API information.

**Response:**
```json
{
  "message": "Disneyland Reviews RAG API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

### 2. Health Check
```http
GET /health
```

Returns the health status of the API and RAG service.

**Response:**
```json
{
  "status": "healthy",
  "app_name": "Disneyland Reviews RAG API",
  "version": "1.0.0",
  "index_loaded": true,
  "total_vectors": 45610
}
```

### 3. Query Reviews (Main Endpoint)
```http
POST /query
```

Query the Disneyland reviews using natural language.

**Request Body:**
```json
{
  "query": "What do visitors like about Disneyland Hong Kong?",
  "k": 5,
  "temperature": 0.2,
  "model": "gpt-4o-mini"
}
```

**Parameters:**
- `query` (string, required): The question to ask about Disneyland reviews
- `k` (integer, optional): Number of top results to retrieve (default: 5, range: 1-50)
- `temperature` (float, optional): Temperature for LLM response generation (default: 0.2, range: 0.0-2.0)
- `model` (string, optional): OpenAI model to use (default: "gpt-4o-mini", options: "gpt-4o-mini", "gpt-4o")

**Response:**
```json
{
  "query": "What do visitors like about Disneyland Hong Kong?",
  "answer": "Visitors to Disneyland Hong Kong appreciate several aspects of the park:\n\n1. **Magical Atmosphere**: Many reviews describe the experience as magical and mystical, creating a sense of wonder for guests.\n2. **Accessibility**: The park is noted for being easily accessible via the MTR, making it convenient for visitors to reach...",
  "k": 5,
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "retrieval_results": [
    {
      "rank": 1,
      "distance": 0.4757,
      "branch": "Disneyland_HongKong",
      "rating": 5.0,
      "reviewer_location": "Australia",
      "snippet": "Amazing experience with beautiful scenery..."
    }
  ]
}
```

## Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What do visitors like about Disneyland Hong Kong?",
    "k": 5,
    "temperature": 0.2,
    "model": "gpt-4o-mini"
  }'
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/query"
payload = {
    "query": "Is spring a good time to visit Disneyland?",
    "k": 10,
    "temperature": 0.2,
    "model": "gpt-4o-mini"
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Answer: {result['answer']}")
print(f"\nTop {len(result['retrieval_results'])} results:")
for r in result['retrieval_results']:
    print(f"  {r['rank']}. {r['branch']} (Rating: {r['rating']}/5)")
```

### Using JavaScript/TypeScript

```javascript
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Is the staff in Paris friendly?',
    k: 5,
    temperature: 0.2,
    model: 'gpt-4o-mini'
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
console.log('Retrieval results:', data.retrieval_results);
```

## Configuration

Configuration is managed through environment variables and the `app/core/config.py` file.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `LOG_LEVEL` | Logging level | INFO |
| `LOG_TO_FILE` | Enable file logging | true |

### Application Settings

You can modify settings in `app/core/config.py`:

- `num_samples`: Number of samples in the index (default: 50000)
- `embed_model`: OpenAI embedding model (default: "text-embedding-3-small")
- `llm_model`: Default LLM model (default: "gpt-4o-mini")
- `temperature`: Default temperature (default: 0.2)
- `default_k`: Default number of results (default: 5)

## Sample Queries

Here are some example queries you can try:

1. **Location-specific questions:**
   - "What do visitors from Australia say about Disneyland in Hong Kong?"
   - "How is Disneyland Paris different from California?"

2. **Timing and crowds:**
   - "Is spring a good time to visit Disneyland?"
   - "Is Disneyland California usually crowded in June?"

3. **Staff and service:**
   - "Is the staff in Paris friendly?"
   - "How is the customer service at Hong Kong Disneyland?"

4. **Attractions and experiences:**
   - "What are the best rides at Disneyland California?"
   - "What do families with young children enjoy most?"

5. **Food and dining:**
   - "What do visitors say about the food at Disneyland?"
   - "Are there good vegetarian options?"

## Logging

The application provides comprehensive logging:

- **Console logs**: INFO level and above
- **File logs**: DEBUG level and above (saved to `logs/rag_YYYYMMDD_HHMMSS.log`)

Logs include:
- Query processing details
- Retrieval results and distances
- LLM generation times
- Error traces for debugging

## Error Handling

The API provides clear error messages:

- **503 Service Unavailable**: RAG service not initialized
- **500 Internal Server Error**: Error processing query
- **422 Unprocessable Entity**: Invalid request parameters

## Performance

- **Embedding generation**: ~0.3-0.5s per query
- **FAISS search**: ~0.001-0.01s (very fast)
- **LLM generation**: ~2-7s depending on model and response length
- **Total response time**: ~3-8s for typical queries

## Development

### Project Structure

The project follows a clean architecture pattern:

- **Core**: Configuration and settings
- **Models**: Data models and schemas (Pydantic)
- **Services**: Business logic (RAG operations)
- **API**: HTTP endpoint handlers
- **Utils**: Utility functions (logging, etc.)

### Adding New Endpoints

1. Define Pydantic models in `app/models/schemas.py`
2. Add route handler in `app/api/routes.py`
3. Implement business logic in `app/services/rag_service.py` if needed

### Testing

You can test the API using the interactive documentation at `/docs` or create automated tests using pytest.

## Production Deployment

### Docker (Recommended)

You can containerize the application:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

COPY app/ ./app/
COPY data/ ./data/
COPY rag_index/ ./rag_index/
COPY .env .env

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms

- **AWS**: Deploy using EC2, ECS, or Lambda
- **Google Cloud**: Deploy using Cloud Run or Compute Engine
- **Azure**: Deploy using App Service or Container Instances
- **Heroku**: Use Procfile with uvicorn

### Performance Optimization

For production:
1. Use multiple workers: `--workers 4`
2. Enable caching for embeddings
3. Use a reverse proxy (nginx)
4. Implement rate limiting
5. Monitor with tools like Prometheus + Grafana

## Troubleshooting

### Common Issues

1. **"Index file not found"**
   - Ensure `rag_index/faiss_50000.index` and `rag_index/meta_50000.jsonl` exist
   - Check file permissions

2. **"OPENAI_API_KEY not set"**
   - Verify `.env` file exists and contains valid API key
   - Check environment variable is loaded

3. **Slow response times**
   - Check network connection to OpenAI API
   - Consider using faster models (gpt-4o-mini instead of gpt-4o)
   - Reduce `k` value for faster retrieval

## License

[Your License Here]

## Support

For issues and questions:
- Check the logs in `logs/` directory
- Review the interactive API documentation at `/docs`
- Open an issue on GitHub (if applicable)

