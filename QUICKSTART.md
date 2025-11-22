# Quick Start Guide

Get the Disneyland Reviews RAG API up and running in minutes!

## Prerequisites

- Installed UV python package mamager
- OpenAI API key
- DisneylandReviews.csv file

## Setup (First Time Only)

1. **Install dependencies:**
  create virual venv
   ```bash
   uv venv 
   uv sync
   uv sync --group test	
  ```
  activate the virtual env 
   ```bash
   source .venv/bin/activate
   ```
   

2. **Configure environment:**
   Make sure your `.env` file contains:
   ```
   OPENAI_API_KEY=sk-proj-your-api-key-here
   ```

   you can reset the config file parameters under ./app/core/config.py - not mnadatory 

3.copy the DisneylandReviews.csv under data folder


## Notebooks Overview

To gain a clear understanding of the project’s design and workflow, open and run the Jupyter notebooks located in the ./notebooks folder:

- analyze_disney_reviews.ipynb → Data exploration and analysis

- rag_flow_query_7.ipynb → Complete RAG pipeline (build + query)

- rag_flow_query_9.ipynb → Advanced RAG flow with logging and monitoring

These notebooks demonstrate:

**Creating the RAG Flow (Indexing):**
- Loading and chunking review data
- Generating embeddings via OpenAI API
- Building FAISS indices from scratch
- Saving indices and metadata to disk

**Running the Retrieval Query Flow:**
- Embedding user queries
- Performing vector similarity search
- Retrieving relevant review chunks
- Generating LLM responses with context

## Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Accessing the API

Once running, access:
- **Interactive API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **ui**: http://localhost:8000/ui 
- **query**: http://localhost:8000/query
- **monitoring** http://localhost:8000/metrics 
- **Health Check**: http://localhost:8000/health



## Testing the API

### Option 1: Use the interactive docs
1. Go to http://localhost:8000/docs
2. Click on "POST /query"
3. Click "Try it out"
4. Enter your query and parameters
5. Click "Execute"


### Option 2: Use the test script
```bash
pytest 
```

### Option 3: Use cURL
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

## Example Queries

Try these queries to see the API in action:

```python
import requests

queries = [
    "What do visitors like about Disneyland Hong Kong?",
    "Is spring a good time to visit Disneyland?",
    "Is the staff in Paris friendly?",
    "What do families with young children enjoy most?",
]

for query in queries:
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": query, "k": 5}
    )
    print(f"Q: {query}")
    print(f"A: {response.json()['answer']}\n")
```

## Project Structure

```
disney_reviews/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── api/               # API endpoints
│   ├── services/          # Business logic (RAG)
│   ├── models/            # Pydantic schemas
│   ├── core/              # Configuration
│   └── utils/             # Utilities (logging)
├── data/                   # Review data CSV files
├── rag_index/             # FAISS index and metadata
├── logs/                  # Application logs
├── API_README.md          # Comprehensive API documentation
├── QUICKSTART.md          # This file
├── run_api.sh            # Run script
└── test_api.py           # Test script
```

## Configuration

Edit `app/core/config.py` to customize:
- Default model (`llm_model`)
- Default temperature
- Default k value
- Embedding model
- Log level

## Troubleshooting

### API won't start
- Check that port 8000 is available
- Verify OPENAI_API_KEY is set in .env
- Ensure FAISS index files exist in rag_index/

### Slow responses
- Use `gpt-4o-mini` instead of `gpt-4o` (faster and cheaper)
- Reduce `k` value (fewer documents to retrieve)
- Check internet connection to OpenAI API

### No relevant results
- Try rephrasing your query
- Increase `k` value to retrieve more documents
- Check that the query is related to Disneyland reviews

## Next Steps

- Read the [comprehensive API documentation](API_README.md)
- Customize the configuration in `app/core/config.py`
- Add authentication/authorization if deploying publicly
- Implement caching for frequently asked questions
- Set up monitoring and logging

## Support

Check the logs in `logs/` directory for detailed debugging information.

