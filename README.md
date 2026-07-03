# SHL Assessment Recommender

A stateless FastAPI service that provides conversational AI for SHL assessment selection. Given a hiring context through natural conversation, the agent recommends appropriate assessments from SHL's 377-item catalog using semantic retrieval + LLM generation.

## API

### `GET /health`
Returns `{"status": "ok"}` immediately. No model calls.

### `POST /chat`
```json
Request:
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer..."},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years"}
  ]
}

Response:
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Architecture

| Component | Description |
|-----------|-------------|
| **Retrieval** | FAISS flat index (cosine similarity) on `sentence-transformers/all-MiniLM-L6-v2` embeddings of 377 catalog items |
| **Intent Extraction** | Single Groq `llama-3.3-70b-versatile` call classifies intent + extracts slots from full conversation history |
| **Generation** | Second LLM call produces reply + shortlist from retrieved catalog context |
| **Post-Filter** | Anti-hallucination: all LLM-proposed URLs replaced with catalog ground truth; non-matches dropped |
| **LLM Fallback** | Groq 429/timeout → immediate Gemini Flash fallback (no backoff) |

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env and fill in your GROQ_API_KEY and GEMINI_API_KEY
```

Get free API keys:
- Groq: https://console.groq.com/keys
- Gemini: https://aistudio.google.com/apikey

### 3. Build the FAISS Index (one-time)
```bash
python scripts/build_index.py
```
This creates `data/catalog_index.faiss` and `data/catalog_metadata.json`.

### 4. Run Locally
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test
```bash
# With server running in another terminal:
pytest tests/ -v
```

## Docker

```bash
# Build (must have run build_index.py first)
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 --env-file .env shl-recommender
```

## Deployment (Render)

1. Push to a GitHub repository
2. Create a new **Web Service** on [Render](https://render.com)
3. Select **Docker** as environment
4. Set environment variables: `GROQ_API_KEY`, `GEMINI_API_KEY`
5. Deploy

> **Note**: Free tier spins down after 15 min inactivity. Cold start takes 30–60s but models/index are preloaded, so the first `/chat` after warmup is fast.

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py              # FastAPI app, /health, /chat
│   ├── schemas.py           # Pydantic models
│   ├── config.py            # Env vars, constants, paths
│   ├── llm_client.py        # Groq + Gemini fallback
│   ├── intent_extraction.py # Slot/intent extraction LLM call
│   ├── generation.py        # Reply + shortlist generation LLM call
│   ├── retrieval.py         # FAISS search + filters + ranking
│   ├── catalog_index.py     # Index wrapper + name lookup
│   ├── postfilter.py        # Anti-hallucination post-filter
│   └── prompts/             # system_prompt.txt, extraction_prompt.txt, generation_prompt.txt
├── data/
│   ├── shl_product_catalogue.json
│   ├── catalog_index.faiss  (built by scripts/build_index.py)
│   └── catalog_metadata.json
├── scripts/
│   └── build_index.py       # One-time offline index build
├── tests/
│   ├── test_schema_compliance.py
│   ├── test_hallucination_check.py
│   └── test_behavior_probes.py
├── Dockerfile
├── requirements.txt
└── .env.example
```
