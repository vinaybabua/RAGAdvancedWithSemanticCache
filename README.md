# Symposium RAG Bot — LiteLLM Gateway Edition

The same RAG + semantic-cache chatbot as `RAG_Semantic_Cache_Symposium_Bot`, but **every LLM and embedding call now goes through a LiteLLM gateway** (an OpenAI-compatible proxy) instead of calling Anthropic and Hugging Face directly.

```
Streamlit ──► FastAPI backend ──► BetterDB semantic cache (Valkey)
                    │
                    ├─ embeddings ─┐
                    └─ chat ───────┤
                                   ▼
                         LiteLLM Gateway :4000
                    ┌──────────────┴──────────────┐
          "hf-embeddings" alias           "claude-chat" alias
     HF Inference API (all-MiniLM-L6-v2)   Anthropic claude-sonnet-4-5
```

## What changed vs. the original project

| | Original | This project |
|---|---|---|
| Embeddings | `HuggingFaceEmbeddings` (local sentence-transformers, torch) | `OpenAIEmbeddings` → gateway alias `hf-embeddings` |
| LLM | `ChatAnthropic` | `ChatOpenAI` → gateway alias `claude-chat` |
| Provider keys | In the app's `.env` | Only on the gateway |
| Swapping models | Code change | Edit `gateway/litellm_config.yaml` |
| Dependencies | `langchain-anthropic`, `langchain-huggingface`, `sentence-transformers` | `langchain-openai` (no torch needed) |

New / changed files:

- `llm_gateway.py` — one place that builds the gateway chat + embedding clients
- `gateway/litellm_config.yaml` — gateway model aliases, provider keys, alternatives
- `phase1_ingestion/ingest.py`, `phase2_retrieval/backend.py`, `demo_embeddings.py` — use `llm_gateway.py`
- `scripts/test_gateway.py` — smoke test for the gateway
- `/health` now also reports `litellm_gateway: true/false`
- The LLM call is now `await llm.ainvoke(...)` so it doesn't block the async server

Unchanged: frontend, chunking, FAISS, BetterDB semantic cache, Valkey / ElastiCache IAM auth.

## Project structure

```
RAG_Semantic_Cache_Symposium_Bot_LiteLLM/
├── data/symposium.pdf
├── gateway/litellm_config.yaml     # LiteLLM gateway config
├── llm_gateway.py                  # Gateway clients used by the app
├── phase1_ingestion/ingest.py
├── phase2_retrieval/
│   ├── backend.py
│   └── frontend.py
├── scripts/test_gateway.py
├── demo_embeddings.py
├── requirements.txt                # App
├── requirements-gateway.txt        # LiteLLM proxy (separate venv)
└── .env.example
```

## Setup (Windows)

### 1. Create `.env`

```powershell
copy .env.example .env
```

Generate a gateway key (LiteLLM **refuses to start** with weak keys like `sk-1234`):

```powershell
python -c "import secrets; print('sk-' + secrets.token_hex(32))"
```

Put that value in **both** `LITELLM_MASTER_KEY` and `LITELLM_API_KEY`, then fill in:

- `ANTHROPIC_API_KEY` — your Claude key
- `HF_TOKEN` — a Hugging Face access token (read) from https://huggingface.co/settings/tokens

### 2. Install the app

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Start the LiteLLM gateway (Terminal 1)

**Option A — pip (own venv, keeps proxy deps separate from the app):**

```powershell
python -m venv .venv-gateway
.\.venv-gateway\Scripts\python.exe -m pip install -r requirements-gateway.txt
.\.venv-gateway\Scripts\litellm.exe --config gateway\litellm_config.yaml --port 4000
```

LiteLLM reads the keys from your environment. If they aren't picked up, load `.env` first, e.g. in PowerShell:

```powershell
Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object { $k,$v = $_ -split '=',2; Set-Item "env:$($k.Trim())" $v.Trim() }
```

**Option B — Docker:**

```powershell
docker run -d --name litellm-gateway -p 4000:4000 --env-file .env `
  -v ${PWD}\gateway\litellm_config.yaml:/app/config.yaml `
  ghcr.io/berriai/litellm:main-stable --config /app/config.yaml --port 4000
```

Check it: http://localhost:4000/health/liveliness → `"I'm alive!"`

### 4. Start Valkey (same as before)

```powershell
docker run -d --name valkey-semantic-cache -p 6379:6379 valkey/valkey-bundle:latest
```

### 5. Test the gateway

```powershell
.\.venv\Scripts\python.exe scripts\test_gateway.py
```

You should see 384-dimensional vectors and a reply from Claude.

### 6. Ingest (required — the old `vector_store/` isn't reused)

```powershell
.\.venv\Scripts\python.exe phase1_ingestion\ingest.py
```

### 7. Run backend and frontend

```powershell
# Terminal 2
.\.venv\Scripts\python.exe -m uvicorn phase2_retrieval.backend:app --reload
# Terminal 3
.\.venv\Scripts\python.exe -m streamlit run phase2_retrieval\frontend.py
```

## Swapping models

Edit `gateway/litellm_config.yaml` and restart the gateway — no app code changes.

- **Chat:** change `model: anthropic/claude-sonnet-4-5` (e.g. `anthropic/claude-haiku-4-5`, `openai/gpt-4o`, `bedrock/...`). Update `CACHE_COST_MODEL` in `.env` so cost-saving stats stay accurate.
- **Embeddings:** the config includes ready-to-uncomment alternatives — local Hugging Face TEI (same model, no token), OpenAI, Ollama. **After changing the embedding model, re-run ingestion and use a new `SEMANTIC_CACHE_NAME`** — vectors from different models can't be mixed.

### Why the embedding config looks unusual

LiteLLM treats any Hugging Face model whose name contains `sentence-transformers` as a *sentence-similarity* request and returns similarity scores instead of vectors. The config works around this by giving the model a name without that prefix, pointing `api_base` at the real model's feature-extraction endpoint, and setting `input_type: feature-extraction`. `scripts/test_gateway.py` checks that you are getting real vectors.

## Notes

- Semantic cache threshold: `0.15` cosine distance (unchanged)
- `SEMANTIC_CACHE_NAME` defaults to `betterdbscache_litellm` so it never mixes with the original project's cache entries
- `/stats` and the console stats work as before
- Gateway extras you can enable later in `litellm_config.yaml`: fallbacks, retries, virtual keys with budgets, spend tracking, rate limits
