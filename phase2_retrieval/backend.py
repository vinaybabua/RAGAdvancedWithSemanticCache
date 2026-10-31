"""
Phase 2: Backend API with BetterDB Semantic Caching

All LLM and embedding calls go through the LiteLLM gateway
(see llm_gateway.py and gateway/litellm_config.yaml).

Run from the project root:
    python -m uvicorn phase2_retrieval.backend:app --reload
"""

import os
import sys
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import valkey.asyncio as valkey_async
from betterdb_semantic_cache import SemanticCache, SemanticCacheOptions
from betterdb_semantic_cache.types import CacheStoreOptions

import botocore.session
from botocore.model import ServiceId
from botocore.signers import RequestSigner
from valkey.credentials import CredentialProvider

# Make the project root importable (llm_gateway.py lives there).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_gateway import (  # noqa: E402
    EMBEDDING_MODEL,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LLM_MODEL,
    get_embeddings,
    get_llm,
)

load_dotenv()

# The gateway alias (e.g. "claude-chat") is not a name BetterDB's cost table
# knows, so cost-savings tracking uses the real underlying model name.
CACHE_COST_MODEL = os.getenv("CACHE_COST_MODEL", "claude-sonnet-4-5")

# ============================================================================
# IAM AUTH TOKEN GENERATION (for AWS ElastiCache Serverless public endpoints)
# ============================================================================


def generate_iam_auth_token(user_id: str, cache_name: str, region: str) -> str:
    """
    Builds a SigV4-signed, short-lived (15 min) auth token for an
    ElastiCache public endpoint, following AWS's documented pattern for
    IAM-authenticated connections.
    """
    session = botocore.session.get_session()
    signer = RequestSigner(
        ServiceId("elasticache"),
        region,
        "elasticache",
        "v4",
        session.get_credentials(),
        session.get_component("event_emitter"),
    )
    url = f"https://{cache_name}/?Action=connect&User={user_id}"
    presigned_url = signer.generate_presigned_url(
        {"method": "GET", "url": url, "body": {}, "headers": {}, "context": {}},
        region_name=region,
        expires_in=900,
        operation_name="",
    )
    # The client sends this (without the scheme) as the password; the
    # server derives everything it needs to verify the signature from it.
    return presigned_url.removeprefix("https://")


class ElastiCacheIAMCredentialProvider(CredentialProvider):
    """
    Generates a fresh IAM auth token on each call to get_credentials().
    Pass an instance of this as `credential_provider=` on the Valkey
    client; the client calls it whenever it needs to (re)authenticate,
    including after the ~15 min token expiry or a reconnect.
    """
 
    def __init__(self, user_id: str, cache_name: str, region: str):
        self._user_id = user_id
        self._cache_name = cache_name
        self._region = region
 
    def get_credentials(self):
        token = generate_iam_auth_token(self._user_id, self._cache_name, self._region)
        return self._user_id, token


# ============================================================================
# VECTOR STORE + LLM  (loaded once at import time, both via LiteLLM gateway)
# ============================================================================

print(f"Loading vector store (embeddings via LiteLLM gateway @ {LITELLM_BASE_URL})...")
embeddings = get_embeddings()
vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True,
)
embedding_dimensions = len(embeddings.embed_query("embedding dimension check"))
if vector_store.index.d != embedding_dimensions:
    raise RuntimeError(
        f"The FAISS index has {vector_store.index.d} dimensions, but "
        f"gateway model {EMBEDDING_MODEL!r} produces {embedding_dimensions}. "
        "Rebuild the index by running `python phase1_ingestion/ingest.py`."
    )
print("✅ Vector store loaded")
llm = get_llm(max_tokens=1024)
print(f"✅ LLM client ready (gateway model alias={LLM_MODEL!r})")


async def embed_cache_question(text: str) -> list[float]:
    # Same gateway embedding model as the vector store, called asynchronously.
    return await embeddings.aembed_query(text)


# ============================================================================
# SEMANTIC CACHE  (async, initialised via lifespan)
# ============================================================================

semantic_cache: SemanticCache | None = None
_valkey_client: valkey_async.Valkey | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global semantic_cache, _valkey_client

    valkey_host = os.getenv("VALKEY_HOST", "localhost")
    valkey_port = int(os.getenv("VALKEY_PORT", 6379))
    use_iam_auth = os.getenv("VALKEY_USE_IAM_AUTH", "false").lower() == "true"
 
    if use_iam_auth:
        _valkey_client = valkey_async.Valkey(
            host=valkey_host,
            port=valkey_port,
            ssl=True,
            credential_provider=ElastiCacheIAMCredentialProvider(
                user_id=os.getenv("VALKEY_IAM_USER_ID", "default.iam-user"),
                cache_name=os.getenv("VALKEY_SERVERLESS_CACHE_NAME"),
                region=os.getenv("AWS_REGION", "us-east-2"),
            ),
        )
    else:
        _valkey_client = valkey_async.Valkey(host=valkey_host, port=valkey_port)

    semantic_cache = SemanticCache(
        SemanticCacheOptions(
            name=os.getenv("SEMANTIC_CACHE_NAME", "betterdbscache"),
            client=_valkey_client,
            embed_fn=embed_cache_question,
            default_threshold=0.15,
            use_default_cost_table=True,
        )
    )
    await semantic_cache.initialize()
    print(
        "✅ Semantic cache initialised (Valkey @ "
        f"{valkey_host}:{valkey_port}, iam_auth={use_iam_auth})"
    )

    yield

    await semantic_cache.shutdown()
    await _valkey_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    pong = await _valkey_client.ping()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"{LITELLM_BASE_URL.removesuffix('/v1')}/health/liveliness",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            )
        gateway_ok = r.status_code == 200
    except httpx.HTTPError:
        gateway_ok = False
    return {"status": "ok", "valkey_ping": pong, "litellm_gateway": gateway_ok}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list
    cache_hit: bool = False
    similarity: float | None = None


# ============================================================================
# CHAT ENDPOINT
# ============================================================================


async def _print_stats() -> None:
    s = await semantic_cache.stats()
    cost = s.cost_saved_micros / 1_000_000
    print(
        f"[STATS]      total={s.total} | hits={s.hits} | misses={s.misses} "
        f"| hit_rate={s.hit_rate:.1%} | saved=${cost:.4f}"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    # 1. Check semantic cache
    cache_result = await semantic_cache.check(request.question)

    if cache_result.hit:
        similarity_score = round(1 - (cache_result.similarity or 0), 4)
        print(f"[CACHE HIT]  similarity={similarity_score:.4f}  \"{request.question}\"")
        await _print_stats()
        return ChatResponse(
            answer=cache_result.response,
            sources=[],
            cache_hit=True,
            similarity=similarity_score,
        )

    print(f"[CACHE MISS] \"{request.question}\"")

    # 2. Retrieve relevant PDF chunks
    relevant_docs = vector_store.similarity_search(request.question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    sources = [f"Page {doc.metadata.get('page', 'Unknown')}" for doc in relevant_docs]

    # 3. Build prompt and call LLM
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers
         questions based on the provided context.
         Use the context below to answer the question.
         If you cannot find the answer in the context, say so.

        Context: {context}"""),
        ("user", "{question}"),
    ])
    prompt = prompt_template.format_messages(context=context, question=request.question)
    response = await llm.ainvoke(prompt)
    answer = response.content

    # 4. Store response in cache with token counts for cost tracking
    input_tokens = (response.usage_metadata or {}).get("input_tokens", 0)
    output_tokens = (response.usage_metadata or {}).get("output_tokens", 0)
    await semantic_cache.store(
        request.question,
        answer,
        CacheStoreOptions(
            model=CACHE_COST_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )
    await _print_stats()

    return ChatResponse(answer=answer, sources=sources, cache_hit=False)


# ============================================================================
# STATS ENDPOINT  (end-of-demo reveal)
# ============================================================================


@app.get("/stats")
async def stats():
    s = await semantic_cache.stats()
    return {
        "hits": s.hits,
        "misses": s.misses,
        "total": s.total,
        "hit_rate": f"{s.hit_rate:.1%}",
        "cost_saved_usd": f"${s.cost_saved_micros / 1_000_000:.4f}",
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================


@app.get("/")
def root():
    return {"status": "Backend is running", "message": "Use /chat endpoint to ask questions"}
