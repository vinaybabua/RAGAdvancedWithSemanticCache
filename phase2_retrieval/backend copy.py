"""
Phase 2: Backend API with BetterDB Semantic Caching
"""

import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import valkey.asyncio as valkey_async
from betterdb_semantic_cache import SemanticCache, SemanticCacheOptions
from betterdb_semantic_cache.types import CacheStoreOptions

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

# ============================================================================
# VECTOR STORE + LLM  (synchronous, loaded once at startup)
# ============================================================================

print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
    model_kwargs={"device": os.getenv("HF_DEVICE", "cpu")},
)
vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True,
)
print("✅ Vector store loaded")

llm = ChatAnthropic(model=CLAUDE_MODEL, temperature=0.7, max_tokens=1024)


async def embed_cache_question(text: str) -> list[float]:
    return await asyncio.to_thread(embeddings.embed_query, text)

# ============================================================================
# SEMANTIC CACHE  (async, initialised via lifespan)
# ============================================================================

semantic_cache: SemanticCache | None = None
_valkey_client: valkey_async.Valkey | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global semantic_cache, _valkey_client

    _valkey_client = valkey_async.Valkey(
        host=os.getenv("VALKEY_HOST", "localhost"),
        port=int(os.getenv("VALKEY_PORT", 6379)),
    )
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
    print("✅ Semantic cache initialised (Valkey @ "
          f"{os.getenv('VALKEY_HOST', 'localhost')}:{os.getenv('VALKEY_PORT', 6379)})")

    yield

    await semantic_cache.shutdown()
    await _valkey_client.aclose()


# ============================================================================
# APP
# ============================================================================

app = FastAPI(lifespan=lifespan)

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
    response = llm.invoke(prompt)
    answer = response.content

    # 4. Store response in cache with token counts for cost tracking
    input_tokens = (response.usage_metadata or {}).get("input_tokens", 0)
    output_tokens = (response.usage_metadata or {}).get("output_tokens", 0)
    await semantic_cache.store(
        request.question,
        answer,
        CacheStoreOptions(
            model=CLAUDE_MODEL,
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
