"""
LiteLLM gateway clients (shared by ingestion, backend and demo scripts).

Every LLM and embedding call in this project goes through the LiteLLM
proxy (an OpenAI-compatible gateway). The app never talks to Anthropic or
Hugging Face directly -- it only knows:

    LITELLM_BASE_URL   where the gateway is running (e.g. http://localhost:4000)
    LITELLM_API_KEY    a gateway key (master key or a virtual key)
    LLM_MODEL          gateway model alias for chat      (e.g. "claude-chat")
    EMBEDDING_MODEL    gateway model alias for embeddings (e.g. "hf-embeddings")

Which real provider/model sits behind each alias is decided in
gateway/litellm_config.yaml, so you can swap Claude -> GPT / Bedrock / Ollama,
or swap embedding models, without touching this code.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "hf-embeddings")


def _openai_base_url() -> str:
    # The LiteLLM proxy serves the OpenAI routes at both "/" and "/v1".
    return LITELLM_BASE_URL if LITELLM_BASE_URL.endswith("/v1") else f"{LITELLM_BASE_URL}/v1"


def get_embeddings() -> OpenAIEmbeddings:
    """Embeddings client that calls the gateway's /v1/embeddings route."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=_openai_base_url(),
        api_key=LITELLM_API_KEY,
        # Send raw text, not tiktoken token IDs. Token-ID input only works
        # for OpenAI's own embedding models, not models behind a gateway.
        check_embedding_ctx_length=False,
        chunk_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
    )


def get_llm(temperature: float = 1.0, max_tokens: int = 1024) -> ChatOpenAI:
    """Chat client that calls the gateway's /v1/chat/completions route."""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=_openai_base_url(),
        api_key=LITELLM_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        # Ask the gateway to include token usage so the semantic cache can
        # track cost savings.
        stream_usage=True,
    )
