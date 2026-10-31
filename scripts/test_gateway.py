"""
Smoke test for the LiteLLM gateway. Run this after starting the gateway and
before ingestion:

    python scripts/test_gateway.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_gateway import EMBEDDING_MODEL, LITELLM_BASE_URL, LLM_MODEL, get_embeddings, get_llm  # noqa: E402

print(f"Gateway: {LITELLM_BASE_URL}\n")

print(f"[1] Embeddings via alias {EMBEDDING_MODEL!r} ...")
vectors = get_embeddings().embed_documents(["What is the symposium about?", "When is the event?"])
print(f"    OK - {len(vectors)} vectors, {len(vectors[0])} dimensions, first values {vectors[0][:3]}")
if len(vectors[0]) < 10:
    sys.exit("    ✗ Vectors look like similarity scores, not embeddings - check gateway/litellm_config.yaml")

print(f"\n[2] Chat via alias {LLM_MODEL!r} ...")
reply = get_llm(max_tokens=50).invoke("Reply with exactly: gateway works")
print(f"    OK - {reply.content!r}")
print(f"    usage: {reply.usage_metadata}")

print("\n✅ LiteLLM gateway is working for both chat and embeddings")
