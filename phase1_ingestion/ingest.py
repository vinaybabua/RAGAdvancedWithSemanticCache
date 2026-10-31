"""
Phase 1: Data Ingestion
This script loads a PDF, chunks it, creates embeddings through the LiteLLM
gateway, and stores them in a FAISS vector store.

Run from the project root:
    python phase1_ingestion/ingest.py
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Make the project root importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_gateway import EMBEDDING_MODEL, LITELLM_BASE_URL, get_embeddings  # noqa: E402

# Load environment variables
load_dotenv()


# ============================================================================
# 1. LOAD PDF
# ============================================================================
print("Loading PDF...")

pdf_path = "data/symposium.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages from PDF")


# ============================================================================
# 2. CHUNK TEXT
# ============================================================================
print("\nChunking text...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Each chunk will be ~1000 characters
    chunk_overlap=200,      # 200 character overlap between chunks for context
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from the document")


# ============================================================================
# 3. CREATE EMBEDDINGS (via LiteLLM gateway)
# ============================================================================
print("\nCreating embeddings...")

embeddings = get_embeddings()

print(f"Embeddings client initialized (gateway={LITELLM_BASE_URL}, model alias={EMBEDDING_MODEL!r})")


# ============================================================================
# 4. STORE IN VECTOR STORE
# ============================================================================
print("\nStoring embeddings in FAISS vector store...")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print(f"Vector store created successfully ({vector_store.index.d}-dimensional vectors)")


# ============================================================================
# 5. SAVE TO DISK
# ============================================================================
print("\nSaving vector store to disk...")

vector_store_path = "vector_store"
vector_store.save_local(vector_store_path)

print(f"Vector store saved to '{vector_store_path}/' folder")
print("\n✅ Data ingestion completed successfully!")
