"""
Demo Script: Visualize Chunks and Embeddings
This script demonstrates the result of the embedding process by showing:
1. Text chunks from the document
2. Their corresponding vector embeddings
3. Vector dimensions and properties
"""

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from llm_gateway import EMBEDDING_MODEL, LITELLM_BASE_URL, get_embeddings

# Load environment variables
load_dotenv()

print("=" * 80)
print("EMBEDDING DEMONSTRATION")
print("=" * 80)

# ============================================================================
# 1. LOAD THE VECTOR STORE
# ============================================================================
print("\n[Step 1] Loading vector store...")

# Query embeddings are created by the LiteLLM gateway (same model alias
# that was used during ingestion).
embeddings = get_embeddings()
print(f"Using gateway {LITELLM_BASE_URL} with embedding model alias {EMBEDDING_MODEL!r}")
vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Vector store loaded successfully")

# ============================================================================
# 2. GET STORE STATISTICS
# ============================================================================
print("\n[Step 2] Vector Store Statistics:")
print("-" * 80)

# Get the total number of vectors
total_vectors = vector_store.index.ntotal
print(f"Total number of chunks: {total_vectors}")

# Get the dimension of vectors
vector_dimension = vector_store.index.d
print(f"Vector dimension: {vector_dimension}")

# ============================================================================
# 3. DEMONSTRATE CHUNKS AND EMBEDDINGS
# ============================================================================
print("\n[Step 3] Sample Chunks and Their Embeddings:")
print("-" * 80)

# Get the docstore (contains all the text chunks)
docstore = vector_store.docstore._dict
index_to_docstore_id = vector_store.index_to_docstore_id

# How many samples to show (change this to show more/less)
NUM_SAMPLES = 3

print(f"\nShowing {NUM_SAMPLES} examples of text chunks and their embeddings:\n")

for i in range(min(NUM_SAMPLES, total_vectors)):
    print(f"\n{'=' * 80}")
    print(f"EXAMPLE {i + 1}")
    print(f"{'=' * 80}")

    # Get the document ID and the document
    doc_id = index_to_docstore_id[i]
    document = docstore[doc_id]

    # Get the text content
    text = document.page_content
    metadata = document.metadata

    # Get the corresponding vector
    vector = vector_store.index.reconstruct(i)

    # Display the text chunk
    print(f"\n📄 TEXT CHUNK {i + 1}:")
    print("-" * 80)
    print(f"Source: {metadata.get('source', 'Unknown')}")
    print(f"Page: {metadata.get('page', 'Unknown')}")
    print(f"\nContent (first 300 characters):")
    print(f"{text[:300]}..." if len(text) > 300 else text)

    # Display the vector
    print(f"\n🔢 CORRESPONDING VECTOR EMBEDDING:")
    print("-" * 80)
    print(f"Vector dimension: {len(vector)}")
    print(f"Vector type: {type(vector)}")
    print(f"\nFirst 10 values: {vector[:10]}")
    print(f"Last 10 values: {vector[-10:]}")

    # Display vector statistics
    print(f"\n📊 VECTOR STATISTICS:")
    print("-" * 80)
    print(f"Mean: {np.mean(vector):.6f}")
    print(f"Std Dev: {np.std(vector):.6f}")
    print(f"Min: {np.min(vector):.6f}")
    print(f"Max: {np.max(vector):.6f}")

# ============================================================================
# 4. DEMONSTRATE VECTOR SIMILARITY
# ============================================================================
print("\n\n" + "=" * 80)
print("[Step 4] Demonstrating Vector Similarity")
print("=" * 80)

# Get two vectors for comparison
if total_vectors >= 2:
    vector_1 = vector_store.index.reconstruct(0)
    vector_2 = vector_store.index.reconstruct(1)

    # Calculate cosine similarity
    def cosine_similarity(v1, v2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    similarity = cosine_similarity(vector_1, vector_2)

    print(f"\nCosine similarity between first two chunks: {similarity:.4f}")
    print(f"\nNote: Similarity ranges from -1 to 1")
    print(f"  • 1.0 = Identical vectors")
    print(f"  • 0.0 = Orthogonal (no similarity)")
    print(f" • -1.0 = Opposite vectors")

# ============================================================================
# 5. DEMONSTRATE SEARCH CAPABILITY
# ============================================================================
print("\n\n" + "=" * 80)
print("[Step 5] Demonstrating Semantic Search")
print("=" * 80)

# Example search query
test_query = "What is this symposium about?"
print(f"\nTest Query: '{test_query}'")
print("\nSearching for similar chunks...")

# Perform similarity search
results = vector_store.similarity_search(test_query, k=2)

print(f"\nFound {len(results)} relevant chunks:\n")

for idx, doc in enumerate(results):
    print(f"\n--- Result {idx + 1} ---")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:200]}...")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
📚 What We Demonstrated:

1. Vector Store Loading
   - Loaded {total_vectors} text chunks from the symposium document

2. Embeddings Structure
   - Each chunk is converted to a {vector_dimension}-dimensional vector
   - These vectors are mathematical representations of meaning

3. Vector Properties
   - Vectors are arrays of floating-point numbers
   - Similar meanings → Similar vector values
   - This enables semantic (meaning-based) search

4. Semantic Search
   - Queries are converted to vectors using the same embedding model
   - FAISS quickly finds chunks with similar vector values
   - This retrieves relevant context without keyword matching

5. Key Concept
   - Text → Numbers → Fast Similarity Search → Relevant Results
   - This is the foundation of RAG (Retrieval-Augmented Generation)
""")

print("=" * 80)
print("✅ DEMONSTRATION COMPLETE")
print("=" * 80)
