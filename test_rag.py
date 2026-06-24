from utils.pdf_loader import load_pdf
from utils.chunking import split_text
from utils.embeddings import create_embeddings
from utils.gemini_rag import generate_answer
from utils.vector_store import (
    store_embeddings,
    retrieve_chunks
)

pdf_path = "data/sample.pdf"

# Load PDF
text = load_pdf(pdf_path)

# Split into chunks
chunks = split_text(text)

# Create embeddings
embeddings = create_embeddings(chunks)

print(f"Total Chunks: {len(chunks)}")
print(f"Total Embeddings: {len(embeddings)}")
print(f"Embedding Dimension: {len(embeddings[0])}")

# Store in ChromaDB
store_embeddings(chunks, embeddings)

print("\nStored in ChromaDB Successfully!")

# ----------------------------
# Retrieval Test
# ----------------------------

query = "What is Cloud Computing?"

print(f"\nQuestion: {query}")

query_embedding = create_embeddings([query])[0]

results = retrieve_chunks(query_embedding)

retrieved_chunks = results["documents"][0]

answer = generate_answer(
    query,
    retrieved_chunks
)

print("\n" + "=" * 50)
print("FINAL ANSWER")
print("=" * 50)

print(answer)

print("\nTop Retrieved Chunks:\n")

for i, doc in enumerate(results["documents"][0], start=1):
    print(f"\n===== Result {i} =====\n")
    print(doc)