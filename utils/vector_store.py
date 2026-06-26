import numpy as np
import faiss
import os
import pickle

# --------------------------------------------------
# PATHS
# --------------------------------------------------

INDEX_PATH = "/tmp/faiss_index.bin"
CHUNKS_PATH = "/tmp/faiss_chunks.pkl"

# --------------------------------------------------
# STORE EMBEDDINGS
# --------------------------------------------------

def store_embeddings(chunks, embeddings):

    if not chunks:
        print("No chunks to store.")
        return

    embeddings_np = np.array(
        [list(map(float, emb)) for emb in embeddings],
        dtype=np.float32
    )

    # Fix shape if somehow 1D
    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Stored {len(chunks)} chunks in FAISS.")

# --------------------------------------------------
# RETRIEVE CHUNKS
# --------------------------------------------------

def retrieve_chunks(query_embedding, top_k=8):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return {"documents": [[]]}

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # ✅ Guard: no chunks stored
    if len(chunks) == 0:
        return {"documents": [[]]}

    query_np = np.array(
        [list(map(float, query_embedding))],
        dtype=np.float32
    )

    # ✅ Guard: don't request more than available
    top_k = min(top_k, len(chunks))

    distances, indices = index.search(query_np, top_k)

    matched_chunks = [
        chunks[i] for i in indices[0] if i < len(chunks)
    ]

    return {"documents": [matched_chunks]}