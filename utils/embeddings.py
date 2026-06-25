from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def create_embeddings(chunks):

    if not chunks:
        return np.array([])

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embedding type:", type(embeddings))
    print("Embedding shape:", embeddings.shape)

    return embeddings


def create_query_embedding(query):

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding