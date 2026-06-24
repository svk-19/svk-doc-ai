from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# --------------------------------------------------
# DOCUMENT EMBEDDINGS
# --------------------------------------------------

def create_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings

# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

def create_query_embedding(query):

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding