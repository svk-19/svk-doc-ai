import chromadb

# --------------------------------------------------
# CHROMA CLIENT
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_rag"
)

# --------------------------------------------------
# STORE EMBEDDINGS
# --------------------------------------------------

def store_embeddings(
    chunks,
    embeddings
):

    try:

        existing = collection.get()

        if existing["ids"]:

            collection.delete(
                ids=existing["ids"]
            )

    except:
        pass

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]
    
    try:
        collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )
        
    except Exception as e:
     print("CHROMA ERROR:", str(e))
     raise e

# --------------------------------------------------
# RETRIEVE CHUNKS
# --------------------------------------------------

def retrieve_chunks(
    query_embedding,
    top_k=8
):

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results