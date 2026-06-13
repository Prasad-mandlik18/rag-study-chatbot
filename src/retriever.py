from src.embeddings import generate_query_embedding

# Lower distance = better match
SIMILARITY_THRESHOLD = 1.2


def retrieve_chunks(vector_store, question, top_k=3):
    query_embedding = generate_query_embedding(question)

    results = vector_store.search(query_embedding, k=top_k)

    relevant_chunks = []

    for item in results:
        if item["distance"] <= SIMILARITY_THRESHOLD:
            relevant_chunks.append(item["chunk"])

    return relevant_chunks