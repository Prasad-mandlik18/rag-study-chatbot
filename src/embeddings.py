from sentence_transformers import SentenceTransformer

# Load the model only once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    """
    Convert a list of text chunks into embeddings.
    """
    embeddings = model.encode(chunks)
    return embeddings


def generate_query_embedding(query):
    return model.encode(query)