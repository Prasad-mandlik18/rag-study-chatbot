import os
import pickle
import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []

    def add_embeddings(self, embeddings, chunks):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding, k=3):
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "distance": float(distance)
                })

        return results

    def save(self):
        os.makedirs("vector_db", exist_ok=True)

        faiss.write_index(
            self.index,
            "vector_db/faiss_index.bin"
        )

        with open("vector_db/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        if (
            os.path.exists("vector_db/faiss_index.bin")
            and os.path.exists("vector_db/chunks.pkl")
        ):
            self.index = faiss.read_index(
                "vector_db/faiss_index.bin"
            )

            with open("vector_db/chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)

            return True

        return False