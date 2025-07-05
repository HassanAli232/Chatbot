import numpy as np
import faiss

class RoadVectorDB:
    def __init__(self):
        self.index = None
        self.road_names = []
        self.dim = None

    def embed_texts(self, texts, client, batch_size=10):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    def build_index(self, road_names, client):
        self.road_names = road_names
        embeddings = self.embed_texts(road_names, client)
        self.dim = len(embeddings[0])
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.array(embeddings))

    def search(self, query, client, top_k=3, threshold=0.6):
        if self.index is None:
            raise RuntimeError("FAISS index not built.")

        response = client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)
        return [self.road_names[i] for i, score in zip(indices[0], scores[0]) if score >= threshold]
