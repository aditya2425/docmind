from typing import Dict, List

import chromadb

from app.config.settings import CHROMA_PATH, COLLECTION_NAME
from app.ingestion.embedder import EmbeddingClient


class ChromaVectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = EmbeddingClient()

    def add_chunks(self, chunks: List[Dict]) -> None:
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunking_method": chunk["chunking_method"],
            }
            for chunk in chunks
        ]
        embeddings = self.embedder.embed_texts(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query: str, top_k: int = 3) -> Dict:
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def count(self) -> int:
        return self.collection.count()
