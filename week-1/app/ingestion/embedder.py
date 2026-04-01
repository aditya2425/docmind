from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config.settings import (
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)


class EmbeddingClient:
    def __init__(self, provider: str = EMBEDDING_PROVIDER) -> None:
        self.provider = provider

        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing in .env")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = OPENAI_EMBEDDING_MODEL
        else:
            self.client = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
            self.model_name = LOCAL_EMBEDDING_MODEL

        print(f"[EmbeddingClient] Provider : {self.provider}")
        print(f"[EmbeddingClient] Model    : {self.model_name}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        print(f"[EmbeddingClient] Embedding {len(texts)} text(s) using '{self.provider}' provider...")

        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            print(f"[EmbeddingClient] Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")
            return embeddings

        embeddings = self.client.encode(texts, convert_to_numpy=True)
        print(f"[EmbeddingClient] Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]
