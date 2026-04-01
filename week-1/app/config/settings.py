import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docmind_chunks")

DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)
