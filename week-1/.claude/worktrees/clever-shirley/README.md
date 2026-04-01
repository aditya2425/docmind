# DocuMind — Week 1 Foundation

DocuMind is a production-style RAG project.

This Week 1 build covers:

- PDF ingestion with PyMuPDF
- Chunking with fixed, recursive, and semantic strategies
- Embeddings using sentence-transformers or OpenAI
- ChromaDB vector storage
- Semantic search over ingested PDF content

## Why this project

This follows the roadmap's Week 1 target:
- project structure
- PDF ingestion with PyMuPDF
- chunking module
- embedding pipeline
- ChromaDB storage

## Project Structure

```bash
docmind/
│
├── app/
│   ├── main.py
│   ├── config/
│   │   └── settings.py
│   └── ingestion/
│       ├── pdf_loader.py
│       ├── chunker.py
│       ├── embedder.py
│       └── vector_store.py
│
├── data/
│   ├── raw/
│   └── chroma/
│
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Create virtual environment:

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Create `.env` file:

```env
EMBEDDING_PROVIDER=local
OPENAI_API_KEY=
CHROMA_PATH=./data/chroma
COLLECTION_NAME=docmind_chunks
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=100
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Add a PDF

Put a test PDF inside:

```bash
data/raw/sample.pdf
```

## Run

### Fixed chunking
```bash
python -m app.main --pdf data/raw/sample.pdf --method fixed --query "What is this document about?"
```

### Recursive chunking
```bash
python -m app.main --pdf data/raw/sample.pdf --method recursive --query "What are the main ideas?"
```

### Semantic chunking
```bash
python -m app.main --pdf data/raw/sample.pdf --method semantic --query "What is the conclusion?"
```

## Notes

- `fixed` is your simplest baseline
- `recursive` preserves structure better
- `semantic` here is a Week 1 approximation, not a fully advanced semantic chunker

## Week 1 Success Criteria

- PDF text extraction works
- chunking works in 3 styles
- embeddings are generated
- vectors are stored in ChromaDB
- semantic retrieval returns top matching chunks
