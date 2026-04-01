import argparse
from pathlib import Path
from typing import Dict, List

from app.config.settings import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.ingestion.chunker import fixed_chunk, recursive_chunk, semantic_chunk
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.vector_store import ChromaVectorStore


def build_chunks(
    pages: List[Dict],
    method: str,
    chunk_size: int,
    overlap: int
) -> List[Dict]:
    if method == "fixed":
        return fixed_chunk(pages, chunk_size=chunk_size, overlap=overlap)
    if method == "recursive":
        return recursive_chunk(pages, chunk_size=chunk_size, overlap=overlap)
    if method == "semantic":
        return semantic_chunk(pages, chunk_size=chunk_size)

    raise ValueError("Invalid method. Use: fixed, recursive, semantic")


def preview_pages(pages: List[Dict], max_chars: int = 250) -> None:
    print("\nExtracted Pages Preview")
    print("-" * 60)
    for item in pages[:3]:
        print(f"Page: {item['page']}, Source: {item['source']}")
        print(item["text"][:max_chars])
        print("-" * 60)


def print_results(results: Dict) -> None:
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else []

    print("\nTop Search Results")
    print("=" * 60)

    for i in range(len(ids)):
        print(f"Rank: {i + 1}")
        print(f"ID: {ids[i]}")
        print(f"Source: {metas[i].get('source')}")
        print(f"Page: {metas[i].get('page')}")
        print(f"Method: {metas[i].get('chunking_method')}")
        if distances:
            print(f"Distance: {distances[i]}")
        print(f"Preview: {docs[i][:350]}")
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind Week 1 pipeline")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF")
    parser.add_argument(
        "--method",
        default="fixed",
        choices=["fixed", "recursive", "semantic"],
        help="Chunking method"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap"
    )
    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Search query"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="How many results to return"
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\nLoading PDF: {pdf_path}")
    pages = load_pdf(str(pdf_path))
    print(f"Non-empty pages extracted: {len(pages)}")
    preview_pages(pages)

    chunks = build_chunks(
        pages=pages,
        method=args.method,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    print(f"\nChunks created: {len(chunks)}")

    store = ChromaVectorStore()
    store.add_chunks(chunks)

    print(f"Collection count: {store.count()}")
    print(f"\nQuery: {args.query}")

    results = store.search(args.query, top_k=args.top_k)
    print_results(results)


if __name__ == "__main__":
    main()
