from typing import Dict, List


def fixed_chunk(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Fixed-size character chunking with overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict] = []

    for page_data in pages:
        text = page_data["text"]
        source = page_data["source"]
        page = page_data["page"]

        start = 0
        chunk_num = 1
        step = chunk_size - overlap

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{source}_p{page}_c{chunk_num}_fixed",
                        "text": chunk_text,
                        "source": source,
                        "page": page,
                        "chunking_method": "fixed",
                    }
                )

            start += step
            chunk_num += 1

    return chunks


def recursive_split_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    """
    Beginner-friendly recursive-ish strategy:
    1. Try grouping sentence-like parts.
    2. If one part is still too large, hard split it.
    """
    if len(text) <= chunk_size:
        return [text]

    parts = [part.strip() for part in text.split(". ") if part.strip()]
    chunks: List[str] = []
    current = ""

    for part in parts:
        if not part.endswith("."):
            part = part + "."

        candidate = f"{current} {part}".strip() if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())

            if len(part) > chunk_size:
                start = 0
                step = chunk_size - overlap
                while start < len(part):
                    piece = part[start:start + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                    start += step
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def recursive_chunk(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Recursive-like chunking that tries to preserve sentence groups.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict] = []

    for page_data in pages:
        split_chunks = recursive_split_text(
            text=page_data["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for idx, chunk_text in enumerate(split_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"{page_data['source']}_p{page_data['page']}_c{idx}_recursive",
                    "text": chunk_text,
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunking_method": "recursive",
                }
            )

    return chunks


def semantic_chunk(
    pages: List[Dict],
    chunk_size: int = 500
) -> List[Dict]:
    """
    Very simple Week 1 semantic-style chunking:
    - split into sentence-like units
    - merge nearby sentences while size allows
    This is not a true embedding-driven semantic chunker,
    but it is a good Week 1 approximation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    chunks: List[Dict] = []

    for page_data in pages:
        text = page_data["text"]
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        current = ""
        chunk_num = 1

        for sentence in sentences:
            sentence = sentence.strip() + "."
            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(
                        {
                            "chunk_id": f"{page_data['source']}_p{page_data['page']}_c{chunk_num}_semantic",
                            "text": current.strip(),
                            "source": page_data["source"],
                            "page": page_data["page"],
                            "chunking_method": "semantic",
                        }
                    )
                    chunk_num += 1
                current = sentence

        if current.strip():
            chunks.append(
                {
                    "chunk_id": f"{page_data['source']}_p{page_data['page']}_c{chunk_num}_semantic",
                    "text": current.strip(),
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunking_method": "semantic",
                }
            )

    return chunks
