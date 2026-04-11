import re


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []

    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step

    return chunks


def chunk_text_by_article(
    text: str,
    fallback_chunk_size: int,
    fallback_chunk_overlap: int,
) -> list[str]:
    raw_text = text.strip()
    if not raw_text:
        return []

    article_header_pattern = re.compile(
        r"(?im)^\s*((?:art[ii]culo|art\.)\s*\d+[a-zA-Z0-9\-\./]*\s*.*)$"
    )
    matches = list(article_header_pattern.finditer(raw_text))

    if not matches:
        return chunk_text(raw_text, fallback_chunk_size, fallback_chunk_overlap)

    sections: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        section = raw_text[start:end].strip()
        if not section:
            continue

        if len(section) <= fallback_chunk_size:
            sections.append(section)
            continue

        header = match.group(1).strip()
        body = section[len(match.group(1)) :].strip()
        body_chunks = chunk_text(body, fallback_chunk_size, fallback_chunk_overlap)
        for part in body_chunks:
            sections.append(f"{header}\n{part}".strip())

    return sections
