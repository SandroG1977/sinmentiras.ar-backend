import re


def _normalize_heading_kind(kind: str) -> str:
    return kind.lower().replace("í", "i").replace("ó", "o")


def _active_headings_before(
    heading_matches: list[re.Match[str]],
    start_pos: int,
) -> list[str]:
    active_by_kind: dict[str, str] = {}
    for match in heading_matches:
        if match.start() >= start_pos:
            break
        kind = _normalize_heading_kind(match.group("kind"))
        active_by_kind[kind] = match.group(1).strip()

    ordered_keys = ["libro", "titulo", "capitulo", "seccion"]
    return [active_by_kind[key] for key in ordered_keys if key in active_by_kind]


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
        r"(?im)^\s*((?:art[íi]culo|art\.)\s*\d+[a-zA-Z0-9\-\./º°]*\s*.*)$"
    )
    heading_pattern = re.compile(
        r"(?im)^\s*((?P<kind>libro|t[íi]tulo|cap[íi]tulo|secci[óo]n)\s+[A-Za-z0-9IVXLCDM\-\.: ]+)$"
    )

    matches = list(article_header_pattern.finditer(raw_text))
    heading_matches = list(heading_pattern.finditer(raw_text))

    if not matches:
        return chunk_text(raw_text, fallback_chunk_size, fallback_chunk_overlap)

    sections: list[str] = []

    preamble_text = raw_text[: matches[0].start()].strip()
    if preamble_text:
        for part in chunk_text(
            preamble_text,
            fallback_chunk_size,
            fallback_chunk_overlap,
        ):
            sections.append(f"PREAMBULO\n{part}".strip())

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
        section = raw_text[start:end].strip()
        if not section:
            continue

        heading_context = "\n".join(_active_headings_before(heading_matches, start))

        if len(section) <= fallback_chunk_size:
            enriched = f"{heading_context}\n{section}" if heading_context else section
            sections.append(enriched.strip())
            continue

        header = match.group(1).strip()
        body = section[len(match.group(1)) :].strip()
        full_header = f"{heading_context}\n{header}" if heading_context else header
        body_chunks = chunk_text(body, fallback_chunk_size, fallback_chunk_overlap)
        for part in body_chunks:
            sections.append(f"{full_header}\n{part}".strip())

    return sections
