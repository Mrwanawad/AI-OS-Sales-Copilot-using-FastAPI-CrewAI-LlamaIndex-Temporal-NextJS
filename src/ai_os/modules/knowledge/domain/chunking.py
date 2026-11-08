from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from ai_os.modules.knowledge.domain.models import Chunk, Citation, ParsedSection, SourceDocument


def chunk_sections(
    document: SourceDocument,
    sections: Sequence[ParsedSection],
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        if section.atomic:
            pieces = [section.text.strip()] if section.text.strip() else []
        else:
            pieces = _with_overlap(_split(section.text, max_chars), overlap)
        for text in pieces:
            chunks.append(_chunk(document, section, text, len(chunks)))
    return chunks


def _chunk(document: SourceDocument, section: ParsedSection, text: str, index: int) -> Chunk:
    return Chunk(
        id=str(uuid5(NAMESPACE_URL, f"{document.id}:{document.version}:{index}")),
        text=text,
        citation=Citation(document.id, document.version, section.page),
        project=document.project,
        kind=document.kind,
        unit_codes=section.unit_codes,
    )


def _split(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_long(paragraph, max_chars))
        elif current and len(current) + len(paragraph) + 2 > max_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = paragraph
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        cut = cut if cut > 0 else max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
    if not overlap:
        return pieces
    result = pieces[:1]
    for previous, piece in zip(pieces, pieces[1:], strict=False):
        tail = previous[-overlap:]
        # Start the overlap on a word boundary.
        tail = tail[tail.find(" ") + 1 :]
        result.append(f"{tail}\n{piece}")
    return result
