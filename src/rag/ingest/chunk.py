"""Token-window baseline and paragraph-aware structural chunking."""

import re

import tiktoken

from rag.ingest.models import Chunk, Section

MAX_TOKENS = 500
OVERLAP = 50
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _encode(text: str) -> list[int]:
    return _ENCODING.encode(text, disallowed_special=())


def _windows(text: str) -> list[str]:
    tokens = _encode(text)
    result: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        result.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += MAX_TOKENS - OVERLAP
    return result


def chunk_fixed(title: str, markdown: str) -> list[Chunk]:
    return [
        Chunk("fixed", i, title, text, len(_encode(text)))
        for i, text in enumerate(_windows(markdown))
        if text.strip()
    ]


def _paragraphs(text: str) -> list[str]:
    """Keep fenced blocks atomic, including blank lines inside them."""
    pieces: list[str] = []
    lines: list[str] = []
    fence: str | None = None

    def flush() -> None:
        piece = "".join(lines).strip()
        if piece:
            pieces.append(piece)
        lines.clear()

    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            lines.append(line)
            if re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*",
                line.rstrip("\r\n"),
            ):
                fence = None
                flush()
        elif marker:
            flush()
            fence = marker[1]
            lines.append(line)
        elif not line.strip():
            flush()
        else:
            lines.append(line)
    flush()
    return pieces


def _section_chunks(text: str) -> list[str]:
    if len(_encode(text)) <= MAX_TOKENS:
        return [text] if text.strip() else []
    chunks: list[str] = []
    pending = ""
    for piece in _paragraphs(text):
        candidate = pending + "\n\n" + piece if pending else piece
        if len(_encode(candidate)) <= MAX_TOKENS:
            pending = candidate
            continue
        if pending:
            chunks.append(pending)
            pending = ""
        if len(_encode(piece)) > MAX_TOKENS:
            chunks.extend(_windows(piece))
        else:
            pending = piece
    if pending:
        chunks.append(pending)
    return chunks


def chunk_structural(sections: list[Section]) -> list[Chunk]:
    # Merge forward into a sibling; the destination supplies the heading path.
    merged: list[Section] = []
    pending: Section | None = None
    for section in sections:
        if not section.text.strip():
            continue
        if pending is not None:
            if (
                len(_encode(pending.text)) < OVERLAP
                and pending.heading_path[:-1] == section.heading_path[:-1]
            ):
                section = Section(section.heading_path, pending.text + "\n\n" + section.text)
            else:
                merged.append(pending)
        pending = section
    if pending is not None:
        merged.append(pending)

    chunks: list[Chunk] = []
    for section in merged:
        for text in _section_chunks(section.text):
            chunks.append(
                Chunk(
                    "structural",
                    len(chunks),
                    " \u203a ".join(section.heading_path),
                    text,
                    len(_encode(text)),
                )
            )
    return chunks
