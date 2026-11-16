"""Token-window baseline and paragraph-aware structural chunking."""

import re

import tiktoken

from rag.ingest.models import Chunk, Section

MAX_TOKENS = 500
OVERLAP = 50
# Sections smaller than this are folded into a neighbour (structural strategy only).
MIN_TOKENS = 50
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


def _absorbs_forward(tiny: tuple[str, ...], following: tuple[str, ...]) -> bool:
    """A sibling, or the tiny section's own child/descendant, can take it in."""
    sibling = tiny[:-1] == following[:-1]
    descendant = len(following) > len(tiny) and following[: len(tiny)] == tiny
    return sibling or descendant


def _common_ancestor(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    shared: list[str] = []
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        shared.append(a)
    return tuple(shared)


def _merge_backward_or_keep(merged: list[Section], tiny: Section) -> None:
    """Append into the previous section if it fits, labelled by the paths' common ancestor.

    The previous section's own path would file the tiny text under a heading it doesn't
    belong to, and the heading path feeds both the embedding and the tsv column.
    """
    if merged:
        previous = merged[-1]
        combined = previous.text + "\n\n" + tiny.text
        if len(_encode(combined)) <= MAX_TOKENS:
            path = _common_ancestor(previous.heading_path, tiny.heading_path)
            merged[-1] = Section(path, combined)
            return
    merged.append(tiny)


def _merge_tiny(sections: list[Section]) -> list[Section]:
    """Fold sections under MIN_TOKENS into a neighbour.

    1. Forward into the next section if it is a sibling or the tiny section's own child,
       keeping the next section's path (a child's path already includes the parent).
    2. Otherwise backward into the previous section if the result fits in MAX_TOKENS.
    3. Otherwise keep it. Small tails of oversized sections are split later and left alone.
    """
    merged: list[Section] = []
    pending: Section | None = None
    for section in sections:
        if not section.text.strip():
            continue
        if pending is not None:
            if _absorbs_forward(pending.heading_path, section.heading_path):
                section = Section(section.heading_path, pending.text + "\n\n" + section.text)
            else:
                _merge_backward_or_keep(merged, pending)
            pending = None
        if len(_encode(section.text)) < MIN_TOKENS:
            pending = section
        else:
            merged.append(section)
    if pending is not None:
        _merge_backward_or_keep(merged, pending)
    return merged


def chunk_structural(sections: list[Section]) -> list[Chunk]:
    merged = _merge_tiny(sections)
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
