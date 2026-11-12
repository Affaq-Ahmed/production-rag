from itertools import pairwise

import tiktoken

from rag.ingest.chunk import MAX_TOKENS, OVERLAP, chunk_fixed, chunk_structural
from rag.ingest.models import Section

ENCODING = tiktoken.get_encoding("cl100k_base")


def test_fixed_short() -> None:
    chunks = chunk_fixed("Title", "A short document.")
    assert len(chunks) == 1
    assert chunks[0].ord == 0
    assert chunks[0].heading_path == "Title"
    assert chunks[0].strategy == "fixed"


def test_fixed_overlap_and_limits() -> None:
    text = " word" * 1400
    chunks = chunk_fixed("Title", text)
    source_tokens = ENCODING.encode(text)
    assert len(chunks) > 1
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    for i, chunk in enumerate(chunks):
        tokens = ENCODING.encode(chunk.text)
        start = i * (MAX_TOKENS - OVERLAP)
        assert tokens == source_tokens[start : start + MAX_TOKENS]
        assert 0 < chunk.token_count == len(tokens) <= MAX_TOKENS
    for left, right in pairwise(chunks):
        assert ENCODING.encode(left.text)[-OVERLAP:] == ENCODING.encode(right.text)[:OVERLAP]


def test_fixed_empty() -> None:
    assert chunk_fixed("Title", "") == []
    assert chunk_fixed("Title", "  \n") == []


def test_structural_small() -> None:
    chunks = chunk_structural([Section(("Title", "A"), "Short text.")])
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Title \u203a A"
    assert chunks[0].strategy == "structural"


def test_structural_large() -> None:
    chunks = chunk_structural([Section(("Title", "A"), " word" * 1600)])
    assert len(chunks) > 1
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert all(c.heading_path == "Title \u203a A" for c in chunks)
    assert all(0 < c.token_count <= MAX_TOKENS for c in chunks)


def test_structural_preserves_code_block() -> None:
    code = "```shell\n# Check pods\nkubectl get pods\n\nkubectl describe pod test\n```"
    text = " word" * 470 + "\n\n" + code + "\n\n" + " tail" * 470
    chunks = chunk_structural([Section(("Title", "A"), text)])
    assert len(chunks) > 1
    assert sum(code in c.text for c in chunks) == 1
    assert all(c.token_count <= MAX_TOKENS for c in chunks)


def test_structural_merges_tiny_sibling_forward() -> None:
    chunks = chunk_structural(
        [
            Section(("Title", "A"), "Tiny intro."),
            Section(("Title", "B"), " word" * 70),
        ]
    )
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Title \u203a B"
    assert "Tiny intro." in chunks[0].text
    assert "word" in chunks[0].text


def test_structural_different_parent_is_not_merged() -> None:
    chunks = chunk_structural(
        [
            Section(("Title", "A", "One"), "Tiny."),
            Section(("Title", "B", "Two"), "Other."),
        ]
    )
    assert len(chunks) == 2


def test_structural_greedy_paragraphs() -> None:
    chunks = chunk_structural([Section(("Title",), "\n\n".join([" word" * 200] * 3))])
    assert len(chunks) == 2
    assert all(c.token_count <= MAX_TOKENS for c in chunks)
