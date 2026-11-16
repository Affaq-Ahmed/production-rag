from itertools import pairwise

import tiktoken

from rag.ingest.chunk import MAX_TOKENS, MIN_TOKENS, OVERLAP, chunk_fixed, chunk_structural
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


def test_tiny_sections_under_different_parents_merge_backward_under_common_ancestor() -> None:
    chunks = chunk_structural(
        [
            Section(("Title", "A", "One"), "Tiny."),
            Section(("Title", "B", "Two"), "Other."),
        ]
    )
    assert [(c.heading_path, c.text) for c in chunks] == [("Title", "Tiny.\n\nOther.")]


def test_parent_intro_merges_into_its_child() -> None:
    chunks = chunk_structural(
        [
            Section(("T", "Probes"), "Probes check containers."),
            Section(("T", "Probes", "Readiness"), " word" * 70),
        ]
    )
    assert len(chunks) == 1
    assert chunks[0].heading_path == "T \u203a Probes \u203a Readiness"
    assert chunks[0].text.startswith("Probes check containers.")


def test_last_tiny_section_merges_backward_under_common_ancestor() -> None:
    chunks = chunk_structural(
        [
            Section(("T", "Troubleshooting", "Logs"), " word" * 100),
            Section(("T", "Clean up"), "Delete the pod."),
        ]
    )
    assert len(chunks) == 1
    assert chunks[0].heading_path == "T"
    assert chunks[0].text.endswith("Delete the pod.")


def test_last_child_before_outline_rises_merges_backward() -> None:
    chunks = chunk_structural(
        [
            Section(("T", "A"), " alpha" * 70),
            Section(("T", "A", "A1"), " beta" * 70),
            Section(("T", "A", "A2"), "Tiny last child."),
            Section(("T", "B"), " gamma" * 70),
        ]
    )
    assert [c.heading_path for c in chunks] == ["T \u203a A", "T \u203a A", "T \u203a B"]
    assert chunks[1].text.startswith(" beta") and chunks[1].text.endswith("Tiny last child.")
    assert "Tiny last child." not in chunks[2].text


def test_tiny_section_is_kept_when_backward_merge_would_exceed_limit() -> None:
    chunks = chunk_structural(
        [
            # 498 tokens alone; 503 with the tiny section appended.
            Section(("T", "A"), " word" * 498),
            Section(("T", "B"), "Tiny tail text."),
        ]
    )
    assert [c.heading_path for c in chunks] == ["T \u203a A", "T \u203a B"]
    assert all(c.token_count <= MAX_TOKENS for c in chunks)


def test_tail_of_oversized_section_is_left_alone() -> None:
    text = "\n\n".join([" word" * 480, " word" * 20])
    chunks = chunk_structural([Section(("T", "A"), text)])
    assert [c.heading_path for c in chunks] == ["T \u203a A", "T \u203a A"]
    assert chunks[1].token_count < MIN_TOKENS


def test_structural_greedy_paragraphs() -> None:
    chunks = chunk_structural([Section(("Title",), "\n\n".join([" word" * 200] * 3))])
    assert len(chunks) == 2
    assert all(c.token_count <= MAX_TOKENS for c in chunks)
