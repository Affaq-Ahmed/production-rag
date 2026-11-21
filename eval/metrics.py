"""Retrieval metrics on document paths: pure functions, no database, no network.

A hit counts as relevant when its path is in the question's relevant_paths. k counts chunks,
not pages, and hits are never de-duplicated by page: if the top 5 chunks all come from one
page, that is what the LLM receives in Week 4, so that is what is scored.
"""

from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from rag.retrieve.models import Hit


def first_relevant_rank(hits: Sequence[Hit], relevant: AbstractSet[str]) -> int | None:
    """1-based position of the first relevant hit, or None if no hit is relevant."""
    _require_labels(relevant)
    # The list order is the ranking. Hit.rank should agree, but position is what the reader
    # (and the LLM) actually sees, so it can't drift from it.
    for position, hit in enumerate(hits, start=1):
        if hit.path in relevant:
            return position
    return None


def hit_at_k(hits: Sequence[Hit], relevant: AbstractSet[str], k: int) -> bool:
    """Whether any of the top k hits is relevant."""
    _require_k(k)
    return first_relevant_rank(hits[:k], relevant) is not None


def reciprocal_rank(hits: Sequence[Hit], relevant: AbstractSet[str]) -> float:
    """1 / rank of the first relevant hit, or 0.0 if none is relevant."""
    rank = first_relevant_rank(hits, relevant)
    return 0.0 if rank is None else 1 / rank


def recall_at_k(hits: Sequence[Hit], relevant: AbstractSet[str], k: int) -> float:
    """Share of the relevant paths that appear at least once in the top k hits.

    Unlike hit_at_k, finding one of two required pages scores 0.5, not a full hit.
    """
    _require_labels(relevant)
    _require_k(k)
    found = {hit.path for hit in hits[:k]} & relevant
    return len(found) / len(relevant)


def _require_labels(relevant: AbstractSet[str]) -> None:
    # A question with no labels is a broken test case; scoring it 0 would look like a
    # retrieval miss. The type is a set so a bare str (a "collection" of characters) can't
    # slip in and turn `in` into a substring match.
    if not relevant:
        raise ValueError("relevant is empty: every question needs at least one labelled path")


def _require_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
