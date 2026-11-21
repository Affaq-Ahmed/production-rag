"""Retrieval metrics, checked against hand-worked examples."""

import pytest
from metrics import first_relevant_rank, hit_at_k, recall_at_k, reciprocal_rank

from rag.retrieve.models import Hit


def _hits(*paths: str) -> list[Hit]:
    return [
        Hit(chunk_id=i, path=path, heading_path="H", text="t", score=0.0, rank=i, source="vector")
        for i, path in enumerate(paths, start=1)
    ]


# The guide's table, plus the first relevant rank each row implies.
@pytest.mark.parametrize(
    ("paths", "relevant", "first", "hit1", "hit3", "rr", "recall3"),
    [
        (("A", "B", "C"), {"B"}, 2, False, True, 0.5, 1.0),
        (("A", "A", "A"), {"B"}, None, False, False, 0.0, 0.0),
        (("B", "A", "C"), {"B", "C"}, 1, True, True, 1.0, 1.0),
        (("B", "B", "A"), {"B", "C"}, 1, True, True, 1.0, 0.5),
    ],
)
def test_metrics_match_the_hand_worked_table(
    paths: tuple[str, ...],
    relevant: set[str],
    first: int | None,
    hit1: bool,
    hit3: bool,
    rr: float,
    recall3: float,
) -> None:
    hits = _hits(*paths)
    assert first_relevant_rank(hits, relevant) == first
    assert hit_at_k(hits, relevant, 1) is hit1
    assert hit_at_k(hits, relevant, 3) is hit3
    assert reciprocal_rank(hits, relevant) == rr
    assert recall_at_k(hits, relevant, 3) == recall3


def test_k_counts_chunks_so_one_page_can_fill_the_top_k() -> None:
    hits = _hits("A", "A", "A", "B")
    assert not hit_at_k(hits, {"B"}, 3)
    assert hit_at_k(hits, {"B"}, 4)
    assert first_relevant_rank(hits, {"B"}) == 4
    assert reciprocal_rank(hits, {"B"}) == 0.25


def test_recall_counts_only_pages_inside_the_top_k() -> None:
    hits = _hits("A", "B", "C", "D")
    assert recall_at_k(hits, {"B", "D"}, 3) == 0.5
    assert recall_at_k(hits, {"B", "D"}, 4) == 1.0


def test_half_a_multi_answer_is_a_full_hit_but_half_recall() -> None:
    hits = _hits("configmap.md", "configmap.md", "pods.md")
    relevant = {"configmap.md", "secret.md"}
    assert hit_at_k(hits, relevant, 3)
    assert recall_at_k(hits, relevant, 3) == 0.5


def test_fewer_hits_than_k_are_scored_as_they_are() -> None:
    hits = _hits("A", "B")
    assert hit_at_k(hits, {"B"}, 10)
    assert recall_at_k(hits, {"B", "C"}, 10) == 0.5


def test_no_hits_score_zero() -> None:
    assert first_relevant_rank([], {"B"}) is None
    assert not hit_at_k([], {"B"}, 5)
    assert reciprocal_rank([], {"B"}) == 0.0
    assert recall_at_k([], {"B"}, 5) == 0.0


def test_rank_is_the_list_position_not_the_hit_rank_field() -> None:
    hits = list(reversed(_hits("B", "A")))  # positions: A then B, rank fields 2 then 1
    assert first_relevant_rank(hits, {"B"}) == 2


def test_empty_labels_are_refused_rather_than_scored_as_a_miss() -> None:
    hits = _hits("A")
    for score in (
        lambda: first_relevant_rank(hits, set()),
        lambda: hit_at_k(hits, set(), 1),
        lambda: reciprocal_rank(hits, set()),
        lambda: recall_at_k(hits, set(), 1),
    ):
        with pytest.raises(ValueError, match="relevant is empty"):
            score()


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_refused(k: int) -> None:
    hits = _hits("A")
    with pytest.raises(ValueError, match="k must be at least 1"):
        hit_at_k(hits, {"A"}, k)
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k(hits, {"A"}, k)
