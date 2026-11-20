"""The result shape every search stage returns, so the evaluator never needs to know which ran."""

from dataclasses import dataclass
from typing import Literal

Source = Literal["vector", "keyword", "rrf", "rerank"]


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk at its position in one stage's result list.

    `score` keeps the producing stage's own scale and direction: for vector it is cosine
    distance (1 - similarity, so 0 is identical and lower is better), while text rank is higher
    is better. Scores are never compared or added across sources; stages combine by `rank`.
    """

    chunk_id: int
    path: str  # documents.path, which is what the questions are labelled with
    heading_path: str
    text: str
    score: float  # see the class docstring for direction
    rank: int  # 1-based position in this result list
    source: Source
