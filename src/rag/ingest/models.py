from dataclasses import dataclass
from typing import Literal

Strategy = Literal["fixed", "structural"]


@dataclass(frozen=True)
class RawDoc:
    path: str  # relative to corpus_dir: "concepts/workloads/pods/pod-lifecycle.md"
    source_url: str  # "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"
    title: str
    content_type: str | None
    body: str  # Markdown after front matter, before cleaning


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, ...]  # ("Pod Lifecycle", "Container probes", "Types of probe")
    text: str


@dataclass(frozen=True)
class Chunk:
    strategy: Strategy
    ord: int
    heading_path: str  # joined with a spaced U+203A separator
    text: str
    token_count: int
