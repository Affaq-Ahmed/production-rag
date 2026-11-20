"""Check eval/questions.jsonl: format, type mix, labels, and evidence on the cleaned pages.

This checks the file, not the quality of the questions. Exit status 0 means every check passed.

    uv run python eval/validate_questions.py [--questions FILE] [--corpus-dir DIR]
"""

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.config import get_settings
from rag.ingest.clean import clean
from rag.ingest.load import load
from rag.ingest.models import RawDoc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
EXPECTED_COUNTS = {"identifier": 7, "paraphrase": 7, "multi": 6}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q\d{2}$")
    type: Literal["identifier", "paraphrase", "multi"]
    question: str = Field(min_length=10)
    relevant_paths: list[str] = Field(min_length=1)
    evidence: dict[str, str]  # path -> phrase, one entry per relevant path
    notes: str | None = None


@dataclass(frozen=True)
class Result:
    questions: list[Question]
    errors: list[str]  # every problem, evidence problems included
    evidence_errors: list[str]


def parse(text: str) -> tuple[list[Question], list[str]]:
    questions: list[Question] = []
    errors: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(Question.model_validate_json(line))
        except ValidationError as error:
            for detail in error.errors():
                field = ".".join(str(part) for part in detail["loc"])
                errors.append(f"line {number}: {field + ': ' if field else ''}{detail['msg']}")
    return questions, errors


def check_ids(questions: list[Question]) -> list[str]:
    counts = Counter(q.id for q in questions)
    return [f"{qid}: id used {n} times" for qid, n in sorted(counts.items()) if n > 1]


def check_mix(questions: list[Question]) -> list[str]:
    counts: Counter[str] = Counter(q.type for q in questions)
    actual = {kind: counts.get(kind, 0) for kind in EXPECTED_COUNTS}
    if actual == EXPECTED_COUNTS:
        return []
    return [f"type mix is {_mix(actual)}; expected {_mix(EXPECTED_COUNTS)}"]


def check_duplicates(questions: list[Question]) -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    for q in questions:
        key = _normalise(q.question).casefold()
        if key in seen:
            errors.append(f"{q.id}: same question as {seen[key]}")
        else:
            seen[key] = q.id
    return errors


def check_labels(q: Question) -> list[str]:
    errors: list[str] = []
    repeated = sorted(p for p, n in Counter(q.relevant_paths).items() if n > 1)
    if repeated:
        errors.append(f"{q.id}: relevant_paths repeats {', '.join(repeated)}")
    if q.type == "multi" and len(set(q.relevant_paths)) < 2:
        errors.append(f"{q.id}: a multi question needs at least 2 relevant paths")
    labelled = set(q.relevant_paths)
    errors += [f"{q.id}: no evidence for {p}" for p in sorted(labelled - q.evidence.keys())]
    errors += [
        f"{q.id}: evidence for unlabelled page {p}" for p in sorted(q.evidence.keys() - labelled)
    ]
    return errors


def load_pages(paths: Iterable[str], corpus_dir: Path) -> dict[str, RawDoc | None]:
    """Load each page once. None means the file is missing or load() skips it."""
    pages: dict[str, RawDoc | None] = {}
    for path in paths:
        if path in pages:
            continue
        file = _corpus_file(corpus_dir, path)
        pages[path] = load(corpus_dir, file) if file is not None else None
    return pages


def _corpus_file(corpus_dir: Path, path: str) -> Path | None:
    # An absolute path would replace corpus_dir in "corpus_dir / path" rather than join it.
    if Path(path).is_absolute() or ".." in Path(path).parts:
        return None
    file = corpus_dir / path
    return file if file.is_file() else None


def check_paths(
    questions: list[Question], pages: dict[str, RawDoc | None], corpus_dir: Path
) -> list[str]:
    errors: list[str] = []
    for q in questions:
        for path in dict.fromkeys(q.relevant_paths):
            if pages[path] is not None:
                continue
            if _corpus_file(corpus_dir, path) is not None:
                # load() skips it, so ingestion never stored it and retrieval can never return it.
                errors.append(f"{q.id}: {path} exists but load() skips it")
            else:
                errors.append(f"{q.id}: {path} does not exist under {corpus_dir}")
    return errors


def check_evidence(questions: list[Question], pages: dict[str, RawDoc | None]) -> list[str]:
    errors: list[str] = []
    cleaned: dict[str, str] = {}
    for q in questions:
        for path in q.relevant_paths:
            phrase, doc = q.evidence.get(path), pages[path]
            if phrase is None or doc is None:
                continue  # already reported by check_labels or check_paths
            if not phrase.strip():
                errors.append(f"{q.id}: empty evidence for {path}")
                continue
            if path not in cleaned:
                cleaned[path] = _normalise(clean(doc.body))
            if _normalise(phrase) in cleaned[path]:
                continue
            if _normalise(phrase) in _normalise(doc.body):
                errors.append(
                    f"{q.id}: evidence for {path} is in the raw page but not after clean() "
                    f"(Hugo markup, or the cleaner is eating text): {phrase!r}"
                )
            else:
                errors.append(f"{q.id}: evidence for {path} not found: {phrase!r}")
    return errors


def validate(text: str, corpus_dir: Path) -> Result:
    questions, errors = parse(text)
    errors += check_ids(questions)
    errors += check_mix(questions)
    errors += check_duplicates(questions)
    for q in questions:
        errors += check_labels(q)
    pages = load_pages((p for q in questions for p in q.relevant_paths), corpus_dir)
    errors += check_paths(questions, pages, corpus_dir)
    evidence_errors = check_evidence(questions, pages)
    return Result(questions, errors + evidence_errors, evidence_errors)


def summary(result: Result) -> list[str]:
    counts: Counter[str] = Counter(q.type for q in result.questions)
    labels = [p for q in result.questions for p in q.relevant_paths]
    lines = [
        f"{len(result.questions)} questions: {_mix(counts)}",
        f"{len(labels)} relevant paths across {len(set(labels))} documents",
    ]
    if not result.evidence_errors:
        lines.append("all evidence found")
    if result.errors:
        lines += [f"  {error}" for error in result.errors]
        lines.append(f"FAILED: {len(result.errors)} problem(s)")
    else:
        lines.append("OK")
    return lines


def _mix(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{counts.get(kind, 0)} {kind}" for kind in EXPECTED_COUNTS)


def _normalise(text: str) -> str:
    # Markdown hard-wraps prose, so a phrase copied from the rendered page can span a line break.
    return " ".join(text.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument(
        "--corpus-dir", type=Path, help="Defaults to corpus_dir from settings (needs .env)"
    )
    args = parser.parse_args(argv)
    corpus_dir: Path = args.corpus_dir or Path(get_settings().corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = PROJECT_ROOT / corpus_dir
    if not corpus_dir.is_dir():
        parser.error(
            f"Corpus directory does not exist: {corpus_dir} (run ./scripts/fetch_corpus.sh)"
        )
    result = validate(args.questions.read_text(encoding="utf-8"), corpus_dir)
    print("\n".join(summary(result)))
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
