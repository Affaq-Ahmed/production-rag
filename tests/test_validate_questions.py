import json
from pathlib import Path

import pytest
from validate_questions import EXPECTED_COUNTS, main, summary, validate

KINDS = [kind for kind, n in EXPECTED_COUNTS.items() for _ in range(n)]


def _paths(n: int, kind: str) -> list[str]:
    return [f"topic/q{n:02d}-{s}.md" for s in ("ab" if kind == "multi" else "a")]


def _phrase(path: str) -> str:
    return f"the fact behind {path}"


def _page(corpus: Path, path: str, body: str, front_matter: str = "title: Page") -> None:
    file = corpus / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"---\n{front_matter}\n---\n{body}", encoding="utf-8")


def _rows() -> list[dict[str, object]]:
    """A valid 7 / 7 / 6 set whose evidence is on the pages the corpus fixture writes."""
    return [
        {
            "id": f"q{n:02d}",
            "type": kind,
            "question": f"What is question {n} asking about?",
            "relevant_paths": _paths(n, kind),
            "evidence": {path: _phrase(path) for path in _paths(n, kind)},
        }
        for n, kind in enumerate(KINDS, start=1)
    ]


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _errors(corpus: Path, rows: list[dict[str, object]]) -> list[str]:
    return validate(_jsonl(rows), corpus).errors


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for n, kind in enumerate(KINDS, start=1):
        for path in _paths(n, kind):
            _page(tmp_path, path, f"Intro.\n\nThis page states {_phrase(path)} plainly.")
    return tmp_path


def test_valid_file_passes_and_summarises(corpus: Path) -> None:
    result = validate(_jsonl(_rows()) + "\n", corpus)
    assert result.errors == []
    assert summary(result) == [
        "20 questions: 7 identifier, 7 paraphrase, 6 multi",
        "26 relevant paths across 26 documents",
        "all evidence found",
        "OK",
    ]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("type", "identifer", "line 1: type: Input should be"),
        ("id", "q1", "line 1: id: String should match pattern"),
        ("question", "Why?", "line 1: question: String should have at least 10 characters"),
        ("relevant_paths", [], "line 1: relevant_paths: List should have at least 1 item"),
        ("tags", ["x"], "line 1: tags: Extra inputs are not permitted"),
    ],
)
def test_invalid_field_is_reported_with_its_line_number(
    corpus: Path, field: str, value: object, expected: str
) -> None:
    rows = _rows()
    rows[0][field] = value
    assert any(error.startswith(expected) for error in _errors(corpus, rows))


def test_missing_field_and_broken_json_are_reported(corpus: Path) -> None:
    rows = _rows()
    del rows[0]["evidence"]
    text = _jsonl(rows) + '{"id": "q21",\n'
    errors = validate(text, corpus).errors
    assert "line 1: evidence: Field required" in errors
    assert any(error.startswith("line 21: Invalid JSON") for error in errors)


def test_duplicate_id_is_reported(corpus: Path) -> None:
    rows = _rows()
    rows[1]["id"] = "q01"
    assert "q01: id used 2 times" in _errors(corpus, rows)


def test_type_mix_drift_is_reported(corpus: Path) -> None:
    rows = _rows()
    rows[0]["type"] = "paraphrase"
    assert _errors(corpus, rows) == [
        "type mix is 6 identifier, 8 paraphrase, 6 multi; "
        "expected 7 identifier, 7 paraphrase, 6 multi"
    ]


@pytest.mark.parametrize("path", ["topic/missing.md", "../outside.md", "/etc/hosts"])
def test_missing_page_is_reported(corpus: Path, path: str) -> None:
    rows = _rows()
    rows[0]["relevant_paths"] = [path]
    rows[0]["evidence"] = {path: "anything"}
    assert _errors(corpus, rows) == [f"q01: {path} does not exist under {corpus}"]


def test_page_that_load_skips_is_reported(corpus: Path) -> None:
    _page(corpus, "topic/untitled.md", "Body text.", front_matter="weight: 10")
    rows = _rows()
    rows[0]["relevant_paths"] = ["topic/untitled.md"]
    rows[0]["evidence"] = {"topic/untitled.md": "Body text."}
    assert _errors(corpus, rows) == ["q01: topic/untitled.md exists but load() skips it"]


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({}, "q01: no evidence for topic/q01-a.md"),
        (
            {"topic/q01-a.md": _phrase("topic/q01-a.md"), "topic/q02-a.md": "x"},
            "q01: evidence for unlabelled page topic/q02-a.md",
        ),
    ],
)
def test_evidence_keys_must_equal_relevant_paths(
    corpus: Path, evidence: dict[str, str], expected: str
) -> None:
    rows = _rows()
    rows[0]["evidence"] = evidence
    assert _errors(corpus, rows) == [expected]


def test_multi_question_needs_two_distinct_paths(corpus: Path) -> None:
    rows = _rows()
    path = "topic/q15-a.md"
    rows[14]["relevant_paths"] = [path, path]
    rows[14]["evidence"] = {path: _phrase(path)}
    assert _errors(corpus, rows) == [
        f"q15: relevant_paths repeats {path}",
        "q15: a multi question needs at least 2 relevant paths",
    ]


def test_identical_questions_are_reported_ignoring_case_and_spacing(corpus: Path) -> None:
    rows = _rows()
    rows[1]["question"] = "  WHAT is question 1   asking about?"
    assert _errors(corpus, rows) == ["q02: same question as q01"]


def test_phrase_not_on_page_is_reported(corpus: Path) -> None:
    rows = _rows()
    rows[0]["evidence"] = {"topic/q01-a.md": "defaults to 30 seconds"}
    assert _errors(corpus, rows) == [
        "q01: evidence for topic/q01-a.md not found: 'defaults to 30 seconds'"
    ]


def test_phrase_only_in_hugo_markup_fails_after_cleaning(corpus: Path) -> None:
    body = 'Data lives in {{< glossary_tooltip text="etcd" term_id="etcd-store" >}}.'
    _page(corpus, "topic/q01-a.md", body)
    rows = _rows()
    rows[0]["evidence"] = {"topic/q01-a.md": "etcd-store"}
    [error] = _errors(corpus, rows)
    assert error.startswith("q01: evidence for topic/q01-a.md is in the raw page but not after")


def test_phrase_wrapped_across_lines_is_found(corpus: Path) -> None:
    _page(corpus, "topic/q01-a.md", "This page states the fact\nbehind topic/q01-a.md plainly.")
    assert _errors(corpus, _rows()) == []


def test_empty_phrase_is_reported(corpus: Path) -> None:
    rows = _rows()
    rows[0]["evidence"] = {"topic/q01-a.md": "  "}
    result = validate(_jsonl(rows), corpus)
    assert result.errors == ["q01: empty evidence for topic/q01-a.md"]
    assert "all evidence found" not in summary(result)


def test_cli_exit_status_without_settings(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good, bad = tmp_path / "good.jsonl", tmp_path / "bad.jsonl"
    good.write_text(_jsonl(_rows()), encoding="utf-8")
    bad.write_text(_jsonl(_rows()[1:]), encoding="utf-8")

    assert main(["--questions", str(good), "--corpus-dir", str(corpus)]) == 0
    assert capsys.readouterr().out.endswith("OK\n")
    assert main(["--questions", str(bad), "--corpus-dir", str(corpus)]) == 1
    assert "FAILED: 1 problem(s)" in capsys.readouterr().out
