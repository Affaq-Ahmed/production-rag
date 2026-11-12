import pytest

from rag.ingest.models import Section
from rag.ingest.sections import split_sections


def test_nested_headings_and_reset() -> None:
    assert split_sections("Title", "## A\nA text\n### B\nB text\n## C\nC text") == [
        Section(("Title", "A"), "A text"),
        Section(("Title", "A", "B"), "B text"),
        Section(("Title", "C"), "C text"),
    ]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_code_comments_are_not_headings(fence: str) -> None:
    code = f"{fence}shell\n# Check the pod status\nkubectl get pods\n{fence}"
    assert split_sections("Title", f"## A\n{code}\n## B\nAfter") == [
        Section(("Title", "A"), code),
        Section(("Title", "B"), "After"),
    ]


def test_intro_uses_page_title() -> None:
    assert split_sections("Title", "Intro\n\n## A\nBody") == [
        Section(("Title",), "Intro"),
        Section(("Title", "A"), "Body"),
    ]


def test_empty_sections_are_dropped() -> None:
    assert split_sections("Title", "\n## A\n\n### B\nText\n## C\n ") == [
        Section(("Title", "A", "B"), "Text"),
    ]
    assert split_sections("Title", " \n") == []


def test_heading_levels_and_required_space() -> None:
    assert split_sections("Title", "# A\n#### B\n##### Deep\n#nospace") == [
        Section(("Title", "A", "B"), "##### Deep\n#nospace"),
    ]
