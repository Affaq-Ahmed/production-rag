import logging
from pathlib import Path

import pytest

from rag.ingest.load import load


@pytest.mark.parametrize(
    ("relative_path", "url_path"),
    [
        ("concepts/workloads/pods/pod-lifecycle.md", "concepts/workloads/pods/pod-lifecycle/"),
        ("concepts/workloads/_index.md", "concepts/workloads/"),
        ("concepts/workloads/index.md", "concepts/workloads/"),
        ("_index.md", ""),
    ],
)
def test_load_maps_url(tmp_path: Path, relative_path: str, url_path: str) -> None:
    file = tmp_path / relative_path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("---\ntitle: Pod Lifecycle\ncontent_type: concept\n---\nBody.")

    doc = load(tmp_path, file)

    assert doc is not None
    assert doc.path == relative_path
    assert doc.source_url == "https://kubernetes.io/docs/" + url_path
    assert doc.title == "Pod Lifecycle"
    assert doc.content_type == "concept"
    assert doc.body == "Body."


def test_load_optional_content_type(tmp_path: Path) -> None:
    file = tmp_path / "page.md"
    file.write_text("---\ntitle: Test\n---\nBody.")

    doc = load(tmp_path, file)

    assert doc is not None
    assert doc.content_type is None


def test_load_missing_title(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    file = tmp_path / "page.md"
    file.write_text("Body without front matter.")

    with caplog.at_level(logging.WARNING):
        assert load(tmp_path, file) is None

    assert "page.md" in caplog.text
    assert "without a title" in caplog.text


def test_load_bad_front_matter(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    file = tmp_path / "broken.md"
    file.write_text("---\ntitle: [unterminated\n---\nBody.")

    assert load(tmp_path, file) is None
    assert "broken.md" in caplog.text
