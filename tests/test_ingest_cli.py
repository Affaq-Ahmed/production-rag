import argparse
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.ingest.__main__ import corpus_commit, positive_limit


@pytest.mark.parametrize("matches", [True, False])
def test_corpus_pin(tmp_path: Path, matches: bool) -> None:
    manifest = tmp_path / "CORPUS.md"
    manifest.write_text("- Commit: `" + "a" * 40 + "`\n")
    actual = ("a" if matches else "b") * 40
    with patch(
        "rag.ingest.__main__.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout=actual + "\n"),
    ):
        if matches:
            assert corpus_commit(tmp_path, manifest) == actual
        else:
            with pytest.raises(ValueError, match="mismatch"):
                corpus_commit(tmp_path, manifest)


def test_missing_pin(tmp_path: Path) -> None:
    manifest = tmp_path / "CORPUS.md"
    manifest.write_text("- Commit: `<hash>`")
    with pytest.raises(ValueError, match="full pinned"):
        corpus_commit(tmp_path, manifest)


def test_limit() -> None:
    assert positive_limit("10") == 10
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        positive_limit("0")
