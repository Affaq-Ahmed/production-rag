"""Ingest the pinned Kubernetes documentation corpus."""

import argparse
import logging
import math
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import psycopg

from rag.config import get_settings
from rag.ingest.chunk import MAX_TOKENS, chunk_fixed, chunk_structural
from rag.ingest.clean import clean
from rag.ingest.load import load
from rag.ingest.sections import split_sections
from rag.ingest.store import store

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def corpus_commit(corpus_dir: Path, manifest: Path) -> str:
    match = re.search(r"^- Commit:\s*`([0-9a-f]{40})`", manifest.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError("CORPUS.md must contain a full pinned commit hash")
    commit = subprocess.run(
        ["git", "-C", str(corpus_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != match[1]:
        raise ValueError(f"Corpus commit mismatch: checkout {commit}, CORPUS.md {match[1]}")
    return commit


def positive_limit(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed", "structural"], required=True)
    parser.add_argument("--limit", type=positive_limit)
    parser.add_argument(
        "--rebuild", action="store_true", help="Delete this strategy's chunks first"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    settings = get_settings()
    corpus_dir = Path(settings.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = PROJECT_ROOT / corpus_dir
    if not corpus_dir.is_dir():
        parser.error(f"Corpus directory does not exist: {corpus_dir}")
    try:
        commit = corpus_commit(corpus_dir, PROJECT_ROOT / "CORPUS.md")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    files = sorted(corpus_dir.rglob("*.md"))
    if args.limit is not None:
        files = files[: args.limit]
    counts: Counter[str] = Counter()
    paths: list[str] = []
    with psycopg.connect(str(settings.database_url), autocommit=True) as conn:
        if args.rebuild:
            with conn.transaction():
                conn.execute("delete from chunks where strategy = %s", (args.strategy,))
        for file in files:
            doc = load(corpus_dir, file)
            if doc is None:
                counts["skipped"] += 1
                continue
            cleaned = clean(doc.body)
            chunks = (
                chunk_fixed(doc.title, cleaned)
                if args.strategy == "fixed"
                else chunk_structural(split_sections(doc.title, cleaned))
            )
            if not chunks:
                logger.warning("Skipping page with no chunks: %s", doc.path)
                counts["skipped"] += 1
                continue
            counts[store(conn, doc, cleaned, chunks, commit)] += 1
            paths.append(doc.path)
        # Report persisted chunks for the files in this run, including unchanged documents.
        rows = conn.execute(
            "select c.token_count from chunks c join documents d on d.id = c.document_id "
            "where c.strategy = %s and d.path = any(%s) order by c.token_count",
            (args.strategy, paths),
        ).fetchall()
    tokens = [row[0] for row in rows]
    print(f"strategy:    {args.strategy}")
    print(
        "documents:   "
        + "  ".join(
            f"{name} {counts[name]}" for name in ("inserted", "updated", "unchanged", "skipped")
        )
    )
    print(f"chunks:      {len(tokens):,}")
    if tokens:
        print(
            f"tokens:      min {tokens[0]}  median {statistics.median(tokens):g}"
            f"  p95 {tokens[math.ceil(len(tokens) * 0.95) - 1]}  max {tokens[-1]}"
        )
    else:
        print("tokens:      min n/a  median n/a  p95 n/a  max n/a")
    print(f"over limit:  {sum(t > MAX_TOKENS for t in tokens)}")


if __name__ == "__main__":
    main()
