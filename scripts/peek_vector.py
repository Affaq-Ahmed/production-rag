"""First look at vector search: every question, both strategies, k = 10 (Step 6.9).

uv run python scripts/peek_vector.py [--show-misses]

Not the evaluation (that is eval/run.py, Step 9), and not for tuning: nothing seen here may
change the chunker, k or the questions. Query vectors come from eval/cache, so a question set
that is already cached makes no API calls.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAI
from pgvector.psycopg import register_vector

from rag.config import get_settings
from rag.db import connect
from rag.ingest.models import Strategy
from rag.retrieve.models import Hit
from rag.retrieve.vector import QueryEmbeddingCache, check_embedding_model, search_vector

# eval/ is scripts, not a package; pytest reaches it through pythonpath for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from metrics import first_relevant_rank, hit_at_k, recall_at_k, reciprocal_rank
from validate_questions import QUESTIONS_PATH, Question, parse

K = 10
STRATEGIES: tuple[Strategy, ...] = ("fixed", "structural")
Results = dict[tuple[str, Strategy], list[Hit]]


def run(questions: Sequence[Question]) -> Results:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    cache = QueryEmbeddingCache()
    results: Results = {}
    with connect(autocommit=True) as conn:
        register_vector(conn)
        for strategy in STRATEGIES:
            check_embedding_model(conn, strategy, settings.embedding_model)
            for q in questions:
                found = search_vector(
                    conn,
                    client,
                    q.question,
                    strategy,
                    K,
                    cache=cache,
                    model=settings.embedding_model,
                    dims=settings.embedding_dims,
                )
                results[q.id, strategy] = found.hits
    return results


def question_line(q: Question, results: Results) -> str:
    relevant = set(q.relevant_paths)
    cells = []
    for strategy in STRATEGIES:
        rank = first_relevant_rank(results[q.id, strategy], relevant)
        cells.append(f"{strategy:<10} {'—' if rank is None else f'rank {rank}':<8}")
    line = f"{q.id} {q.type:<11} " + "  ".join(cells)
    if q.type == "multi":
        recalls = (f"{recall_at_k(results[q.id, s], relevant, K):.2f}" for s in STRATEGIES)
        line += f"  recall@{K} " + " / ".join(recalls)
    return line


def summary_lines(questions: Sequence[Question], results: Results, label: str) -> list[str]:
    n = len(questions)

    def hits(strategy: Strategy, k: int) -> str:
        count = sum(hit_at_k(results[q.id, strategy], set(q.relevant_paths), k) for q in questions)
        return f"{strategy} {count}/{n}"

    def mrr(strategy: Strategy) -> str:
        total = sum(
            reciprocal_rank(results[q.id, strategy], set(q.relevant_paths)) for q in questions
        )
        return f"{strategy} {total / n:.2f}"

    return [
        f"{label:<11} hit@1   " + "   ".join(hits(s, 1) for s in STRATEGIES),
        f"{label:<11} hit@5   " + "   ".join(hits(s, 5) for s in STRATEGIES),
        f"{label:<11} hit@{K}  " + "   ".join(hits(s, K) for s in STRATEGIES),
        f"{label:<11} MRR     " + "   ".join(mrr(s) for s in STRATEGIES),
    ]


def miss_lines(q: Question, results: Results) -> list[str]:
    relevant = set(q.relevant_paths)
    lines: list[str] = []
    for strategy in STRATEGIES:
        hits = results[q.id, strategy]
        if hit_at_k(hits, relevant, K):
            continue
        lines.append(f"\n== {q.id} {q.type} MISS ({strategy}): {q.question}")
        lines.append(f"   wanted: {', '.join(q.relevant_paths)}")
        for hit in hits:
            lines.append(f"   {hit.rank:>2}  {hit.score:.4f}  {hit.path}  |  {hit.heading_path}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-misses", action="store_true", help=f"print the top {K} for every miss at {K}"
    )
    args = parser.parse_args()
    questions, errors = parse(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if errors:
        sys.exit("eval/questions.jsonl does not parse; run eval/validate_questions.py")

    results = run(questions)
    for q in questions:
        print(question_line(q, results))
    print()
    print("\n".join(summary_lines(questions, results, "all")))
    for kind in ("identifier", "paraphrase", "multi"):
        subset = [q for q in questions if q.type == kind]
        print("\n".join(summary_lines(subset, results, f"{kind} ({len(subset)})")))
    if args.show_misses:
        for q in questions:
            print("\n".join(miss_lines(q, results)), end="")
        print()


if __name__ == "__main__":
    main()
