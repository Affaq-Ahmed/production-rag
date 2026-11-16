# AGENTS.md — production-rag

Instructions for AI coding agents (Cursor, Claude Code, Codex) working in this repository.
Read this whole file before making changes. If anything here conflicts with a direct request from
the user, follow the user and mention the conflict.

---

## 1. What this project is

A retrieval-augmented generation (RAG) system over the English Kubernetes documentation, built to
**measure** how retrieval quality changes across configurations:

| Config | Search | Rerank | Chunking |
|---|---|---|---|
| 1 | vector only | off | fixed |
| 2 | hybrid (vector + Postgres full-text, fused with RRF) | off | fixed |
| 3 | hybrid | cross-encoder | fixed |
| 4 | hybrid | cross-encoder | structural |

The deliverable is the comparison table and the evidence behind it, not just working code. Every
number in the README must come from a committed results file.

## 2. Stack

- Python 3.12, managed with **uv** (never `pip install`)
- Postgres on Supabase with **pgvector**; `psycopg` 3 for access
- OpenAI `text-embedding-3-small` (1,536 dimensions, `cl100k_base` tokenizer)
- `sentence-transformers` cross-encoder for reranking (added in Step 8)
- FastAPI for the retrieval endpoint
- `pytest`, `mypy --strict`, `ruff`

## 3. Commands

```bash
uv sync                                   # install
uv run pytest                             # tests
uv run mypy                               # type check (strict)
uv run ruff check . && uv run ruff format --check .
uv run python scripts/migrate.py          # apply pending migrations
./scripts/fetch_corpus.sh                 # fetch the corpus into data/ (pinned commit in CORPUS.md)
uv run python -m rag.ingest --strategy fixed|structural [--limit N] [--rebuild]
uv run python -m rag.embed [--strategy fixed|structural] [--limit N]
```

**Before saying a task is done, run all four checks:** `pytest`, `mypy`, `ruff check`,
`ruff format --check`. Report their output. Don't claim success from code inspection alone.

## 4. Layout

```
src/rag/
  config.py        # the ONLY place that reads environment variables
  ingest/          # load → clean → sections → chunk → store
  embed.py         # batched, resumable embedding
  retrieve/        # vector, keyword, fuse (RRF), rerank
migrations/        # numbered SQL, applied by scripts/migrate.py
eval/
  questions.jsonl  # the test set — human-written
  results/         # committed evaluation outputs
scripts/           # migrate, fetch_corpus, check_recall
tests/             # mirrors src/
data/              # corpus checkout — git-ignored, never commit
```

## 5. Architecture rules

- **Pure functions first.** Loading, cleaning, sectioning, chunking, batching, fusion and scoring
  are pure: data in, data out, no database, no network. Only `store.py`, `embed.py` and the
  retrieval query functions touch the database.
- **Settings only through `rag.config.get_settings()`.** Never call `os.environ` or `os.getenv`
  anywhere else.
- **Secrets are `SecretStr`.** Never log, print or put an API key or connection string in an
  exception message.
- **Inject external clients.** Functions that call OpenAI or a model take the client as a parameter,
  so tests can pass a fake.
- **Dataclasses in `ingest/models.py` are the contracts between stages.** Change them deliberately,
  and update every stage and test that uses them.
- **One transaction per document** when writing ingestion results. Use `executemany` for batches.

## 6. Code style

- Type hints on everything; the code must pass `mypy --strict`. No `Any` unless unavoidable, and
  then with a comment saying why.
- No `# type: ignore` without an error code and a reason: `# type: ignore[call-arg]  # pydantic env`.
- Small functions with one job. Prefer returning values to mutating arguments.
- Comments explain **why**, not what. No commented-out code.
- Don't add dependencies without asking. If one is needed, say which and why, and add it with
  `uv add`.

## 7. Database rules

- **Never edit a migration that has already been applied.** Add a new numbered file
  (`003_…sql`) instead.
- Every schema change goes through a migration. Never change the schema by hand in the dashboard.
- The `chunks.tsv` column is generated. Never write to it.
- `embedding` and `embedding_model` are always set together (a check constraint enforces it).
- Vector indexes are **partial, one per strategy**. In vector queries, put the strategy into the SQL
  as a literal with `psycopg.sql.Literal` (it's a fixed two-value set, so this is safe). A `%s`
  parameter can stop Postgres from using the partial index.
- Use the Supabase **session pooler** (port 5432), never the transaction pooler (6543).
- Call `register_vector(conn)` after connecting, and pass embeddings as `pgvector.Vector(values)`.
  A plain Python list is sent as a Postgres array, not a vector. numpy is not a dependency.

## 8. Domain gotchas — read before touching ingestion or retrieval

- **Hugo shortcodes.** Pages contain `{{< … >}}` and `{{% … %}}`. Cleaning rules are ordered:
  specific rules (glossary tooltips, notes, headings, feature-state) run **before** the catch-all,
  and images are removed **before** links. No cleaned chunk may contain `{{`.
- **`#` inside a code fence is a comment, not a heading.** Section splitting must track fence state.
- **Keep code blocks.** They contain `kubectl` commands and YAML field names, the identifiers
  keyword search exists for.
- **Embed `heading_path + "\n\n" + text`** via `embedding_input()`, never the text alone. Queries
  must be embedded consistently.
- **Sort embedding responses by `index`** before pairing vectors with chunk IDs.
- **Postgres full-text search treats camelCase identifiers as one token.** A search for
  `grace period` won't match `terminationGracePeriodSeconds`. This is expected, not a bug to hide.
- **`ts_rank_cd` is BM25-like, not BM25.** Never describe it as BM25.
- **Never add raw scores across search methods.** Cosine distance and text rank aren't on the same
  scale. Fuse by rank with RRF.

## 9. Testing rules

- **Tests never call a paid API or need network access.** Use fake clients and recorded fixtures.
- **The suite must pass without the corpus.** Tests that read real pages use a `corpus_dir` fixture
  that skips when `data/k8s-docs` is missing.
- **The suite must pass without API keys.** Tests construct `Settings(_env_file=None)` with values
  set through `monkeypatch`.
- **Behaviour that protects data is tested in the default suite.** Use a fake connection that
  records `execute`/`executemany` calls rather than relying on a database test that skips without
  `TEST_DATABASE_URL`. Database-backed tests are extra integration checks, never the only coverage.
- Write a test for each bug before fixing it.
- Test chunk overlap on **tokens**, not decoded strings.
- Name tests after the behaviour they check: `test_tooltip_mid_sentence_keeps_word`.

## 10. Cost and safety

- **Ask before running anything that calls a paid API**, including `rag.embed` without `--limit`
  and any evaluation run that generates answers. State the expected cost.
- Prefer `--limit` runs to check behaviour before full runs.
- Never run destructive SQL (`drop`, `truncate`, an unscoped `delete`) without asking first.
- Never commit `.env`, anything under `data/`, or API keys. Check `git status` before committing.

## 11. Evaluation integrity

- **`eval/questions.jsonl` is written by the human.** Don't generate, edit or "improve" test
  questions or their relevance labels.
- Relevance is labelled by **document path**, never chunk ID, so re-chunking doesn't break labels.
- Never tune a parameter (chunk size, `k`, RRF constant) by repeatedly running the eval until the
  numbers look good. If tuning happens, record it in the README.
- Retrieval is evaluated **without** an LLM in Week 3. Don't add generation to the eval path.
- Results go to `eval/results/<config>.json` and are committed. Never type a number into the README
  that isn't in a results file.

## 12. Learning mode

This repository is part of a learning programme. The human writes the core logic so they
understand it. In these files, **don't write the implementation unless the human explicitly asks
you to**:

- `src/rag/ingest/clean.py`, `sections.py`, `chunk.py`
- `src/rag/embed.py`
- everything in `src/rag/retrieve/`
- `eval/run.py`

In these files, do help by:
- explaining a concept or an API
- pointing out a bug and why it's a bug, with a hint rather than the fix
- reviewing code the human wrote
- writing tests the human asks for

Boilerplate is fine to write in full: configuration, CLI argument parsing, migration SQL, scripts,
`__init__.py` files and test fixtures.

## 13. How to work

- Read the relevant files before changing them. Don't guess at existing APIs.
- Make the smallest change that solves the task. Don't refactor unrelated code.
- When a request is ambiguous, ask one question rather than guessing.
- Explain non-obvious decisions in a sentence or two, especially trade-offs.
- If you hit an error you can't explain, say so and show the output, rather than retrying blindly.
- Keep `NOTES.md` in mind: when you observe surprising behaviour from Postgres, pgvector, the
  embeddings API or the corpus, suggest a line for it.

## 14. Git

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`.
- One logical change per commit. Run the four checks in §3 before committing.
- Commit only when the human asks you to.
- **Do not add co-author trailers or any AI attribution to commits.** No `Co-Authored-By:` lines,
  no "Generated with …" lines, no agent signatures, in commit messages or pull request descriptions.
  Commits are authored by the human alone.