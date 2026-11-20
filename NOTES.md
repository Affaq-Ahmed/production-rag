# Notes

## PostgreSQL full-text search and code identifiers

Observed on 2026-11-11 using the `english` text configuration:

```sql
select to_tsvector('english', 'Set terminationGracePeriodSeconds on the pod. kubectl rollout undo. CrashLoopBackOff.');
```

Output:

```text
'crashloopbackoff':9 'kubectl':6 'pod':5 'rollout':7 'set':1 'terminationgraceperiodsecond':2 'undo':8
```

- `terminationGracePeriodSeconds` becomes the single lowercase, stemmed token
  `terminationgraceperiodsecond`; camelCase boundaries do not split it into words.
- `CrashLoopBackOff` becomes `crashloopbackoff`, and `kubectl` stays intact.
- Stop words `on` and `the` are omitted, but their positions still count.

This supports identifier lookups when the query uses the same text configuration.
However, a query for `grace period` will not match `terminationGracePeriodSeconds`
through that identifier alone: its component words are not separate indexed tokens.
Other words in the document could still provide a match.

When interpreting Step 9 results, distinguish full identifier queries from queries
using their component words. This tokenization behavior may explain keyword-search
misses; it does not by itself establish whether keyword or vector search performs
better overall.

## Step 3.10: reading the stored chunks

Inspected on 2026-11-12 against corpus commit
`64652743292622fd66cfaa75d290f89a18d5beab`. Ran the pod-lifecycle comparison,
five-random-structural-chunks query, and brace-count query against Supabase.
Also inspected code samples and the smallest structural chunks. Saved the sampled
output in `eval/results/chunk-inspection-2026-11-12.json`; the random sample will
not necessarily be the same on another run.

### Heading paths and chunk boundaries

- Pod Lifecycle has **25 fixed chunks** and **46 structural chunks**. Fixed paths
  are just `Pod Lifecycle`. Structural paths read like a table of contents, for
  example `Pod Lifecycle › Container states › Running` (with Markdown backticks
  and anchor attributes still present in the actual stored path).
- The inspected heading paths do not promote shell or YAML comments to headings.
  The `# container restart delays...` comments remain inside the YAML examples
  under `Configurable container restart delay`.
- Fixed chunks visibly cut prose and code: fixed chunk 2 begins `-level summary`,
  and chunk 3 begins with the trailing `2d9h` and closing fence from a command's
  output. Fixed chunk 6 ends halfway through a YAML comment at
  `# Container restarts regardless`.
- Structural samples mostly start and end at paragraph or code-block boundaries.
  The sampled kube-proxy command and DNS examples retain both fences and their
  contents. Pod Lifecycle's YAML examples for restart policies are intact in the
  inspected structural chunks. This is a sample check, not a guarantee that every
  code block is whole: blocks over 500 tokens deliberately use token windows.

### Five random structural samples

| Topic | Tokens | Observation |
| --- | ---: | --- |
| Logging Architecture: sidecar logging agent | 209 | Notes and example-file references remain readable. |
| Reconfiguring a kubeadm cluster: kube-proxy changes | 80 | Complete fenced `kubectl delete po` command. |
| Configuring kubelets using kubeadm | 113 | Identifiers and inline commands survive. |
| DNS for Services and Pods: A/AAAA records | 230 | Four complete fenced DNS examples. |
| StatefulSet Basics: rolling out a canary | 352 | Complete output and command blocks; valid Go-template syntax contains `{{`. |

### Shortcode check: failed, with an important distinction

```sql
select count(*) from chunks where text like '%{{%';
-- 127
```

- **127 chunks** contain `{{`: 69 fixed and 58 structural. These are chunk counts,
  not counts of unique shortcode occurrences; both strategies and overlapping
  fixed windows can repeat content.
- The broad query also matches valid code, such as
  `kubectl get pod web-1 --template '{{range $i, $c := .spec.containers}}{{$c.image}}{{end}}'`,
  Prometheus template annotations, and literal data. Removing all double braces
  would damage code examples.
- However, **62 chunks** contain actual Hugo delimiter strings `{{<` or `{{%`
  (33 fixed, 29 structural). Examples include `{{< param "version" >}}` in package
  repository URLs and `{{< skew currentPatchVersion >}}` in installation commands.
  There are also escaped shortcode demonstrations in `test.md`. The cleaner
  currently preserves fenced code verbatim, including Hugo tags within it.
- Therefore the requested zero-shortcode gate is **not met**. A follow-up must
  distinguish Hugo markup from legitimate code templates while preserving code
  blocks. A zero result for the broad `{{` query is not an appropriate target for
  this corpus because valid Go templates intentionally remain.

### Other surprises and follow-ups

- Glossary tooltips without a `text` attribute lose the term. The Pod Lifecycle
  introduction reads `If a  dies` where the source has
  `{{< glossary_tooltip term_id="node" >}}`. A fallback or glossary lookup is needed.
- Feature-state tags using only `feature_gate_name` become `Feature state:  since `.
  The explicit version/state case works, but this source variant needs handling
  without inventing a version or stability level.
- The source `## {{% heading "whatsnext" %}}` becomes a heading path ending in
  `## Whatsnext`, because the cleaner adds a second heading marker. Anchor
  attributes such as `{#pod-fault-recovery}` also remain in paths.
- The smallest structural chunk is **1 token**, containing only `---`, under
  `Volumes › Types of volumes {#volume-types} › hostPath {#hostpath} › hostPath configuration example`.
  Other small chunks remain at the end of branches or before a different parent;
  the tiny-sibling merge rule does not eliminate every tiny chunk.
- `test.md`, a documentation-format demonstration page, was ingested from the docs
  root. The CLI currently scans all Markdown under that root, rather than only the
  four intended sections (`concepts`, `tasks`, `tutorials`, `setup`). The ingestion
  file selection needs tightening before treating the dataset as section-only.

The inspection is complete, but the content-quality checks are not all passing.
These findings are recorded for follow-up; this inspection did not modify the
cleaner, chunker, or stored rows.

## Step 4.1: embedding input and the Step 6 query convention

Use `rag.embed.embedding_input(heading_path, text)` everywhere a chunk is
embedded. Its output is exactly the heading path, two newlines, and the chunk text;
it preserves the original text, including code and whitespace. Structural chunks
use the complete stored heading path; fixed chunks use only the page title.
This gives embeddings topic context even when the body does not name the topic.
The full-text `tsv` column also indexes the heading path and text.

Example input to the embedding model:

```text
Pod Lifecycle › Container probes › Types of probe

A readinessProbe indicates whether the container is ready to respond to requests.
```

For Step 6, embed the user's query as query text, using the same embedding model
and dimensions as the stored chunk vectors. Queries do not have a document
heading path: do not invent one, prepend a retrieved document's heading, or add
the chunk formatter's blank-line prefix to the query. Consistency means using
the same vector space while retaining this explicit distinction between document
context and query text. There is no embedding API call in Step 4.1; later chunk
embedding and re-embedding code must call the shared formatter.

## Step 4.3: embedding price check

Checked 2026-11-12 on OpenAI's pricing page
(<https://developers.openai.com/api/docs/pricing>, redirected from
`platform.openai.com/docs/pricing`):

| Model | Price | Dimensions |
|---|---|---|
| `text-embedding-3-small` | $0.02 per 1M input tokens (Standard tier) | 1,536 |

This matches the course's $0.02/1M assumption. The page lists no Batch API
price for embeddings, so this project uses the synchronous endpoint at the
standard rate. Cost formula for an embedding run:
`cost_usd = total_tokens / 1_000_000 * 0.02`, where `total_tokens` is the
sum of `response.usage.total_tokens` across all requests (not the local
tiktoken estimate). Re-check this price before any re-embedding run.

## Step 4: embedding run (2026-11-12)

Command: `uv run python -m rag.embed --strategy <s> [--limit N]`. It reads chunks
where `embedding is null`, formats each with `embedding_input()`, groups them with
`batches()` (≤ 100 inputs, ≤ 50,000 tokens), calls `text-embedding-3-small`
(`dimensions=1536`), sorts `response.data` by `index`, and writes each batch in
its own transaction. Re-running skips finished chunks, so an interrupted run resumes.

| Run | Chunks | Batches | Billed tokens | Cost (USD) | Wall time |
|---|---|---|---|---|---|
| trial: structural `--limit 20` | 20 | 1 | 1,986 | 0.000040 | 14 s |
| fixed | 2,272 | 23 | 1,046,373 | 0.020927 | 222 s |
| structural (remaining) | 4,753 | 48 | 983,271 | 0.019665 | 488 s |
| **Total** | **7,045** | **72** | **2,031,630** | **0.040633** | ~12 min |

Price: $0.02 / 1M tokens (checked 2026-11-12, see Step 4.3). Cost is from
`response.usage.total_tokens`, not an estimate.

Observations:
- The local tiktoken count of `embedding_input()` matches the billed tokens exactly
  (fixed 1,046,373; structural 985,257 = 1,986 + 983,271), so `batches()` counts
  tokens the way the API does.
- The course's "roughly $0.03" figure was low. The real cost was about $0.04, because
  the fixed chunks average about 460 tokens each.
- The largest batches came to 47,657 tokens, under the 50,000 cap. No single chunk
  was big enough to go in a batch alone.
- Every stored vector has 1,536 dimensions, and their lengths fall between 0.9994 and 1.0006.
  OpenAI returns unit-length vectors, so cosine distance and inner product give the same
  ranking.
- Batches took about 4–10 s each. I didn't time the API call and the database write
  separately. The write (100 rows of `vector(1536)` through the Supabase session pooler)
  is the likely cost; time it before optimising.

## Step 4.6: cost record and rerun (2026-11-12)

`eval/results/embedding_run.json` is the source for the README's embedding cost.
The top-level fields (`chunks`, `total_tokens`, `cost_usd`, `elapsed_s`) are totals;
`runs` lists each run that embedded something. `rag.embed` appends a run only when
it embedded at least one chunk, so a no-op rerun cannot overwrite the real cost with $0.00.

The three entries dated 2026-11-12 were written after the runs: I passed the numbers
from the CLI output through the same `merge_run()` function, because the code that
writes the report did not exist yet. The tokens are billed tokens from
`response.usage`, and they match the local tiktoken count exactly (Step 4 above).
Elapsed times in those entries include connecting to the database; runs recorded
from now on measure only the select/embed/write loop.

Rerun check: `uv run python -m rag.embed` with no flags → 0 chunks, 0 batches,
0 tokens, $0.000000, 1.6 s; the report was left unchanged.

## Step 4.7: HNSW indexes (2026-11-12)

`migrations/002_vector_indexes.sql` was applied after every chunk had an embedding
(2,272/2,272 fixed, 4,773/4,773 structural). It creates two partial
`hnsw (embedding vector_cosine_ops)` indexes, one per strategy:
`chunks_embedding_fixed_hnsw` (18 MB) and `chunks_embedding_structural_hnsw` (37 MB).

**The guide's `maintenance_work_mem = '256MB'` does not work on this Supabase instance.**
A parallel HNSW build (`max_parallel_maintenance_workers` is 1 here) reserves
`maintenance_work_mem` as one shared-memory segment up front:
- 256MB → `DiskFull: could not resize shared memory segment … No space left on device`
- 128MB (the guide's suggested fix) → `OutOfMemory: … Cannot allocate memory`

The fix was a serial build: `set local max_parallel_maintenance_workers = 0`, keeping
128MB. That allocates memory as it goes and finished in ~14 s for 7,045 vectors (~41 MB).
Both failed attempts rolled back cleanly (no index, no `schema_migrations` row), so the
migration file was edited before it was ever applied. Both settings use `set local`
so they last only for the migration's transaction.

**For Step 6: put the strategy into vector queries as a SQL literal** (`psycopg.sql.Literal`;
safe because `Strategy` has exactly two values), never as a `%s` parameter. I checked this
with `explain` on `order by embedding <=> … limit 10`:
- `where strategy = 'fixed'` → `Index Scan using chunks_embedding_fixed_hnsw`
- `where strategy = 'structural'` → `Index Scan using chunks_embedding_structural_hnsw`
- `where strategy = $1` (prepared, generic plan) → `Seq Scan on chunks` + `Sort`. Postgres
  can't prove `$1` matches either partial index's condition, so it computes the exact distance
  for every row. The results are still correct but slower, and it fails silently. psycopg
  sends `%s` as a server-side parameter, so this is the plan a `%s` strategy can get.

## Step 4.8: HNSW recall vs exact search (2026-11-12)

`uv run python scripts/check_recall.py` (defaults: 20 samples per strategy, k=10, seed 0).
Each sampled chunk's own embedding is the query. Indexed search = normal query with the
strategy as a literal; exact search = same query with `set local enable_indexscan = off`.
Before measuring, the script checks with `explain` that the indexed search uses
`chunks_embedding_<strategy>_hnsw` and the exact search uses no HNSW index, and it stops
if either is wrong.

| Strategy | Chunks | hnsw.ef_search | Mean recall@10 | Min | Perfect |
|---|---|---|---|---|---|
| fixed | 2,272 | 40 (default) | 1.000 | 1.00 | 20/20 |
| structural | 4,773 | 40 (default) | 1.000 | 1.00 | 20/20 |

The default `ef_search = 40` was enough; 100 was not needed. The run took ~1m50s,
mostly the 40 exact searches (seq scan + sort) through the session pooler.

**Gotcha found on the way: `set local` leaks out of a psycopg savepoint.** On a default
(non-autocommit) psycopg connection, the first `execute` opens a transaction that stays
open. Every later `with conn.transaction():` then becomes a *savepoint*, and `set local`
lasts until the outer transaction ends, not the savepoint. So the first exact search's
`enable_indexscan = off` stayed on, and every later "indexed" search was really a seq scan,
which would have reported a meaningless recall of 1.0. The `explain` check caught it.
Fix: connect with `autocommit=True` and check that the connection is idle
(`conn.info.transaction_status`) before each `set local`. The same applies in Step 6 to any
per-query `set local hnsw.ef_search`.

## Step 4.9: smoke test (2026-11-12)

`uv run python scripts/smoke_search.py` → "my pod keeps restarting", structural, top 5
(9 tokens billed). The query was embedded two ways to inform the Step 6 query convention:

| # | `embedding_input("", q)` | dist | bare query | dist |
|---|---|---|---|---|
| 1 | Pod Lifecycle › … › Container restarts | 0.4025 | Pod Lifecycle › … › Container restarts | 0.4282 |
| 2 | Debug Pods › … › My pod stays terminating | 0.4146 | Pod Lifecycle › How Pods handle problems with containers | 0.4580 |
| 3 | Pod Lifecycle › How Pods handle problems with containers | 0.4206 | … › Container restarts and resilience | 0.4583 |
| 4 | … › Container restarts and resilience | 0.4218 | … › Individual container restart policy and rules | 0.4584 |
| 5 | … › Individual container restart policy and rules | 0.4258 | Pod Lifecycle › Pod lifetime › Pods and fault recovery | 0.4638 |

- **Pass.** Both forms return the guide's expected `Pod Lifecycle › … Container restarts`
  first, so queries and chunks share a model and the vectors are on the right rows.
  No CrashLoopBackOff chunk is in the top 5 in either form (11 structural chunks mention it).
- The two query vectors have cosine similarity 0.9313. The `"\n\n"`-prefixed form sits
  ~0.03 closer to every chunk, which is expected because it looks more like the chunk format.
  The rankings share 4 of 5 results. One query can't show which form retrieves better.
  **Still open:** this conflicts with the Step 4.1 note above (bare query).
  Decide before Step 6, or measure both in Step 9.

**Data-quality finding: Hugo heading anchors survive cleaning.** Headings like
`## Container restarts {#restart-policy}` keep the `{#…}` anchor:
- 986 of 4,773 structural chunks (21%) have it in `heading_path`; fixed: 0
  (their heading path is only the page title).
- 529 chunk texts contain `{#…}` (headings inside chunk bodies, both strategies).
- It passes the "no `{{`" check because it has a single brace.
The anchors add noise tokens to both the embeddings and `tsv` (e.g. `restart-policy`),
and they hit structural chunks far more than fixed, which could confound the config 3
vs 4 comparison. A fix belongs in cleaning (strip ` {#id}` from heading lines), then
re-ingest and re-embed both strategies (~$0.04).

## Query embedding decision (2026-11-12, settles the open question in Step 4.9)

**Queries are embedded bare, using `rag.embed.query_input(query)` (the query, stripped).**
Every configuration in Step 6 and Step 9 uses this. Don't change it mid-evaluation.

- Not `embedding_input("", query)`: with an empty heading path it produces `"\n\n" + query`.
  The two leading blank lines are a side effect of that function, not a decision.
- The heading path is context a chunk has and a query doesn't. `text-embedding-3-small`
  is symmetric (no query/passage prefixes), so different query and chunk inputs are normal.
- Decided before Step 6 so that config 1 isn't measured under one format and later
  configs under another.
- Possible extra row for Step 9: query with vs without a heading-style prefix. It's cheap
  because queries aren't stored.

## Heading-anchor fix (2026-11-12/13)

**Hugo heading anchors `{#…}` survive Markdown-level cleaning and leak into heading paths.**
All 713 anchors in the pinned corpus are trailing on a heading line (712 `{#id}`, one
`{#v1.0}`-style). `clean()` now strips ` {#…}` from heading lines only, before section
splitting, so both `heading_path` and chunk text come out clean. Fenced code never reaches
that rule, and prose mentioning `{#id}` is kept. Tests: `test_heading_anchor_removed`,
`test_anchor_text_outside_heading_or_in_code_is_kept`.

Before the fix: 986 of 4,773 structural heading paths (21%) and 529 fixed chunk texts had
anchors, from 182 documents. Re-ingest: `fixed` normal run updated exactly those 182
documents (264 unchanged); `structural` needed `--rebuild` because of the hash bug below.

## Per-strategy source hash (migration 003)

**Idempotency keys must match the granularity of what they protect.** `documents.content_hash`
is one value per document, but chunks belong to a strategy. Re-ingesting `fixed` after the
anchor fix updated the document hash; `structural` then looked `unchanged` and would have
kept its anchored chunks. `chunks.source_hash` (migration 003) now records the hash each
chunk was built from, and `store()` treats a strategy as unchanged only when *all* of its
chunks carry the current hash. The backfill copied the document hash. I checked first that
all 446 document hashes equal `sha256(clean(body))` under the current cleaning, so that was
correct. Tests: scripted-fake tests in the default suite (other strategy re-chunked; partial
match replaced; inserts carry the hash) plus a `TEST_DATABASE_URL` integration test.

## Supabase pooler stalls and the connection helper (2026-11-13)

Symptoms during the re-ingest: one run died with `server closed the connection
unexpectedly`; two restarts and two embed runs hung for 20–40 min in `poll()` on an
ESTABLISHED TCP connection to the pooler, while `pg_stat_activity` showed no session, lock
or blocked query for them. A test of 3 fresh connections: 2 connected in 2.5 s, 1 hung (killed
at 20 s). Postgres uptime was 7 days and status.supabase.com showed no incident. Likely cause:
an undersized instance or the network path to the pooler.

`rag.db.connect()` is now the only way to connect (all CLIs and scripts use it):
- `connect_timeout=10`, TCP keepalives (30 s idle, 10 s interval, 3 probes),
  `tcp_user_timeout=60000` (Linux only; no effect on macOS)
- 3 attempts with full-jitter backoff (`uniform(0, 1 s × 2^attempt)`)
- `statement_timeout = 60s` and `idle_in_transaction_session_timeout = 120s` applied with `SET`
  after connecting. **The Supabase pooler silently drops the `options` startup parameter**:
  `statement_timeout` stayed at the role default (`2min`) and the idle timeout at `0`.

Limitation: these turn a stalled *connect* into a retry, but a query the pooler swallows
mid-session can still hang. The server's `statement_timeout` never starts, and the pooler
answers TCP keepalives. Long runs are therefore driven by a watchdog that kills and resumes
a run when the chunk count stops moving. Both commands resume safely.

Also learned: the free-size instance can't give a parallel HNSW build 128MB of shared
memory. Future index migrations should use `maintenance_work_mem = '64MB'` (and a serial
build). Inserts and embedding updates now maintain the HNSW indexes as they go, which makes
them slower; that is expected, not a fault.

## Heading shortcode fix and dropping "What's next" (2026-11-16)

**Doubled heading marker.** Pages write `## {{% heading "whatsnext" %}}`, and the old rule
replaced the shortcode with `## Whatsnext`, giving `## ## Whatsnext` (463 structural heading
paths across 325 pages showed `› ## Whatsnext`). The shortcode now supplies only the text,
from Hugo's fixed strings: `whatsnext` → "What's next", `prerequisites` → "Before you begin",
`objectives`, `cleanup` → "Clean up", `seealso` → "See also" (plus `synopsis`, `options`);
unknown keys fall back to the key. Corpus usage: whatsnext 297, prerequisites 204,
objectives 32, cleanup 19, seealso 4.

**Decision: drop "What's next" sections (judgement call).** They are navigation: lists of
related page titles with the links already stripped. As chunks they match related queries
without containing the answer, the distractor problem. `clean()` removes each one up to the
next heading at its level or higher, tracking code fences. It matches the shortcode form and
the plain `## What's next` that 2 pages write by hand; "What's next" in prose is kept.
This applies to both strategies, so configs 1–4 see the same cleaned text.

## Structural merge rule (2026-11-16)

Before: tiny sections only merged forward into a same-parent sibling, using `OVERLAP` (50) as
the threshold. 639 of 4,773 structural chunks were < 50 tokens:
parent intro followed by its own child 290, last section of a page 221, last child before the
outline rises 93, tail of a split > 500-token section 35.

Now `MIN_TOKENS = 50` (its own constant), and a section under it is:
1. merged **forward** into the next section if that is a sibling *or its own descendant*,
   keeping the next section's path (a child's path already includes the parent);
2. otherwise merged **backward** into the previous section if the result fits in 500 tokens,
   labelled with the **common ancestor** of the two paths, because the previous section's
   own path would file the text under a heading it doesn't belong to;
3. otherwise kept. Split tails are a natural result of the 500-token cap and stay.

Local preview over the whole corpus (pure functions, before re-ingest):

| | chunks | p10 | median | p90 | < 50 tokens | < 20 tokens |
|---|---|---|---|---|---|---|
| structural before | 4,773 | 36 | 146 | 436 | 639 | 193 |
| structural after | 4,137 | 66 | 176 | 450 | 60 | 12 |

**Follow-up: `source_hash` covers the input, not the chunker.** After the merge-rule change,
a normal structural re-ingest reported 87 documents `unchanged`: their cleaned text has no
heading shortcodes, so its hash didn't change, and `store()` kept structural chunks built by
the *old* merge rule. Result in the database: 4,206 chunks and 136 under 50 tokens, against
4,137 and 60 in the local preview. Fixed was unaffected (its chunker didn't change).
Fixed this time with `--rebuild` for structural (nothing structural was embedded yet, so it
cost only time). Lasting fix to consider: hash the cleaned text together with a per-strategy
chunker version (bumped whenever `chunk.py` changes behaviour), so a chunking change
invalidates exactly the strategy it affects. Until then: **after any `chunk.py` change, re-ingest
that strategy with `--rebuild`.**

## Step 4 final state after the cleaning and merge fixes (2026-11-16)

Re-ingest (both normal runs, then structural `--rebuild` for the chunker change):
fixed 359 documents updated / 87 unchanged → 2,227 chunks; structural 446 rebuilt →
4,137 chunks (p10 66, median 176, p90 450; 60 under 50 tokens, 12 under 20), matching the
local preview exactly. No `## ##`, no `{#…}`, no stale `source_hash`.

| Re-embed run | Chunks | Billed tokens | Cost (USD) |
|---|---|---|---|
| 2026-11-13 fixed (after anchor fix) | 1,309 | 624,201 | 0.012484 |
| 2026-11-16 fixed (after heading/What's-next fix) | 1,952 | 906,554 | 0.018131 |
| 2026-11-16 structural (all, new merge rule) | 4,137 | 955,718 | 0.019114 |

**Reading `embedding_run.json`:** top-level `cost_usd` = **$0.090362** is the total *spent*
across all six runs (4,518,103 tokens), including re-embeds caused by the two cleaning fixes.
It is not the cost of embedding the corpus once. That is ≈ $0.04 (the 2026-11-16 structural
run alone is $0.0191 for 4,137 chunks; fixed is ~2,227 chunks of ≤ 500 tokens). For the README,
report the total spent and say what caused the re-embeds.

Rerun check: `rag.embed` → 0 chunks, $0.000000. Gate SQL: fixed 2,227/2,227, structural
4,137/4,137 embedded; one model (`text-embedding-3-small`); both HNSW indexes present.

Recall rerun (`scripts/check_recall.py`, seed 0, 20 samples, k=10, ef_search 40): fixed
1.000 (20/20), structural 1.000 (20/20); both plans use their HNSW index. HNSW stayed exact
through ~10k incremental deletes and inserts.

Smoke rerun: "my pod keeps restarting" → `Pod Lifecycle › How Pods handle problems with
containers › Container restarts` first under both query forms, now with clean heading paths.

## Step 5: question set v1 (frozen 2026-11-20)

`eval/questions.jsonl` has 20 questions: 7 identifier, 7 paraphrase and 6 multi. They carry 27 path
labels across 24 documents. It is frozen as the annotated tag `questions-v1` on commit `4a95f4f`.
Every Step 9 result records the tag it ran against. Changing a label after this means a
`questions-v2` tag, a rerun of all configurations, and a line in the README.

- **Provenance.** All 20 `notes` fields record "Drafted with Claude."
- **Labels are document paths, not chunk IDs,** so re-chunking doesn't break them. Three pages
  are labelled in two questions each:
  - `service.md` in q03 and q20
  - `deployment.md` in q04 and q16
  - `job.md` in q07 and q18
- **Four questions are marked hard in `notes`.** Look at these first in Step 9:
  - q01: the field appears on about 8 pages.
  - q03: one number deep in a long page.
  - q10: the answer is one bullet near the end.
  - q17: Docker appears on about 12 unrelated pages.
- **Two label fixes were made before freezing.** No results existed yet.
  1. q01 was first labelled `concepts/workloads/pods/pod-termination.md`, which doesn't exist in
     the pinned corpus. Termination is a section of `pod-lifecycle.md`.
  2. In the ConfigMap/Secret question, the evidence for `secret.md` was ConfigMap's sentence,
     which isn't on that page.

### Validator: `eval/validate_questions.py`

Run it with `uv run python eval/validate_questions.py`. On v1 it reports OK and all evidence
found. Its checks:

- every line parses into the Pydantic model, with extra fields forbidden
- IDs are unique
- the type mix is 7 / 7 / 6
- every labelled page exists and `load()` accepts it
- evidence keys equal `relevant_paths`
- each phrase is in its page's cleaned text
- multi questions have at least 2 distinct paths
- no two questions are the same

Decisions and findings:

- **Evidence is matched in `clean()` output.** Runs of whitespace are ignored because Markdown
  hard-wraps prose; otherwise matching is exact and case-sensitive. On v1 none of the 27 phrases
  needed the whitespace rule, and none exist only in raw markup. So the cleaner removes none of
  the labelled evidence.
- **A missing phrase gets one of two messages:** in the raw page but not after `clean()` (Hugo
  markup, or the cleaner eating text), or not on the page at all (wrong label).
- **Pages that `load()` skips (no title) are flagged.** Ingestion never stores them, so retrieval
  can never return them.
- **The tests found a bug.** An absolute path like `/etc/hosts` passed the "file exists" check,
  because `corpus_dir / "/etc/hosts"` evaluates to `/etc/hosts`. The fix is one guarded helper.
- **`pyproject.toml` changes.** `eval` is added to mypy `files`, so eval scripts (including
  Step 9's `run.py`) get strict checking. pytest gets `pythonpath = ["eval"]` so tests can import
  the script.
