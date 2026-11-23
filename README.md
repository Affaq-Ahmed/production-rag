# production-rag

A retrieval system over the English Kubernetes documentation, built to measure how retrieval
quality changes from one configuration to the next:

| Config | Search | Rerank | Chunking |
|---|---|---|---|
| 1 | vector only | off | fixed |
| 2 | hybrid (vector + Postgres full-text, fused with RRF) | off | fixed |
| 3 | hybrid | cross-encoder | fixed |
| 4 | hybrid | cross-encoder | structural |

Results will be added when the evaluation runs. Every number in them will come from a committed
file in `eval/results/`.

## Test set

The test set is `eval/questions.jsonl`, frozen as the git tag `questions-v2`. It has 20
questions of three types:

| Type | Questions | What it tests |
|---|---|---|
| identifier | 7 | The question names an exact field, command or status, such as `terminationGracePeriodSeconds` or `ImagePullBackOff`. |
| paraphrase | 7 | The question describes the problem without the page's own terms (for example, it avoids "disruption budget" or "liveness probe"). |
| multi | 6 | The answer needs two or more pages. |

- **Labels are document paths, not chunk IDs.** Re-chunking doesn't change them. The set has 28
  labels across 25 documents.
- **Every label has an evidence phrase.** The validator checks that the phrase appears in that
  page's cleaned text, so each label is backed by text that actually reaches the index.
- **Provenance.** Each question's `notes` field records that it was drafted with Claude.

To check the file:

```bash
uv run python eval/validate_questions.py
```

### Versioning

Every evaluation result records the version of the question set it ran against. If a label
turns out to be wrong, it gets fixed under a new tag, every configuration is rerun, and the
change is listed here.

| Version | Change |
|---|---|
| `questions-v1` | First version. |
| `questions-v2` | q02 reworded (label unchanged). q18 gains `tasks/job/automated-tasks-with-cron-jobs.md`, an answering page that was missing from its labels and was found during the Step 6.9 vector-search check. Details in NOTES.md. |
