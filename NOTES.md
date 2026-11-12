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
