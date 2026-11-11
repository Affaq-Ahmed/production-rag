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
