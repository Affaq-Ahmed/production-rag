\set ON_ERROR_STOP on

begin;

insert into documents (path, source_url, title, content_type, content_hash, corpus_commit)
values ('concepts/test.md', 'https://kubernetes.io/docs/concepts/test/', 'Test', 'concept',
        'abc', 'deadbeef')
returning id \gset doc_

insert into chunks (document_id, strategy, ord, heading_path, text, token_count)
values (:doc_id, 'fixed', 0, 'Pod Lifecycle',
        'A pod in CrashLoopBackOff keeps restarting its containers.', 12);

-- 1. tsv was generated, and an identifier is searchable
select tsv @@ websearch_to_tsquery('english', 'CrashLoopBackOff') as finds_identifier
from chunks where document_id = :doc_id;

-- 2. a 1536-dimension vector is accepted
update chunks set embedding = array_fill(0.01, array[1536])::vector,
                  embedding_model = 'text-embedding-3-small'
where document_id = :doc_id;

rollback;