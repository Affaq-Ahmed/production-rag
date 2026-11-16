-- 002: HNSW indexes, one per chunking strategy, built after the bulk load

-- `local` scopes the setting to this migration's transaction (scripts/migrate.py wraps each
-- file in one), so the session pooler connection doesn't keep the raised value afterwards.
-- A parallel HNSW build reserves maintenance_work_mem as one shared-memory segment up front.
-- On this Supabase instance that failed at 256MB (DiskFull) and at 128MB (OutOfMemory).
-- A serial build allocates as it goes; the vectors total ~41MB, so it is quick at this size.
set local max_parallel_maintenance_workers = 0;
set local maintenance_work_mem = '128MB';

-- Partial, one per strategy: every search filters by strategy, and one shared approximate index
-- would find the nearest vectors first and filter afterwards, returning fewer than k rows.
-- vector_cosine_ops: OpenAI embeddings are unit length, so cosine ranks the same as inner
-- product; it matches the <=> operator used in retrieval.
create index chunks_embedding_fixed_hnsw
    on chunks using hnsw (embedding vector_cosine_ops)
    where strategy = 'fixed';

create index chunks_embedding_structural_hnsw
    on chunks using hnsw (embedding vector_cosine_ops)
    where strategy = 'structural';
