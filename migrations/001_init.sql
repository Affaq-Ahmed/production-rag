-- 001: documents and chunks

create extension if not exists vector;

create table documents (
    id             bigint generated always as identity primary key,
    path           text        not null unique,   -- e.g. concepts/workloads/pods/pod-lifecycle.md
    source_url     text        not null,          -- https://kubernetes.io/docs/...
    title          text        not null,
    content_type   text,                          -- concept | task | tutorial | ...
    content_hash   text        not null,          -- sha256 of cleaned text; makes ingestion idempotent
    corpus_commit  text        not null,          -- the pinned kubernetes/website commit
    ingested_at    timestamptz not null default now()
);

create table chunks (
    id               bigint generated always as identity primary key,
    document_id      bigint      not null references documents(id) on delete cascade,
    strategy         text        not null check (strategy in ('fixed', 'structural')),
    ord              integer     not null check (ord >= 0),     -- position within the document
    heading_path     text        not null default '',           -- "Pod Lifecycle › Container probes"
    text             text        not null check (length(text) > 0),
    token_count      integer     not null check (token_count > 0),
    embedding        vector(1536),                              -- null until Step 4
    embedding_model  text,                                      -- set together with embedding
    tsv              tsvector generated always as (
                         to_tsvector('english', heading_path || ' ' || text)
                     ) stored,
    created_at       timestamptz not null default now(),

    unique (document_id, strategy, ord),
    check ((embedding is null) = (embedding_model is null))
);

create index chunks_document_idx on chunks (document_id);
create index chunks_strategy_idx on chunks (strategy);
create index chunks_tsv_idx      on chunks using gin (tsv);

-- Supabase exposes tables in `public` through its REST API. RLS with no policies blocks that
-- route entirely. Your app connects as the table owner, which bypasses RLS, so nothing breaks.
alter table documents enable row level security;
alter table chunks    enable row level security;