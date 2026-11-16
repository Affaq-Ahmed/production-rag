-- 003: per-strategy source hash, so re-chunking one strategy can't hide stale chunks of another
--
-- documents.content_hash is one value per document, but chunks belong to a strategy. Re-ingesting
-- strategy A updated the document hash, after which strategy B looked unchanged and kept chunks
-- built from the old cleaned text. Each chunk now records the hash of the text it was built from,
-- and store() treats a strategy as unchanged only when all of its chunks carry the current hash.

alter table chunks add column source_hash text;

-- Backfill from the document hash. This is correct at the time this migration was written
-- (2026-11-13): every existing chunk was built from its document's current cleaned text. The
-- 182 documents changed by the heading-anchor fix were re-ingested for `fixed`, and their
-- `structural` chunks were deleted by a --rebuild, so no stale chunks remain to mislabel.
update chunks c
set source_hash = d.content_hash
from documents d
where d.id = c.document_id;

alter table chunks alter column source_hash set not null;
