-- Add soft-delete support to stored_files so the config editor can track
-- which files are currently "active" (free or referenced) and tombstone
-- files the user removes. Config loading queries stored_files by UUID and
-- never filters on deleted_at, so tombstoning a row does not affect the
-- live config — it only affects the editor view.
ALTER TABLE tensorzero.stored_files ADD COLUMN deleted_at TIMESTAMPTZ;

-- Backfill: before this migration the write path could leave multiple rows
-- per `file_path` (a referenced file's content change inserted a new row
-- without tombstoning the old one). All such rows have
-- `deleted_at IS NULL` now that the column exists, which would break the
-- post-migration editor read path — it enforces "at most one active row
-- per file_path". Tombstone every non-latest row per `file_path` so the
-- invariant holds from day one. Use `(created_at DESC, id DESC)` so the
-- "latest" row matches the tie-breaking previously applied at read time.
UPDATE tensorzero.stored_files AS t
SET deleted_at = NOW()
FROM (
    SELECT id
    FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY file_path
                ORDER BY created_at DESC, id DESC
            ) AS rn
        FROM tensorzero.stored_files
    ) ranked
    WHERE rn > 1
) superseded
WHERE t.id = superseded.id;

-- Partial index supporting the editor's "all non-deleted rows" query and
-- the free-file write path's per-path lookups:
--   SELECT file_path, source_body
--   FROM tensorzero.stored_files
--   WHERE deleted_at IS NULL
-- The write path maintains the invariant of at most one active row per
-- file_path; the read path returns an error if that invariant is violated.
CREATE INDEX idx_stored_files_editor_latest
    ON tensorzero.stored_files (file_path)
    WHERE deleted_at IS NULL;
