-- Add soft-delete support to stored_files so the config editor can track
-- which files are currently "active" (free or referenced) and tombstone
-- files the user removes. Config loading queries stored_files by UUID and
-- never filters on deleted_at, so tombstoning a row does not affect the
-- live config — it only affects the editor view.
ALTER TABLE tensorzero.stored_files ADD COLUMN deleted_at TIMESTAMPTZ;

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
