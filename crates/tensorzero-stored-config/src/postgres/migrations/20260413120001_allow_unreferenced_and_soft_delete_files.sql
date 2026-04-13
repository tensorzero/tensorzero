-- Add soft-delete support to stored_files so the config editor can track
-- which files are currently "active" (free or referenced) and tombstone
-- files the user removes. Config loading queries stored_files by UUID and
-- never filters on deleted_at, so tombstoning a row does not affect the
-- live config — it only affects the editor view.
ALTER TABLE tensorzero.stored_files ADD COLUMN deleted_at TIMESTAMPTZ;

-- Partial index supporting the editor's "latest non-deleted version per
-- file path" query:
--   SELECT DISTINCT ON (file_path) file_path, source_body
--   FROM tensorzero.stored_files
--   WHERE deleted_at IS NULL
--   ORDER BY file_path, created_at DESC
CREATE INDEX idx_stored_files_editor_latest
    ON tensorzero.stored_files (file_path, created_at DESC)
    WHERE deleted_at IS NULL;
