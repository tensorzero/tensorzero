ALTER TABLE tensorzero.prompt_template_configs RENAME TO stored_files;

ALTER INDEX tensorzero.idx_prompt_template_configs_content_lookup
    RENAME TO idx_stored_files_content_lookup;
