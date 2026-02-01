//! TensorZero Config Writer
//!
//! A crate for applying targeted edits to TensorZero config TOML files while preserving formatting.
//! Supports 4 operations: upsert variant, upsert experimentation config, upsert evaluation, upsert evaluator.

mod edit;
mod error;
mod locator;
mod path_resolver;
mod toml_writer;

pub use edit::{
    EditPayload, UpsertEvaluationPayload, UpsertEvaluatorPayload, UpsertExperimentationPayload,
    UpsertVariantPayload,
};
pub use error::ConfigWriterError;

use std::path::{Path, PathBuf};

use locator::LoadedConfigFile;
use path_resolver::FileToWrite;
use tensorzero_core::config::ConfigFileGlob;
use toml_edit::DocumentMut;

/// ConfigWriter handles applying edits to TensorZero config files.
pub struct ConfigWriter {
    /// The glob pattern used to find config files
    glob_pattern: String,
    /// The base directory extracted from the glob pattern
    glob_base: PathBuf,
    /// The loaded config files with their parsed TOML documents
    files: Vec<LoadedConfigFile>,
}

impl ConfigWriter {
    /// Create a new ConfigWriter by loading all config files matching the glob pattern.
    pub async fn new(glob_pattern: &str) -> Result<Self, ConfigWriterError> {
        let config_glob = ConfigFileGlob::new(glob_pattern.to_string()).map_err(|e| {
            ConfigWriterError::InvalidGlob {
                pattern: glob_pattern.to_string(),
                message: e.to_string(),
            }
        })?;

        if config_glob.paths.is_empty() {
            return Err(ConfigWriterError::NoConfigFiles {
                pattern: glob_pattern.to_string(),
            });
        }

        let glob_base = config_glob.base_path();

        // Load all config files using toml_edit for format preservation
        let mut files = Vec::new();
        for path in &config_glob.paths {
            let content = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| ConfigWriterError::io(path, e))?;

            let document: DocumentMut = content
                .parse()
                .map_err(|e: toml_edit::TomlError| ConfigWriterError::toml_parse(path, e))?;

            files.push(LoadedConfigFile::new(path.clone(), document));
        }

        Ok(Self {
            glob_pattern: glob_pattern.to_string(),
            glob_base,
            files,
        })
    }

    /// Apply an edit to the config files.
    /// Returns the paths of all files that were written (TOML config + any template/schema files).
    pub async fn apply_edit(
        &mut self,
        edit: &EditPayload,
    ) -> Result<Vec<PathBuf>, ConfigWriterError> {
        let files_to_write = match edit {
            EditPayload::UpsertVariant(payload) => self.apply_upsert_variant(payload)?,
            EditPayload::UpsertExperimentation(payload) => {
                self.apply_upsert_experimentation(payload)?
            }
            EditPayload::UpsertEvaluation(payload) => self.apply_upsert_evaluation(payload)?,
            EditPayload::UpsertEvaluator(payload) => self.apply_upsert_evaluator(payload)?,
        };

        // Write all files (TOML and any extracted template/schema files)
        let mut written_paths = Vec::with_capacity(files_to_write.len());
        for file_to_write in files_to_write {
            // Create parent directories if needed
            if let Some(parent) = file_to_write.absolute_path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| ConfigWriterError::io(&file_to_write.absolute_path, e))?;
            }

            tokio::fs::write(&file_to_write.absolute_path, &file_to_write.content)
                .await
                .map_err(|e| ConfigWriterError::io(&file_to_write.absolute_path, e))?;

            written_paths.push(file_to_write.absolute_path);
        }

        Ok(written_paths)
    }

    fn apply_upsert_variant(
        &mut self,
        payload: &UpsertVariantPayload,
    ) -> Result<Vec<FileToWrite>, ConfigWriterError> {
        let location = locator::locate_function(&mut self.files, &payload.function_name)?;
        let toml_file_dir = location
            .file
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        // Serialize the variant to a TOML item
        let mut variant_item = toml_writer::serialize_to_item(&payload.variant)?;

        // Extract any ResolvedTomlPathData fields and convert to relative paths
        let template_files = toml_writer::extract_resolved_paths(
            &mut variant_item,
            &self.glob_base,
            &toml_file_dir,
            &payload.function_name,
            &payload.variant_name,
            None,
        )?;

        // Apply the edit to the document
        toml_writer::upsert_variant(
            &mut location.file.document,
            &payload.function_name,
            &payload.variant_name,
            variant_item,
        );

        // Prepare files to write
        let mut files = template_files;
        files.push(FileToWrite {
            absolute_path: location.file.path.clone(),
            content: location.file.document.to_string(),
        });

        Ok(files)
    }

    fn apply_upsert_experimentation(
        &mut self,
        payload: &UpsertExperimentationPayload,
    ) -> Result<Vec<FileToWrite>, ConfigWriterError> {
        let location = locator::locate_function(&mut self.files, &payload.function_name)?;

        // Serialize the experimentation config to a TOML item
        let experimentation_item = toml_writer::serialize_to_item(&payload.experimentation)?;

        // Apply the edit to the document
        toml_writer::upsert_experimentation(
            &mut location.file.document,
            &payload.function_name,
            experimentation_item,
        );

        // Prepare files to write (just the TOML file, no templates for experimentation)
        let files = vec![FileToWrite {
            absolute_path: location.file.path.clone(),
            content: location.file.document.to_string(),
        }];

        Ok(files)
    }

    fn apply_upsert_evaluation(
        &mut self,
        payload: &UpsertEvaluationPayload,
    ) -> Result<Vec<FileToWrite>, ConfigWriterError> {
        let (location, _is_new) =
            locator::locate_evaluation(&mut self.files, &payload.evaluation_name)?;

        // Serialize the evaluation to a TOML item
        let evaluation_item = toml_writer::serialize_to_item(&payload.evaluation)?;

        // Apply the edit to the document
        toml_writer::upsert_evaluation(
            &mut location.file.document,
            &payload.evaluation_name,
            evaluation_item,
        );

        // Prepare files to write (just the TOML file for now)
        // Note: evaluations may contain evaluators with templates, but those would be
        // handled when upserting individual evaluators
        let files = vec![FileToWrite {
            absolute_path: location.file.path.clone(),
            content: location.file.document.to_string(),
        }];

        Ok(files)
    }

    fn apply_upsert_evaluator(
        &mut self,
        payload: &UpsertEvaluatorPayload,
    ) -> Result<Vec<FileToWrite>, ConfigWriterError> {
        let location =
            locator::locate_evaluation_required(&mut self.files, &payload.evaluation_name)?;
        let toml_file_dir = location
            .file
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        // Serialize the evaluator to a TOML item
        let mut evaluator_item = toml_writer::serialize_to_item(&payload.evaluator)?;

        // Extract any ResolvedTomlPathData fields from evaluator variants
        // For LLM judge evaluators, we need to process each variant
        let mut template_files = Vec::new();
        if let Some(variants) = evaluator_item
            .as_table_mut()
            .and_then(|t| t.get_mut("variants"))
            .and_then(|v| v.as_table_mut())
        {
            let variant_names: Vec<String> = variants.iter().map(|(k, _)| k.to_string()).collect();
            for variant_name in variant_names {
                if let Some(variant_item) = variants.get_mut(&variant_name) {
                    let variant_files = toml_writer::extract_resolved_paths_evaluator(
                        variant_item,
                        &self.glob_base,
                        &toml_file_dir,
                        &payload.evaluation_name,
                        &payload.evaluator_name,
                        &variant_name,
                        None,
                    )?;
                    template_files.extend(variant_files);
                }
            }
        }

        // Apply the edit to the document
        toml_writer::upsert_evaluator(
            &mut location.file.document,
            &payload.evaluation_name,
            &payload.evaluator_name,
            evaluator_item,
        );

        // Prepare files to write
        let mut files = template_files;
        files.push(FileToWrite {
            absolute_path: location.file.path.clone(),
            content: location.file.document.to_string(),
        });

        Ok(files)
    }

    /// Get the glob pattern used to find config files.
    pub fn glob_pattern(&self) -> &str {
        &self.glob_pattern
    }

    /// Get the base directory extracted from the glob pattern.
    pub fn glob_base(&self) -> &Path {
        &self.glob_base
    }

    /// Get the paths of all loaded config files.
    pub fn config_paths(&self) -> Vec<&Path> {
        self.files.iter().map(|f| f.path.as_path()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_test_config(dir: &Path) {
        fs::write(
            dir.join("tensorzero.toml"),
            r#"[functions.my_function]
type = "chat"

[functions.my_function.variants.baseline]
type = "chat_completion"
model = "gpt-4"
"#,
        )
        .expect("failed to write test config");
    }

    fn setup_test_config_with_evaluation(dir: &Path) {
        fs::write(
            dir.join("tensorzero.toml"),
            r#"[functions.my_function]
type = "chat"

[functions.my_function.variants.baseline]
type = "chat_completion"
model = "gpt-4"

[evaluations.my_evaluation]
type = "inference"
function_name = "my_function"

[evaluations.my_evaluation.evaluators.exact_match]
type = "exact_match"
"#,
        )
        .expect("failed to write test config");
    }

    #[tokio::test]
    async fn test_config_writer_new() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let writer = ConfigWriter::new(&glob)
            .await
            .expect("failed to create writer");

        assert_eq!(writer.config_paths().len(), 1);
    }

    #[tokio::test]
    async fn test_config_writer_no_files() {
        let tmp = TempDir::new().expect("failed to create temp dir");

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let result = ConfigWriter::new(&glob).await;

        assert!(result.is_err());
        if let Err(ConfigWriterError::InvalidGlob { pattern, .. }) = result {
            assert_eq!(pattern, glob);
        } else {
            panic!("expected InvalidGlob error");
        }
    }

    #[tokio::test]
    async fn test_locate_function() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let mut writer = ConfigWriter::new(&glob)
            .await
            .expect("failed to create writer");

        // Test that we can find an existing function
        let location = locator::locate_function(&mut writer.files, "my_function");
        assert!(location.is_ok());

        // Test that we get an error for a non-existent function
        let location = locator::locate_function(&mut writer.files, "nonexistent");
        assert!(location.is_err());
    }

    #[tokio::test]
    async fn test_locate_evaluation() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config_with_evaluation(tmp.path());

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let mut writer = ConfigWriter::new(&glob)
            .await
            .expect("failed to create writer");

        // Test that we can find an existing evaluation
        let (location, is_new) = locator::locate_evaluation(&mut writer.files, "my_evaluation")
            .expect("failed to locate");
        assert!(!is_new);
        assert!(location.file.path.ends_with("tensorzero.toml"));

        // Test that a new evaluation returns is_new=true and uses the first file
        let mut writer = ConfigWriter::new(&glob)
            .await
            .expect("failed to create writer");
        let (location, is_new) = locator::locate_evaluation(&mut writer.files, "new_evaluation")
            .expect("failed to locate");
        assert!(is_new);
        assert!(location.file.path.ends_with("tensorzero.toml"));
    }
}
