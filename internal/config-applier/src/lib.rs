//! TensorZero Config Applier
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
pub use error::ConfigApplierError;

use std::path::{Path, PathBuf};

use locator::LoadedConfigFile;
use path_resolver::FileToWrite;
use tensorzero_core::config::ConfigFileGlob;
use toml_edit::DocumentMut;

/// ConfigApplier handles applying edits to TensorZero config files.
pub struct ConfigApplier {
    /// The base directory extracted from the glob pattern
    glob_base: PathBuf,
    /// The loaded config files with their parsed TOML documents
    files: Vec<LoadedConfigFile>,
}

impl ConfigApplier {
    /// Create a new ConfigApplier by loading all config files matching the glob pattern.
    pub async fn new(glob_pattern: &str) -> Result<Self, ConfigApplierError> {
        let config_glob = ConfigFileGlob::new(glob_pattern.to_string()).map_err(|e| {
            ConfigApplierError::InvalidGlob {
                pattern: glob_pattern.to_string(),
                message: e.to_string(),
            }
        })?;

        let mut glob_base = config_glob.base_path();
        if glob_base.is_file() {
            glob_base = glob_base
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
        }

        // Load all config files using toml_edit for format preservation
        let mut files = Vec::new();
        for path in &config_glob.paths {
            let content = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| ConfigApplierError::io(path, e))?;

            let document: DocumentMut = content
                .parse()
                .map_err(|e: toml_edit::TomlError| ConfigApplierError::toml_parse(path, e))?;

            files.push(LoadedConfigFile::new(path.clone(), document));
        }

        Ok(Self { glob_base, files })
    }

    /// Apply an edit to the config files.
    /// Returns the paths of all files that were written (TOML config + any template/schema files).
    pub async fn apply_edit(
        &mut self,
        edit: &EditPayload,
    ) -> Result<Vec<PathBuf>, ConfigApplierError> {
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
                    .map_err(|e| ConfigApplierError::io(&file_to_write.absolute_path, e))?;
            }

            tokio::fs::write(&file_to_write.absolute_path, &file_to_write.content)
                .await
                .map_err(|e| ConfigApplierError::io(&file_to_write.absolute_path, e))?;

            written_paths.push(file_to_write.absolute_path);
        }

        Ok(written_paths)
    }

    fn apply_upsert_variant(
        &mut self,
        payload: &UpsertVariantPayload,
    ) -> Result<Vec<FileToWrite>, ConfigApplierError> {
        path_resolver::validate_path_component(&payload.function_name, "function_name")?;
        path_resolver::validate_path_component(&payload.variant_name, "variant_name")?;

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
        let template_files = path_resolver::extract_resolved_paths(
            &mut variant_item,
            &self.glob_base,
            &toml_file_dir,
            &[
                "functions",
                &payload.function_name,
                "variants",
                &payload.variant_name,
            ],
        )?;

        // Apply the edit to the document
        toml_writer::upsert_variant(
            &mut location.file.document,
            &payload.function_name,
            &payload.variant_name,
            variant_item,
        )?;

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
    ) -> Result<Vec<FileToWrite>, ConfigApplierError> {
        path_resolver::validate_path_component(&payload.function_name, "function_name")?;

        let location = locator::locate_function(&mut self.files, &payload.function_name)?;

        // Serialize the experimentation config to a TOML item
        let experimentation_item = toml_writer::serialize_to_item(&payload.experimentation)?;

        // Apply the edit to the document
        toml_writer::upsert_experimentation(
            &mut location.file.document,
            &payload.function_name,
            experimentation_item,
        )?;

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
    ) -> Result<Vec<FileToWrite>, ConfigApplierError> {
        path_resolver::validate_path_component(&payload.evaluation_name, "evaluation_name")?;

        let (location, _is_new) =
            locator::locate_evaluation(&mut self.files, &payload.evaluation_name)?;
        let toml_file_dir = location
            .file
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        // Serialize the evaluation to a TOML item
        let mut evaluation_item = toml_writer::serialize_to_item(&payload.evaluation)?;

        // Extract any ResolvedTomlPathData fields from evaluator variants in this evaluation
        let template_files = path_resolver::extract_resolved_paths(
            &mut evaluation_item,
            &self.glob_base,
            &toml_file_dir,
            &["evaluations", &payload.evaluation_name],
        )?;

        // Apply the edit to the document
        toml_writer::upsert_evaluation(
            &mut location.file.document,
            &payload.evaluation_name,
            evaluation_item,
        )?;

        // Prepare files to write
        let mut files = template_files;
        files.push(FileToWrite {
            absolute_path: location.file.path.clone(),
            content: location.file.document.to_string(),
        });

        Ok(files)
    }

    fn apply_upsert_evaluator(
        &mut self,
        payload: &UpsertEvaluatorPayload,
    ) -> Result<Vec<FileToWrite>, ConfigApplierError> {
        path_resolver::validate_path_component(&payload.evaluation_name, "evaluation_name")?;
        path_resolver::validate_path_component(&payload.evaluator_name, "evaluator_name")?;

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
        let template_files = path_resolver::extract_resolved_paths(
            &mut evaluator_item,
            &self.glob_base,
            &toml_file_dir,
            &[
                "evaluations",
                &payload.evaluation_name,
                "evaluators",
                &payload.evaluator_name,
            ],
        )?;

        // Apply the edit to the document
        toml_writer::upsert_evaluator(
            &mut location.file.document,
            &payload.evaluation_name,
            &payload.evaluator_name,
            evaluator_item,
        )?;

        // Prepare files to write
        let mut files = template_files;
        files.push(FileToWrite {
            absolute_path: location.file.path.clone(),
            content: location.file.document.to_string(),
        });

        Ok(files)
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
    use std::collections::HashMap;
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tensorzero_core::config::path::ResolvedTomlPathData;
    use tensorzero_core::evaluations::{
        LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize, LLMJudgeOutputType,
        UninitializedEvaluationConfig, UninitializedEvaluatorConfig,
        UninitializedInferenceEvaluationConfig, UninitializedLLMJudgeChatCompletionVariantConfig,
        UninitializedLLMJudgeConfig, UninitializedLLMJudgeVariantConfig,
        UninitializedLLMJudgeVariantInfo,
    };
    use tensorzero_core::utils::retries::RetryConfig;
    use tensorzero_core::variant::JsonMode;

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
        let writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        assert_eq!(writer.config_paths().len(), 1);
    }

    #[tokio::test]
    async fn test_config_writer_no_files() {
        let tmp = TempDir::new().expect("failed to create temp dir");

        let glob = format!("{}/**/*.toml", tmp.path().display());
        let result = ConfigApplier::new(&glob).await;

        assert!(result.is_err());
        if let Err(ConfigApplierError::InvalidGlob { pattern, .. }) = result {
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
        let mut writer = ConfigApplier::new(&glob)
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
        let mut writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        // Test that we can find an existing evaluation
        let (location, is_new) = locator::locate_evaluation(&mut writer.files, "my_evaluation")
            .expect("failed to locate");
        assert!(!is_new);
        assert!(location.file.path.ends_with("tensorzero.toml"));

        // Test that a new evaluation returns is_new=true and uses the first file
        let mut writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");
        let (location, is_new) = locator::locate_evaluation(&mut writer.files, "new_evaluation")
            .expect("failed to locate");
        assert!(is_new);
        assert!(location.file.path.ends_with("tensorzero.toml"));
    }

    #[tokio::test]
    async fn test_config_writer_base_path_for_single_file_glob() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let config_path = tmp.path().join("tensorzero.toml");
        let glob = config_path.display().to_string();
        let writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        assert_eq!(
            writer.glob_base(),
            tmp.path(),
            "expected glob_base to be the parent dir for a single-file path"
        );
        assert!(
            writer.glob_base().is_dir(),
            "expected glob_base to be a directory path"
        );
    }

    #[tokio::test]
    async fn test_apply_upsert_evaluation_extracts_templates() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        setup_test_config(tmp.path());

        let glob = format!("{}/tensorzero.toml", tmp.path().display());
        let mut writer = ConfigApplier::new(&glob)
            .await
            .expect("failed to create writer");

        let system_instructions = ResolvedTomlPathData::new_fake_path(
            "inline".to_string(),
            "You are a friendly judge.".to_string(),
        );
        let variant = UninitializedLLMJudgeVariantInfo {
            inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                UninitializedLLMJudgeChatCompletionVariantConfig {
                    active: Some(true),
                    model: Arc::from("gpt-4"),
                    system_instructions,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    presence_penalty: None,
                    frequency_penalty: None,
                    seed: None,
                    json_mode: JsonMode::Strict,
                    stop_sequences: None,
                    reasoning_effort: None,
                    service_tier: None,
                    thinking_budget_tokens: None,
                    verbosity: None,
                    retries: RetryConfig::default(),
                    extra_body: None,
                    extra_headers: None,
                },
            ),
            timeouts: None,
        };

        let mut variants = HashMap::new();
        variants.insert("v1".to_string(), variant);

        let evaluator = UninitializedEvaluatorConfig::LLMJudge(UninitializedLLMJudgeConfig {
            input_format: LLMJudgeInputFormat::Messages,
            variants,
            output_type: LLMJudgeOutputType::Boolean,
            optimize: LLMJudgeOptimize::Max,
            include: LLMJudgeIncludeConfig::default(),
            cutoff: None,
            description: None,
        });

        let mut evaluators = HashMap::new();
        evaluators.insert("judge".to_string(), evaluator);

        let evaluation =
            UninitializedEvaluationConfig::Inference(UninitializedInferenceEvaluationConfig {
                evaluators,
                function_name: "my_function".to_string(),
                description: None,
            });

        let edit = EditPayload::UpsertEvaluation(UpsertEvaluationPayload {
            evaluation_name: "my_evaluation".to_string(),
            evaluation,
        });

        let written_paths = writer
            .apply_edit(&edit)
            .await
            .expect("failed to apply evaluation edit");

        let expected_template_path = tmp
            .path()
            .join("evaluations")
            .join("my_evaluation")
            .join("evaluators")
            .join("judge")
            .join("variants")
            .join("v1")
            .join("system_instructions.txt");

        assert!(
            expected_template_path.exists(),
            "expected evaluator system_instructions template to be written"
        );
        let template_contents =
            fs::read_to_string(&expected_template_path).expect("failed to read template file");
        assert_eq!(
            template_contents, "You are a friendly judge.",
            "expected template file to contain system_instructions data"
        );
        assert!(
            written_paths
                .iter()
                .any(|path| path == &expected_template_path),
            "expected written_paths to include evaluator template path"
        );

        let toml_contents =
            fs::read_to_string(tmp.path().join("tensorzero.toml")).expect("failed to read config");
        let doc: toml_edit::DocumentMut =
            toml_contents.parse().expect("failed to parse updated TOML");
        let system_instructions = doc
            .get("evaluations")
            .and_then(|v| v.get("my_evaluation"))
            .and_then(|v| v.get("evaluators"))
            .and_then(|v| v.get("judge"))
            .and_then(|v| v.get("variants"))
            .and_then(|v| v.get("v1"))
            .and_then(|v| v.get("system_instructions"))
            .and_then(|v| v.as_str())
            .expect("expected system_instructions to be a string");
        assert_eq!(
            system_instructions,
            "evaluations/my_evaluation/evaluators/judge/variants/v1/system_instructions.txt",
            "expected TOML to reference the extracted system_instructions path"
        );
    }
}
