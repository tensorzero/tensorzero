//! Locates entities (functions, evaluations) within a set of loaded TOML config files.
//!
//! TensorZero configs can be split across multiple TOML files. When modifying an entity,
//! we need to find the file containing its canonical definition (identified by having a
//! `type` key), rather than files that merely extend it (e.g., by adding variants).

use std::path::PathBuf;

use toml_edit::DocumentMut;

use crate::error::ConfigWriterError;

// =============================================================================
// Public Types
// =============================================================================

/// Represents a loaded config file with its parsed TOML document.
#[derive(Debug)]
pub struct LoadedConfigFile {
    pub path: PathBuf,
    pub document: DocumentMut,
}

impl LoadedConfigFile {
    pub fn new(path: PathBuf, document: DocumentMut) -> Self {
        Self { path, document }
    }
}

/// Result of locating a function in the config files.
pub struct FunctionLocation<'a> {
    /// Reference to the config file
    pub file: &'a mut LoadedConfigFile,
}

/// Result of locating an evaluation in the config files.
pub struct EvaluationLocation<'a> {
    /// Reference to the config file
    pub file: &'a mut LoadedConfigFile,
}

// =============================================================================
// Public Functions
// =============================================================================

/// Find the config file that contains the definition of a specific function.
///
/// Searches through the loaded config files and returns the file whose
/// `[functions.<function_name>]` table contains a `type` key. This ensures we find
/// the canonical definition file, not a file that merely extends the function
/// (e.g., by adding variants).
pub fn locate_function<'a>(
    files: &'a mut [LoadedConfigFile],
    function_name: &str,
) -> Result<FunctionLocation<'a>, ConfigWriterError> {
    match find_entity_index(files, "functions", function_name) {
        Some(index) => Ok(FunctionLocation {
            file: &mut files[index],
        }),
        None => Err(ConfigWriterError::FunctionNotFound {
            function_name: function_name.to_string(),
        }),
    }
}

/// Find the config file that contains the definition of a specific evaluation, or the first file if not found.
/// Returns (location, is_new) where is_new indicates if the evaluation doesn't exist yet.
///
/// Searches through the loaded config files and returns the file whose
/// `[evaluations.<evaluation_name>]` table contains a `type` key. This ensures we find
/// the canonical definition file, not a file that merely extends the evaluation.
pub fn locate_evaluation<'a>(
    files: &'a mut [LoadedConfigFile],
    evaluation_name: &str,
) -> Result<(EvaluationLocation<'a>, bool), ConfigWriterError> {
    match find_entity_index(files, "evaluations", evaluation_name) {
        Some(index) => Ok((
            EvaluationLocation {
                file: &mut files[index],
            },
            false,
        )),
        None if !files.is_empty() => Ok((
            EvaluationLocation {
                file: &mut files[0],
            },
            true,
        )),
        None => Err(ConfigWriterError::EvaluationNotFound {
            evaluation_name: evaluation_name.to_string(),
        }),
    }
}

/// Find the config file that contains the definition of a specific evaluation (required to exist).
///
/// Searches through the loaded config files and returns the file whose
/// `[evaluations.<evaluation_name>]` table contains a `type` key. This ensures we find
/// the canonical definition file, not a file that merely extends the evaluation.
pub fn locate_evaluation_required<'a>(
    files: &'a mut [LoadedConfigFile],
    evaluation_name: &str,
) -> Result<EvaluationLocation<'a>, ConfigWriterError> {
    match find_entity_index(files, "evaluations", evaluation_name) {
        Some(index) => Ok(EvaluationLocation {
            file: &mut files[index],
        }),
        None => Err(ConfigWriterError::EvaluationNotFound {
            evaluation_name: evaluation_name.to_string(),
        }),
    }
}

// =============================================================================
// Private Helpers
// =============================================================================

/// Find the index of a config file containing an entity definition.
///
/// Searches for `[{table_name}.{entity_name}]` with a `type` key, which indicates
/// the canonical definition (not just extensions like variants).
fn find_entity_index(
    files: &[LoadedConfigFile],
    table_name: &str,
    entity_name: &str,
) -> Option<usize> {
    files.iter().enumerate().find_map(|(index, file)| {
        file.document
            .get(table_name)
            .and_then(|t| t.as_table())
            .and_then(|t| t.get(entity_name))
            .and_then(|e| e.as_table())
            .filter(|t| t.contains_key("type"))
            .map(|_| index)
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file(path: &str, toml: &str) -> LoadedConfigFile {
        LoadedConfigFile::new(PathBuf::from(path), toml.parse::<DocumentMut>().unwrap())
    }

    #[test]
    fn test_locate_function_finds_file_with_type_key() {
        let mut files = vec![make_file(
            "config.toml",
            r#"
[functions.my_function]
type = "chat"
"#,
        )];

        let result = locate_function(&mut files, "my_function");
        assert!(result.is_ok(), "should find function with type key");
        assert_eq!(result.unwrap().file.path, PathBuf::from("config.toml"));
    }

    #[test]
    fn test_locate_function_returns_file_with_type_not_variants_only() {
        let mut files = vec![
            make_file(
                "variants.toml",
                r#"
[functions.my_function.variants.variant_a]
model = "gpt-4"
"#,
            ),
            make_file(
                "functions.toml",
                r#"
[functions.my_function]
type = "chat"
"#,
            ),
        ];

        let result = locate_function(&mut files, "my_function");
        assert!(result.is_ok(), "should find function definition file");
        assert_eq!(
            result.unwrap().file.path,
            PathBuf::from("functions.toml"),
            "should return the file with the type key, not the variants-only file"
        );
    }

    #[test]
    fn test_locate_function_not_found() {
        let mut files = vec![make_file(
            "config.toml",
            r#"
[functions.other_function]
type = "chat"
"#,
        )];

        let result = locate_function(&mut files, "my_function");
        assert!(
            matches!(
                result,
                Err(ConfigWriterError::FunctionNotFound { function_name }) if function_name == "my_function"
            ),
            "should return FunctionNotFound error"
        );
    }

    #[test]
    fn test_locate_function_ignores_function_without_type() {
        let mut files = vec![make_file(
            "config.toml",
            r#"
[functions.my_function]
description = "A function without a type key"
"#,
        )];

        let result = locate_function(&mut files, "my_function");
        assert!(
            matches!(
                result,
                Err(ConfigWriterError::FunctionNotFound { function_name }) if function_name == "my_function"
            ),
            "should not match function without type key"
        );
    }

    #[test]
    fn test_locate_function_empty_files() {
        let mut files: Vec<LoadedConfigFile> = vec![];

        let result = locate_function(&mut files, "my_function");
        assert!(
            matches!(result, Err(ConfigWriterError::FunctionNotFound { .. })),
            "should return FunctionNotFound for empty files"
        );
    }

    #[test]
    fn test_locate_evaluation_finds_file_with_type_key() {
        let mut files = vec![make_file(
            "config.toml",
            r#"
[evaluations.my_eval]
type = "exact_match"
"#,
        )];

        let result = locate_evaluation(&mut files, "my_eval");
        assert!(result.is_ok(), "should find evaluation with type key");
        let (location, is_new) = result.unwrap();
        assert_eq!(location.file.path, PathBuf::from("config.toml"));
        assert!(!is_new, "should not be marked as new");
    }

    #[test]
    fn test_locate_evaluation_returns_file_with_type_not_extensions_only() {
        let mut files = vec![
            make_file(
                "extensions.toml",
                r#"
[evaluations.my_eval.some_extension]
value = "test"
"#,
            ),
            make_file(
                "evaluations.toml",
                r#"
[evaluations.my_eval]
type = "exact_match"
"#,
            ),
        ];

        let result = locate_evaluation(&mut files, "my_eval");
        assert!(result.is_ok(), "should find evaluation definition file");
        let (location, is_new) = result.unwrap();
        assert_eq!(
            location.file.path,
            PathBuf::from("evaluations.toml"),
            "should return the file with the type key"
        );
        assert!(!is_new, "should not be marked as new");
    }

    #[test]
    fn test_locate_evaluation_returns_first_file_when_not_found() {
        let mut files = vec![
            make_file("first.toml", "[other]\nkey = \"value\""),
            make_file("second.toml", "[other]\nkey = \"value\""),
        ];

        let result = locate_evaluation(&mut files, "my_eval");
        assert!(
            result.is_ok(),
            "should return first file for new evaluation"
        );
        let (location, is_new) = result.unwrap();
        assert_eq!(
            location.file.path,
            PathBuf::from("first.toml"),
            "should return the first file"
        );
        assert!(is_new, "should be marked as new");
    }

    #[test]
    fn test_locate_evaluation_required_finds_file_with_type_key() {
        let mut files = vec![make_file(
            "config.toml",
            r#"
[evaluations.my_eval]
type = "exact_match"
"#,
        )];

        let result = locate_evaluation_required(&mut files, "my_eval");
        assert!(result.is_ok(), "should find evaluation with type key");
        assert_eq!(result.unwrap().file.path, PathBuf::from("config.toml"));
    }

    #[test]
    fn test_locate_evaluation_required_not_found() {
        let mut files = vec![make_file(
            "config.toml",
            r#"
[evaluations.other_eval]
type = "exact_match"
"#,
        )];

        let result = locate_evaluation_required(&mut files, "my_eval");
        assert!(
            matches!(
                result,
                Err(ConfigWriterError::EvaluationNotFound { evaluation_name }) if evaluation_name == "my_eval"
            ),
            "should return EvaluationNotFound error"
        );
    }

    #[test]
    fn test_locate_evaluation_required_ignores_evaluation_without_type() {
        let mut files = vec![make_file(
            "config.toml",
            r#"
[evaluations.my_eval]
description = "An evaluation without a type key"
"#,
        )];

        let result = locate_evaluation_required(&mut files, "my_eval");
        assert!(
            matches!(
                result,
                Err(ConfigWriterError::EvaluationNotFound { evaluation_name }) if evaluation_name == "my_eval"
            ),
            "should not match evaluation without type key"
        );
    }
}
