use std::path::PathBuf;

use toml_edit::DocumentMut;

use crate::error::ConfigWriterError;

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

/// Find the config file that contains a specific function.
pub fn locate_function<'a>(
    files: &'a mut [LoadedConfigFile],
    function_name: &str,
) -> Result<FunctionLocation<'a>, ConfigWriterError> {
    // First pass: find the index (immutable borrow)
    let found_index = files.iter().enumerate().find_map(|(index, file)| {
        file.document
            .get("functions")
            .and_then(|f| f.as_table())
            .filter(|t| t.contains_key(function_name))
            .map(|_| index)
    });

    match found_index {
        Some(index) => {
            let file = &mut files[index];
            Ok(FunctionLocation { file })
        }
        None => Err(ConfigWriterError::FunctionNotFound {
            function_name: function_name.to_string(),
        }),
    }
}

/// Find the config file that contains a specific evaluation, or the first file if not found.
/// Returns (location, is_new) where is_new indicates if the evaluation doesn't exist yet.
pub fn locate_evaluation<'a>(
    files: &'a mut [LoadedConfigFile],
    evaluation_name: &str,
) -> Result<(EvaluationLocation<'a>, bool), ConfigWriterError> {
    // First pass: find the index of an existing evaluation (immutable borrow)
    let found_index = files.iter().enumerate().find_map(|(index, file)| {
        file.document
            .get("evaluations")
            .and_then(|e| e.as_table())
            .filter(|t| t.contains_key(evaluation_name))
            .map(|_| index)
    });

    match found_index {
        Some(index) => {
            let file = &mut files[index];
            Ok((EvaluationLocation { file }, false))
        }
        None => {
            // If not found, return the first file (for creating a new evaluation)
            if files.is_empty() {
                return Err(ConfigWriterError::EvaluationNotFound {
                    evaluation_name: evaluation_name.to_string(),
                });
            }
            let file = &mut files[0];
            Ok((EvaluationLocation { file }, true))
        }
    }
}

/// Find the config file that contains a specific evaluation (required to exist).
pub fn locate_evaluation_required<'a>(
    files: &'a mut [LoadedConfigFile],
    evaluation_name: &str,
) -> Result<EvaluationLocation<'a>, ConfigWriterError> {
    // First pass: find the index (immutable borrow)
    let found_index = files.iter().enumerate().find_map(|(index, file)| {
        file.document
            .get("evaluations")
            .and_then(|e| e.as_table())
            .filter(|t| t.contains_key(evaluation_name))
            .map(|_| index)
    });

    match found_index {
        Some(index) => {
            let file = &mut files[index];
            Ok(EvaluationLocation { file })
        }
        None => Err(ConfigWriterError::EvaluationNotFound {
            evaluation_name: evaluation_name.to_string(),
        }),
    }
}
