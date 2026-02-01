use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigWriterError {
    #[error("IO error for `{path}`: {message}")]
    Io { path: PathBuf, message: String },

    #[error("Function `{function_name}` not found in config")]
    FunctionNotFound { function_name: String },

    #[error("Evaluation `{evaluation_name}` not found in config")]
    EvaluationNotFound { evaluation_name: String },

    #[error("TOML parse error in `{path}`: {message}")]
    TomlParse { path: PathBuf, message: String },

    #[error("TOML serialization error: {message}")]
    TomlSerialize { message: String },

    #[error("Invalid glob pattern `{pattern}`: {message}")]
    InvalidGlob { pattern: String, message: String },

    #[error("Path error: {message}")]
    Path { message: String },

    #[error("Invalid ResolvedTomlPathData: {message}")]
    InvalidResolvedPathData { message: String },

    #[error("Invalid {field_name}: `{value}` - {reason}")]
    InvalidPathComponent {
        field_name: String,
        value: String,
        reason: String,
    },
}

impl ConfigWriterError {
    pub fn io(path: impl Into<PathBuf>, err: impl std::fmt::Display) -> Self {
        Self::Io {
            path: path.into(),
            message: err.to_string(),
        }
    }

    pub fn toml_parse(path: impl Into<PathBuf>, err: impl std::fmt::Display) -> Self {
        Self::TomlParse {
            path: path.into(),
            message: err.to_string(),
        }
    }
}
