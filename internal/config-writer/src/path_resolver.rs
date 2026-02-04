//! Path resolution utilities for the config writer.
//!
//! This module provides utilities for:
//! - Validating user-provided names are safe to use as path components
//! - Generating standardized file paths for variants and evaluators
//! - Computing relative paths between directories
//!
//! # Path Validation
//!
//! [`validate_path_component`] prevents path traversal attacks by ensuring names resolve
//! to exactly one "normal" path component. It uses `Path::components()` from the standard
//! library rather than manual character checks, which correctly handles platform-specific
//! path parsing (e.g., backslash is a separator on Windows but a valid character on Unix).
//!
//! # Path Contexts
//!
//! [`VariantPathContext`] and [`EvaluatorPathContext`] generate standardized paths for
//! config files. They implement [`PathContext`] which provides a uniform interface for
//! resolving template/schema keys to file paths.

use std::path::{Component, Path, PathBuf};

use crate::error::ConfigWriterError;

// ============================================================================
// Public Types
// ============================================================================

/// Information about a template or schema file that needs to be written.
#[derive(Debug, Clone)]
pub struct FileToWrite {
    /// The absolute path where the file should be written
    pub absolute_path: PathBuf,
    /// The content to write
    pub content: String,
}

/// Trait for path contexts that can resolve keys to file paths.
///
/// This allows generic handling of path resolution for both variants and evaluators.
pub trait PathContext {
    /// Resolve a key to an absolute and relative path.
    ///
    /// Returns `Some(Ok(...))` if the key is recognized.
    /// Returns `Some(Err(...))` if the key is recognized but path computation fails.
    /// Returns `None` if the key is not recognized (will use fallback).
    fn resolve_key(&self, key: &str) -> Option<Result<(PathBuf, String), ConfigWriterError>>;

    /// Generate a fallback path for an unknown key using the original filename.
    fn fallback_path(&self, filename: &str) -> Result<(PathBuf, String), ConfigWriterError>;
}

/// Context for generating standardized paths for a variant.
pub struct VariantPathContext<'a> {
    pub glob_base: &'a Path,
    pub toml_file_dir: &'a Path,
    pub function_name: &'a str,
    pub variant_name: &'a str,
}

/// Context for generating standardized paths for an evaluator variant.
pub struct EvaluatorPathContext<'a> {
    pub glob_base: &'a Path,
    pub toml_file_dir: &'a Path,
    pub evaluation_name: &'a str,
    pub evaluator_name: &'a str,
    pub variant_name: &'a str,
}

// ============================================================================
// Public Functions
// ============================================================================

/// Validates that a name is safe to use as a path component.
///
/// This function prevents path traversal attacks by ensuring the input resolves to
/// exactly one "normal" path component. We use `Path::components()` from the standard
/// library rather than manually checking for specific characters because it correctly
/// handles all platform-specific path parsing:
///
/// - `..` -> `[ParentDir]` — rejected (not a Normal component)
/// - `../foo` -> `[ParentDir, Normal("foo")]` — rejected (multiple components)
/// - `foo/../bar` -> `[Normal("foo"), ParentDir, Normal("bar")]` — rejected (multiple components)
/// - `./foo` -> `[CurDir, Normal("foo")]` — rejected (multiple components)
/// - `foo/bar` -> `[Normal("foo"), Normal("bar")]` — rejected (multiple components)
/// - `/foo` -> `[RootDir, Normal("foo")]` — rejected (multiple components)
/// - `my_function` -> `[Normal("my_function")]` — accepted
///
/// The check `(Some(Component::Normal(os_str)), None) if os_str == name` ensures:
/// 1. There is exactly one component
/// 2. That component is `Normal` (not `ParentDir`, `CurDir`, `RootDir`, or `Prefix`)
/// 3. The component equals the original input (guards against normalization surprises)
pub fn validate_path_component(name: &str, field_name: &str) -> Result<(), ConfigWriterError> {
    // Use the stdlib to verify this is a single, normal path component.
    // This handles empty strings (no components), path separators (multiple components),
    // and special components like `.` (CurDir) and `..` (ParentDir).
    let path = Path::new(name);
    let mut components = path.components();

    match (components.next(), components.next()) {
        (Some(Component::Normal(os_str)), None) if os_str == name => Ok(()),
        _ => Err(ConfigWriterError::InvalidPathComponent {
            field_name: field_name.to_string(),
            value: name.to_string(),
            reason: "name must be a valid single path component".to_string(),
        }),
    }
}

impl<'a> VariantPathContext<'a> {
    /// Generate a standardized path for a variant template file.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn template_path(
        &self,
        template_kind: &str,
    ) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("functions")
            .join(self.function_name)
            .join("variants")
            .join(self.variant_name)
            .join(format!("{template_kind}.minijinja"));

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    /// Generate a standardized path for a variant schema file.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn schema_path(&self, schema_kind: &str) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("functions")
            .join(self.function_name)
            .join("variants")
            .join(self.variant_name)
            .join(format!("{schema_kind}.json"));

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    /// Generate a standardized path for system_instructions.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn system_instructions_path(&self) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("functions")
            .join(self.function_name)
            .join("variants")
            .join(self.variant_name)
            .join("system_instructions.txt");

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    fn variant_base_path(&self) -> PathBuf {
        self.glob_base
            .join("functions")
            .join(self.function_name)
            .join("variants")
            .join(self.variant_name)
    }
}

impl PathContext for VariantPathContext<'_> {
    fn resolve_key(&self, key: &str) -> Option<Result<(PathBuf, String), ConfigWriterError>> {
        match key {
            "system_template" | "user_template" | "assistant_template" => {
                Some(self.template_path(key))
            }
            "system_instructions" => Some(self.system_instructions_path()),
            "system_schema" | "user_schema" | "assistant_schema" | "output_schema" => {
                Some(self.schema_path(key))
            }
            _ => None,
        }
    }

    fn fallback_path(&self, filename: &str) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self.variant_base_path().join(filename);
        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }
}

// ============================================================================
// EvaluatorPathContext Implementation
// ============================================================================

impl<'a> EvaluatorPathContext<'a> {
    /// Generate a standardized path for an evaluator variant's system_instructions.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn system_instructions_path(&self) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("evaluations")
            .join(self.evaluation_name)
            .join("evaluators")
            .join(self.evaluator_name)
            .join("variants")
            .join(self.variant_name)
            .join("system_instructions.txt");

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    /// Generate a standardized path for an evaluator variant template.
    /// Returns (absolute_path, relative_path_from_toml).
    pub fn template_path(
        &self,
        template_kind: &str,
    ) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self
            .glob_base
            .join("evaluations")
            .join(self.evaluation_name)
            .join("evaluators")
            .join(self.evaluator_name)
            .join("variants")
            .join(self.variant_name)
            .join(format!("{template_kind}.minijinja"));

        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }

    fn evaluator_variant_base_path(&self) -> PathBuf {
        self.glob_base
            .join("evaluations")
            .join(self.evaluation_name)
            .join("evaluators")
            .join(self.evaluator_name)
            .join("variants")
            .join(self.variant_name)
    }
}

impl PathContext for EvaluatorPathContext<'_> {
    fn resolve_key(&self, key: &str) -> Option<Result<(PathBuf, String), ConfigWriterError>> {
        match key {
            "system_template" | "user_template" | "assistant_template" => {
                Some(self.template_path(key))
            }
            "system_instructions" => Some(self.system_instructions_path()),
            _ => None,
        }
    }

    fn fallback_path(&self, filename: &str) -> Result<(PathBuf, String), ConfigWriterError> {
        let absolute = self.evaluator_variant_base_path().join(filename);
        let relative = compute_relative_path(self.toml_file_dir, &absolute)?;
        Ok((absolute, relative))
    }
}

// ============================================================================
// Private Helpers
// ============================================================================

/// Compute a relative path from a base directory to a target file.
fn compute_relative_path(from_dir: &Path, to_file: &Path) -> Result<String, ConfigWriterError> {
    pathdiff::diff_paths(to_file, from_dir)
        .ok_or_else(|| ConfigWriterError::Path {
            message: format!(
                "Cannot compute relative path from `{}` to `{}`",
                from_dir.display(),
                to_file.display()
            ),
        })?
        .to_str()
        .ok_or_else(|| ConfigWriterError::Path {
            message: format!("Path contains invalid UTF-8: {}", to_file.display()),
        })
        .map(|s| s.to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_relative_path() {
        let from = Path::new("/project/config");
        let to = Path::new("/project/config/functions/my_fn/variants/v1/template.minijinja");
        let result = compute_relative_path(from, to).expect("should compute relative path");
        assert_eq!(result, "functions/my_fn/variants/v1/template.minijinja");
    }

    #[test]
    fn test_validate_path_component_rejects_invalid() {
        assert!(
            validate_path_component("../etc", "function_name").is_err(),
            "should reject parent directory traversal"
        );
        assert!(
            validate_path_component("foo/bar", "function_name").is_err(),
            "should reject forward slash"
        );
        assert!(
            validate_path_component("", "function_name").is_err(),
            "should reject empty string"
        );
        assert!(
            validate_path_component("..", "function_name").is_err(),
            "should reject parent directory reference"
        );
        assert!(
            validate_path_component(".", "function_name").is_err(),
            "should reject current directory reference"
        );
        assert!(
            validate_path_component("foo/../bar", "function_name").is_err(),
            "should reject embedded parent directory traversal"
        );
    }

    #[test]
    #[cfg(windows)]
    fn test_validate_path_component_rejects_backslash_on_windows() {
        // On Windows, backslash is a path separator
        assert!(
            validate_path_component("foo\\bar", "function_name").is_err(),
            "should reject backslash as path separator on Windows"
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn test_validate_path_component_allows_backslash_on_unix() {
        // On Unix, backslash is a valid filename character (though unusual)
        assert!(
            validate_path_component("foo\\bar", "function_name").is_ok(),
            "backslash is a valid filename character on Unix"
        );
    }

    #[test]
    fn test_validate_path_component_allows_valid_names() {
        assert!(
            validate_path_component("my_function", "function_name").is_ok(),
            "should allow underscores"
        );
        assert!(
            validate_path_component("my-variant", "variant_name").is_ok(),
            "should allow hyphens"
        );
        assert!(
            validate_path_component("eval_v1", "evaluation_name").is_ok(),
            "should allow alphanumeric with underscores"
        );
        assert!(
            validate_path_component("judge-1", "evaluator_name").is_ok(),
            "should allow alphanumeric with hyphens"
        );
        assert!(
            validate_path_component("CamelCase", "function_name").is_ok(),
            "should allow mixed case"
        );
        assert!(
            validate_path_component("name123", "function_name").is_ok(),
            "should allow trailing numbers"
        );
        // Hidden files (dot prefix) are valid filenames
        assert!(
            validate_path_component(".hidden", "function_name").is_ok(),
            "should allow dot prefix (hidden files)"
        );
        assert!(
            validate_path_component("..hidden", "function_name").is_ok(),
            "should allow double dot prefix (valid filename, not parent dir)"
        );
    }
}
