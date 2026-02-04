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
use tensorzero_config_paths::{PathComponent, TARGET_PATH_COMPONENTS};

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

// ============================================================================
// Pattern-based Path Utilities
// ============================================================================

/// The prefix pattern for function variant paths in TARGET_PATH_COMPONENTS.
/// Matches: `["functions", *, "variants", *]`
const VARIANT_PREFIX_LEN: usize = 4;

/// The prefix pattern for evaluator variant paths in TARGET_PATH_COMPONENTS.
/// Matches: `["evaluations", *, "evaluators", *, "variants", *]`
const EVALUATOR_PREFIX_LEN: usize = 6;

/// Returns the suffix portions of all variant-related path patterns.
/// These are the path components after `["functions", *, "variants", *]`.
///
/// For example, `["functions", *, "variants", *, "system_template"]` returns `&["system_template"]`.
/// And `["functions", *, "variants", *, "evaluator", "system_template"]` returns `&["evaluator", "system_template"]`.
pub fn variant_path_suffixes() -> impl Iterator<Item = &'static [PathComponent]> {
    TARGET_PATH_COMPONENTS.iter().filter_map(|pattern| {
        if pattern.len() > VARIANT_PREFIX_LEN
            && matches!(pattern[0], PathComponent::Literal("functions"))
            && matches!(pattern[1], PathComponent::Wildcard)
            && matches!(pattern[2], PathComponent::Literal("variants"))
            && matches!(pattern[3], PathComponent::Wildcard)
        {
            Some(&pattern[VARIANT_PREFIX_LEN..])
        } else {
            None
        }
    })
}

/// Returns the suffix portions of all evaluator variant-related path patterns.
/// These are the path components after `["evaluations", *, "evaluators", *, "variants", *]`.
pub fn evaluator_path_suffixes() -> impl Iterator<Item = &'static [PathComponent]> {
    TARGET_PATH_COMPONENTS.iter().filter_map(|pattern| {
        if pattern.len() > EVALUATOR_PREFIX_LEN
            && matches!(pattern[0], PathComponent::Literal("evaluations"))
            && matches!(pattern[1], PathComponent::Wildcard)
            && matches!(pattern[2], PathComponent::Literal("evaluators"))
            && matches!(pattern[3], PathComponent::Wildcard)
            && matches!(pattern[4], PathComponent::Literal("variants"))
            && matches!(pattern[5], PathComponent::Wildcard)
        {
            Some(&pattern[EVALUATOR_PREFIX_LEN..])
        } else {
            None
        }
    })
}

/// Returns the file extension for a given terminal key name.
///
/// - `*_template` → `.minijinja`
/// - `*_schema` or `parameters` → `.json`
/// - `system_instructions` → `.txt`
/// - `user`, `system`, `assistant` (input wrappers) → `.minijinja`
/// - `path` → `None` (preserve original extension)
pub fn file_extension_for_key(key: &str) -> Option<&'static str> {
    if key.ends_with("_template") {
        Some(".minijinja")
    } else if key.ends_with("_schema") || key == "parameters" {
        Some(".json")
    } else if key == "system_instructions" {
        Some(".txt")
    } else if key == "user" || key == "system" || key == "assistant" {
        // input_wrappers keys are minijinja templates
        Some(".minijinja")
    } else {
        // `path` and other keys preserve original extension
        None
    }
}

/// Converts a path suffix (sequence of PathComponents) to a list of literal key names.
/// Returns None if any component is a Wildcard (which shouldn't happen for our suffixes).
pub fn suffix_to_keys(suffix: &[PathComponent]) -> Option<Vec<&'static str>> {
    suffix
        .iter()
        .map(|c| match c {
            PathComponent::Literal(s) => Some(*s),
            PathComponent::Wildcard => None,
        })
        .collect()
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

// ============================================================================
// Private Helpers
// ============================================================================

/// Compute a relative path from a base directory to a target file.
pub fn compute_relative_path(from_dir: &Path, to_file: &Path) -> Result<String, ConfigWriterError> {
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
