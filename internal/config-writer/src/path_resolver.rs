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

use toml_edit::{Item, Value};

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

/// Check if a pattern's first N components match a given prefix.
///
/// Literals must match exactly, wildcards match any prefix component.
/// For example, pattern `["functions", *, "variants", *, "system_template"]`
/// matches prefix `["functions", "my_func", "variants", "v1"]`.
fn pattern_matches_prefix(pattern: &[PathComponent], prefix: &[&str]) -> bool {
    if pattern.len() < prefix.len() {
        return false;
    }
    pattern
        .iter()
        .zip(prefix.iter())
        .all(|(component, key)| match component {
            PathComponent::Wildcard => true,
            PathComponent::Literal(lit) => *lit == *key,
        })
}

/// Returns the file extension for a given terminal key name.
///
/// - `*_template` → `.minijinja`
/// - `*_schema` or `parameters` → `.json`
/// - `system_instructions` → `.txt`
/// - `user`, `system`, `assistant` (input wrappers) → `.minijinja`
/// - `path` → `None` (preserve original extension)
fn file_extension_for_key(key: &str) -> Option<&'static str> {
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

/// Check if an Item is a ResolvedTomlPathData (has __tensorzero_remapped_path and __data).
/// If so, extract the data and return it along with the original path.
fn extract_resolved_path_data(
    item: &Item,
    key_path: &str,
) -> Result<Option<(String, String)>, ConfigWriterError> {
    let Some(table) = item.as_table() else {
        return Ok(None);
    };

    if !table.contains_key("__tensorzero_remapped_path") {
        return Ok(None);
    }

    let data = table
        .get("__data")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ConfigWriterError::InvalidResolvedPathData {
            message: format!("`{key_path}` must contain string `__data`"),
        })?;

    let original_path = table
        .get("__tensorzero_remapped_path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    Ok(Some((data.to_string(), original_path.to_string())))
}

/// Generate a canonical file path for a given terminal key and base directory.
/// The extension is derived from the key name.
fn canonical_file_path(base_dir: &Path, terminal_key: &str, original_path: &str) -> PathBuf {
    let filename = if let Some(ext) = file_extension_for_key(terminal_key) {
        format!("{terminal_key}{ext}")
    } else {
        // For `path` keys, preserve original filename
        Path::new(original_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("file")
            .to_string()
    };

    base_dir.join(filename)
}

/// Process a pattern suffix recursively, handling wildcards by enumerating actual keys.
///
/// - `item`: The current TOML item to process
/// - `pattern_suffix`: Remaining pattern components to process
/// - `path_so_far`: Accumulated path components for output file path
/// - `glob_base`: Base directory for output files
/// - `toml_file_dir`: Directory containing the TOML file (for relative path calculation)
fn process_pattern_suffix(
    item: &mut Item,
    pattern_suffix: &[PathComponent],
    path_so_far: &Path,
    glob_base: &Path,
    toml_file_dir: &Path,
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let mut files = Vec::new();

    let Some(first) = pattern_suffix.first() else {
        // Empty suffix - nothing to process
        return Ok(files);
    };

    match first {
        PathComponent::Literal(key) => {
            // Navigate into this key
            let Some(table) = item.as_table_mut() else {
                return Ok(files);
            };
            let Some(nested_item) = table.get_mut(key) else {
                return Ok(files);
            };

            if pattern_suffix.len() == 1 {
                // Terminal element - check for __tensorzero_remapped_path
                let key_path = path_so_far.join(key).display().to_string();
                if let Some((content, original_path)) =
                    extract_resolved_path_data(nested_item, &key_path)?
                {
                    let base_dir = glob_base.join(path_so_far);
                    let absolute_path = canonical_file_path(&base_dir, key, &original_path);
                    let relative_path = compute_relative_path(toml_file_dir, &absolute_path)?;

                    // Replace the table with a simple string value
                    *nested_item = Item::Value(Value::from(relative_path));

                    files.push(FileToWrite {
                        absolute_path,
                        content,
                    });
                }
            } else {
                // More components to process - recurse
                let new_path = path_so_far.join(key);
                files.extend(process_pattern_suffix(
                    nested_item,
                    &pattern_suffix[1..],
                    &new_path,
                    glob_base,
                    toml_file_dir,
                )?);
            }
        }
        PathComponent::Wildcard => {
            // Enumerate all keys in the current table
            let Some(table) = item.as_table_mut() else {
                return Ok(files);
            };

            let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
            for key in keys {
                if let Some(nested_item) = table.get_mut(&key) {
                    let new_path = path_so_far.join(&key);
                    files.extend(process_pattern_suffix(
                        nested_item,
                        &pattern_suffix[1..],
                        &new_path,
                        glob_base,
                        toml_file_dir,
                    )?);
                }
            }
        }
    }

    Ok(files)
}

/// Extract all resolved paths from an item.
///
/// `matched_prefix` describes the position in the config tree:
/// - For variant: `["functions", "my_func", "variants", "my_var"]`
/// - For evaluator: `["evaluations", "my_eval", "evaluators", "my_evaluator"]`
/// - For evaluation: `["evaluations", "my_eval"]`
///
/// This function finds all patterns in TARGET_PATH_COMPONENTS that match the given prefix,
/// then processes the remaining suffix. Wildcards in the suffix are handled by enumerating
/// actual keys in the TOML item.
pub fn extract_resolved_paths(
    item: &mut Item,
    glob_base: &Path,
    toml_file_dir: &Path,
    matched_prefix: &[&str],
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let mut files = Vec::new();

    // Build the base path from the prefix
    let base_path: PathBuf = matched_prefix.iter().collect();

    // Find all patterns matching this prefix and process their suffixes
    for pattern in TARGET_PATH_COMPONENTS {
        if pattern_matches_prefix(pattern, matched_prefix) && pattern.len() > matched_prefix.len() {
            let suffix = &pattern[matched_prefix.len()..];
            files.extend(process_pattern_suffix(
                item,
                suffix,
                &base_path,
                glob_base,
                toml_file_dir,
            )?);
        }
    }

    Ok(files)
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
