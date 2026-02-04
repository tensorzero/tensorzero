//! Extracts resolved file content from TOML items and prepares them for writing to disk.
//!
//! When TensorZero loads config files, it resolves relative paths (like `system_template = "template.minijinja"`)
//! into inline content with metadata:
//!
//! ```toml
//! system_template = { __tensorzero_remapped_path = "/abs/path/template.minijinja", __data = "template content..." }
//! ```
//!
//! This module reverses that process for the config writer: it extracts the inline content,
//! determines canonical file paths, and rewrites the TOML to use relative path references.
//!
//! # Algorithm
//!
//! The core algorithm walks a TOML tree using patterns from [`TARGET_PATH_COMPONENTS`]:
//!
//! 1. [`extract_resolved_paths`] takes a TOML item and a "matched prefix" describing its position
//!    in the config tree (e.g., `["functions", "my_func", "variants", "my_var"]`).
//!
//! 2. It finds all patterns in `TARGET_PATH_COMPONENTS` that start with this prefix,
//!    then processes each pattern's remaining suffix.
//!
//! 3. [`process_pattern_suffix`] recursively walks the TOML tree:
//!    - **Literal** components navigate directly into that key
//!    - **Wildcard** components enumerate all keys and recurse into each
//!
//! 4. At terminal nodes, if the item contains `__tensorzero_remapped_path` and `__data`,
//!    the content is extracted, a canonical file path is generated, and the TOML item
//!    is replaced with a relative path string.
//!
//! # File Extensions
//!
//! [`file_extension_for_key`] determines the output file extension based on the terminal key:
//!
//! | Key pattern | Extension |
//! |-------------|-----------|
//! | `*_template` | `.minijinja` |
//! | `*_schema`, `parameters` | `.json` |
//! | `system_instructions` | `.txt` |
//! | `user`, `system`, `assistant` | `.minijinja` |
//! | `path` | Uses parent key name + context-based extension |
//!
//! # Public Interface
//!
//! - [`FileToWrite`] - Describes a file to be written (path + content)
//! - [`extract_resolved_paths`] - Main entry point for extracting resolved paths from a TOML item
//! - [`validate_path_component`] - Validates user-provided names are safe path components
//! - [`compute_relative_path`] - Computes relative path between two filesystem locations

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
// Public Functions
// ============================================================================

/// Extract all resolved paths from an item and rewrite them as relative path references.
///
/// This is the main entry point for path extraction. It mutates the TOML `item` in place,
/// replacing resolved path tables with simple string values, and returns a list of files
/// to write.
///
/// # Arguments
///
/// * `item` - The TOML item to process (mutated in place)
/// * `glob_base` - Base directory for output files (typically the config directory)
/// * `toml_file_dir` - Directory containing the TOML file (for relative path calculation)
/// * `matched_prefix` - Position in the config tree, e.g.:
///   - Variant: `["functions", "my_func", "variants", "my_var"]`
///   - Evaluator: `["evaluations", "my_eval", "evaluators", "my_evaluator"]`
///   - Evaluation: `["evaluations", "my_eval"]`
///
/// # Returns
///
/// A list of [`FileToWrite`] describing files that should be written to disk.
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

/// Validates that a name is safe to use as a path component.
///
/// Prevents path traversal attacks by ensuring the input resolves to exactly one
/// "normal" path component. Uses `Path::components()` from the standard library
/// for platform-correct parsing (e.g., backslash is a separator on Windows but
/// a valid character on Unix).
///
/// # Examples
///
/// ```text
/// "../etc"     → rejected (parent directory traversal)
/// "foo/bar"    → rejected (multiple components)
/// ""           → rejected (empty)
/// "my_func"    → accepted
/// ".hidden"    → accepted (valid filename)
/// ```
pub fn validate_path_component(name: &str, field_name: &str) -> Result<(), ConfigWriterError> {
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
// Private: Pattern Matching
// ============================================================================

/// Check if a pattern's first N components match a given prefix.
///
/// Literals must match exactly, wildcards match any prefix component.
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

// ============================================================================
// Private: Tree Walking
// ============================================================================

/// Process a pattern suffix recursively, handling wildcards by enumerating actual keys.
fn process_pattern_suffix(
    item: &mut Item,
    pattern_suffix: &[PathComponent],
    path_so_far: &Path,
    glob_base: &Path,
    toml_file_dir: &Path,
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let mut files = Vec::new();

    let Some(first) = pattern_suffix.first() else {
        return Ok(files);
    };

    match first {
        PathComponent::Literal(key) => {
            let Some(table) = item.as_table_mut() else {
                return Ok(files);
            };
            let Some(nested_item) = table.get_mut(key) else {
                return Ok(files);
            };

            if pattern_suffix.len() == 1 {
                // Terminal element - check for resolved path data
                let key_path = path_so_far.join(key).display().to_string();
                if let Some((content, _original_path)) =
                    extract_resolved_path_data(nested_item, &key_path)?
                {
                    let base_dir = glob_base.join(path_so_far);
                    let absolute_path = canonical_file_path(&base_dir, key);
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

// ============================================================================
// Private: Path Generation
// ============================================================================

/// Check if an Item contains resolved path data (`__tensorzero_remapped_path` and `__data`).
/// If so, extract and return the content and original path.
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
///
/// For most keys, creates `{base_dir}/{key}{ext}` (e.g., `variants/v1/system_template.minijinja`).
/// For `path` keys, creates `{base_dir.parent()}/{name}{ext}` (e.g., `templates/greeting.minijinja`).
fn canonical_file_path(base_dir: &Path, terminal_key: &str) -> PathBuf {
    if let Some(ext) = file_extension_for_key(terminal_key) {
        // Standard keys: {base_dir}/{key}{ext}
        let filename = format!("{terminal_key}{ext}");
        base_dir.join(filename)
    } else {
        // For `path` keys (e.g., templates.greeting.path or schemas.my_schema.path):
        // Output to {grandparent}/{parent_name}{ext} (e.g., templates/greeting.minijinja)
        let parent_name = base_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("file");

        let grandparent = base_dir.parent().unwrap_or(base_dir);
        let grandparent_name = grandparent.file_name().and_then(|s| s.to_str());

        // Currently the config writer only handles templates (variants, evaluators, fusers),
        // not schemas. The schemas case is included for completeness but not actively used.
        let ext = match grandparent_name {
            Some("schemas") => ".json",
            _ => ".minijinja",
        };

        let filename = format!("{parent_name}{ext}");
        grandparent.join(filename)
    }
}

/// Returns the file extension for a given terminal key name.
///
/// - `*_template` → `.minijinja`
/// - `*_schema` or `parameters` → `.json`
/// - `system_instructions` → `.txt`
/// - `user`, `system`, `assistant` (input wrappers) → `.minijinja`
/// - `path` → `None` (handled specially by `canonical_file_path`)
fn file_extension_for_key(key: &str) -> Option<&'static str> {
    if key.ends_with("_template") {
        Some(".minijinja")
    } else if key.ends_with("_schema") || key == "parameters" {
        Some(".json")
    } else if key == "system_instructions" {
        Some(".txt")
    } else if key == "user" || key == "system" || key == "assistant" {
        Some(".minijinja")
    } else {
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // file_extension_for_key tests
    // ========================================================================

    #[test]
    fn test_file_extension_for_template_keys() {
        assert_eq!(
            file_extension_for_key("system_template"),
            Some(".minijinja"),
            "system_template should get .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("user_template"),
            Some(".minijinja"),
            "user_template should get .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("assistant_template"),
            Some(".minijinja"),
            "assistant_template should get .minijinja extension"
        );
    }

    #[test]
    fn test_file_extension_for_schema_keys() {
        assert_eq!(
            file_extension_for_key("system_schema"),
            Some(".json"),
            "system_schema should get .json extension"
        );
        assert_eq!(
            file_extension_for_key("output_schema"),
            Some(".json"),
            "output_schema should get .json extension"
        );
        assert_eq!(
            file_extension_for_key("parameters"),
            Some(".json"),
            "parameters should get .json extension"
        );
    }

    #[test]
    fn test_file_extension_for_system_instructions() {
        assert_eq!(
            file_extension_for_key("system_instructions"),
            Some(".txt"),
            "system_instructions should get .txt extension"
        );
    }

    #[test]
    fn test_file_extension_for_input_wrapper_keys() {
        assert_eq!(
            file_extension_for_key("user"),
            Some(".minijinja"),
            "user (input wrapper) should get .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("system"),
            Some(".minijinja"),
            "system (input wrapper) should get .minijinja extension"
        );
        assert_eq!(
            file_extension_for_key("assistant"),
            Some(".minijinja"),
            "assistant (input wrapper) should get .minijinja extension"
        );
    }

    #[test]
    fn test_file_extension_for_path_key() {
        assert_eq!(
            file_extension_for_key("path"),
            None,
            "path should return None (handled specially)"
        );
    }

    // ========================================================================
    // canonical_file_path tests
    // ========================================================================

    #[test]
    fn test_canonical_file_path_for_template() {
        let base_dir = Path::new("/config/functions/my_func/variants/v1");
        let result = canonical_file_path(base_dir, "system_template");
        assert_eq!(
            result,
            PathBuf::from("/config/functions/my_func/variants/v1/system_template.minijinja"),
            "should create system_template.minijinja in the base directory"
        );
    }

    #[test]
    fn test_canonical_file_path_for_path_key_in_templates() {
        // Simulates: functions.my_func.variants.v1.templates.greeting.path
        // base_dir would be: /config/functions/my_func/variants/v1/templates/greeting
        let base_dir = Path::new("/config/functions/my_func/variants/v1/templates/greeting");
        let result = canonical_file_path(base_dir, "path");
        assert_eq!(
            result,
            PathBuf::from("/config/functions/my_func/variants/v1/templates/greeting.minijinja"),
            "path key under templates should output to templates/<name>.minijinja"
        );
    }

    #[test]
    fn test_canonical_file_path_for_path_key_in_schemas() {
        // Simulates: functions.my_func.schemas.my_schema.path
        // base_dir would be: /config/functions/my_func/schemas/my_schema
        let base_dir = Path::new("/config/functions/my_func/schemas/my_schema");
        let result = canonical_file_path(base_dir, "path");
        assert_eq!(
            result,
            PathBuf::from("/config/functions/my_func/schemas/my_schema.json"),
            "path key under schemas should output to schemas/<name>.json"
        );
    }

    // ========================================================================
    // pattern_matches_prefix tests
    // ========================================================================

    #[test]
    fn test_pattern_matches_prefix_exact_match() {
        let pattern = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("variants"),
            PathComponent::Wildcard,
            PathComponent::Literal("system_template"),
        ];
        let prefix = &["functions", "my_func", "variants", "v1"];
        assert!(
            pattern_matches_prefix(pattern, prefix),
            "pattern should match prefix with wildcard substitution"
        );
    }

    #[test]
    fn test_pattern_matches_prefix_literal_mismatch() {
        let pattern = &[
            PathComponent::Literal("functions"),
            PathComponent::Wildcard,
            PathComponent::Literal("variants"),
        ];
        let prefix = &["evaluations", "my_eval"];
        assert!(
            !pattern_matches_prefix(pattern, prefix),
            "pattern should not match when literal doesn't match"
        );
    }

    #[test]
    fn test_pattern_matches_prefix_too_short() {
        let pattern = &[PathComponent::Literal("functions"), PathComponent::Wildcard];
        let prefix = &["functions", "my_func", "variants"];
        assert!(
            !pattern_matches_prefix(pattern, prefix),
            "pattern should not match when it's shorter than prefix"
        );
    }

    #[test]
    fn test_pattern_matches_prefix_empty_prefix() {
        let pattern = &[PathComponent::Literal("functions"), PathComponent::Wildcard];
        let prefix: &[&str] = &[];
        assert!(
            pattern_matches_prefix(pattern, prefix),
            "empty prefix should match any pattern"
        );
    }

    // ========================================================================
    // extract_resolved_path_data tests
    // ========================================================================

    #[test]
    fn test_extract_resolved_path_data_valid() {
        let mut table = toml_edit::Table::new();
        table.insert(
            "__tensorzero_remapped_path",
            Item::Value(Value::from("/abs/path/template.minijinja")),
        );
        table.insert("__data", Item::Value(Value::from("template content here")));
        let item = Item::Table(table);

        let result = extract_resolved_path_data(&item, "test.path")
            .expect("should not error")
            .expect("should find resolved path data");

        assert_eq!(result.0, "template content here", "should extract __data");
        assert_eq!(
            result.1, "/abs/path/template.minijinja",
            "should extract __tensorzero_remapped_path"
        );
    }

    #[test]
    fn test_extract_resolved_path_data_not_a_table() {
        let item = Item::Value(Value::from("just a string"));
        let result = extract_resolved_path_data(&item, "test.path").expect("should not error");
        assert!(result.is_none(), "non-table item should return None");
    }

    #[test]
    fn test_extract_resolved_path_data_missing_marker() {
        let mut table = toml_edit::Table::new();
        table.insert("some_key", Item::Value(Value::from("some_value")));
        let item = Item::Table(table);

        let result = extract_resolved_path_data(&item, "test.path").expect("should not error");
        assert!(
            result.is_none(),
            "table without __tensorzero_remapped_path should return None"
        );
    }

    #[test]
    fn test_extract_resolved_path_data_missing_data() {
        let mut table = toml_edit::Table::new();
        table.insert(
            "__tensorzero_remapped_path",
            Item::Value(Value::from("/some/path")),
        );
        // Missing __data
        let item = Item::Table(table);

        let result = extract_resolved_path_data(&item, "test.path");
        assert!(
            result.is_err(),
            "table with marker but no __data should error"
        );
    }

    // ========================================================================
    // extract_resolved_paths integration tests
    // ========================================================================

    #[test]
    fn test_extract_resolved_paths_with_system_template() {
        // Build a variant item with a resolved system_template
        let mut variant_table = toml_edit::Table::new();
        variant_table.insert("model", Item::Value(Value::from("gpt-4")));

        let mut template_table = toml_edit::Table::new();
        template_table.insert(
            "__tensorzero_remapped_path",
            Item::Value(Value::from("/original/path/template.minijinja")),
        );
        template_table.insert(
            "__data",
            Item::Value(Value::from("You are a helpful assistant.")),
        );
        variant_table.insert("system_template", Item::Table(template_table));

        let mut item = Item::Table(variant_table);

        let glob_base = Path::new("/config");
        let toml_file_dir = Path::new("/config");
        let matched_prefix = &["functions", "my_func", "variants", "v1"];

        let files = extract_resolved_paths(&mut item, glob_base, toml_file_dir, matched_prefix)
            .expect("should extract paths");

        assert_eq!(files.len(), 1, "should extract one file");
        assert_eq!(
            files[0].absolute_path,
            PathBuf::from("/config/functions/my_func/variants/v1/system_template.minijinja")
        );
        assert_eq!(files[0].content, "You are a helpful assistant.");

        // Check that the TOML was rewritten
        let rewritten = item
            .as_table()
            .unwrap()
            .get("system_template")
            .unwrap()
            .as_str()
            .expect("system_template should be rewritten as string");
        assert_eq!(
            rewritten, "functions/my_func/variants/v1/system_template.minijinja",
            "should rewrite as relative path"
        );
    }

    #[test]
    fn test_extract_resolved_paths_no_resolved_data() {
        // Build a variant item without any resolved paths
        let mut variant_table = toml_edit::Table::new();
        variant_table.insert("model", Item::Value(Value::from("gpt-4")));
        variant_table.insert(
            "system_template",
            Item::Value(Value::from("inline template")),
        );

        let mut item = Item::Table(variant_table);

        let glob_base = Path::new("/config");
        let toml_file_dir = Path::new("/config");
        let matched_prefix = &["functions", "my_func", "variants", "v1"];

        let files = extract_resolved_paths(&mut item, glob_base, toml_file_dir, matched_prefix)
            .expect("should not error");

        assert!(
            files.is_empty(),
            "should return empty list when no resolved paths"
        );
    }

    // ========================================================================
    // compute_relative_path tests
    // ========================================================================

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
        assert!(
            validate_path_component("foo\\bar", "function_name").is_err(),
            "should reject backslash as path separator on Windows"
        );
    }

    #[test]
    #[cfg(not(windows))]
    fn test_validate_path_component_allows_backslash_on_unix() {
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
