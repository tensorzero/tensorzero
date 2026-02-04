use std::path::{Path, PathBuf};

use serde::Serialize;
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::error::ConfigWriterError;
use crate::path_resolver::{
    FileToWrite, compute_relative_path, evaluator_path_suffixes, file_extension_for_key,
    suffix_to_keys, variant_path_suffixes,
};

/// Serialize a value to a TOML Item using toml_edit.
pub fn serialize_to_item<T: Serialize>(value: &T) -> Result<Item, ConfigWriterError> {
    // First serialize to toml string, then parse to toml_edit
    let toml_string = toml::to_string(value).map_err(|e| ConfigWriterError::TomlSerialize {
        message: e.to_string(),
    })?;

    let doc: DocumentMut = toml_string.parse().map_err(|e: toml_edit::TomlError| {
        ConfigWriterError::TomlSerialize {
            message: e.to_string(),
        }
    })?;

    // Convert the document to an Item::Table
    Ok(Item::Table(doc.into_table()))
}

/// Ensure a table exists at the given path, creating it if necessary.
/// Returns a mutable reference to the table.
/// Returns an error if an existing entry is not a table.
pub fn ensure_table<'a>(
    doc: &'a mut DocumentMut,
    path: &[&str],
) -> Result<&'a mut Table, ConfigWriterError> {
    let mut current = doc.as_table_mut();

    for &key in path {
        if !current.contains_key(key) {
            current.insert(key, Item::Table(Table::new()));
        }
        current = current[key].as_table_mut().ok_or_else(|| {
            ConfigWriterError::InvalidConfigStructure {
                path: path.join("."),
                key: key.to_string(),
            }
        })?;
    }

    Ok(current)
}

/// Upsert a variant into a function's variants table.
pub fn upsert_variant(
    doc: &mut DocumentMut,
    function_name: &str,
    variant_name: &str,
    variant_item: Item,
) -> Result<(), ConfigWriterError> {
    let variants_table = ensure_table(doc, &["functions", function_name, "variants"])?;
    variants_table.insert(variant_name, variant_item);
    Ok(())
}

/// Upsert an experimentation config into a function.
pub fn upsert_experimentation(
    doc: &mut DocumentMut,
    function_name: &str,
    experimentation_item: Item,
) -> Result<(), ConfigWriterError> {
    let function_table = ensure_table(doc, &["functions", function_name])?;
    function_table.insert("experimentation", experimentation_item);
    Ok(())
}

/// Upsert an evaluation into the evaluations table.
pub fn upsert_evaluation(
    doc: &mut DocumentMut,
    evaluation_name: &str,
    evaluation_item: Item,
) -> Result<(), ConfigWriterError> {
    let evaluations_table = ensure_table(doc, &["evaluations"])?;
    evaluations_table.insert(evaluation_name, evaluation_item);
    Ok(())
}

/// Upsert an evaluator into an evaluation's evaluators table.
pub fn upsert_evaluator(
    doc: &mut DocumentMut,
    evaluation_name: &str,
    evaluator_name: &str,
    evaluator_item: Item,
) -> Result<(), ConfigWriterError> {
    let evaluators_table = ensure_table(doc, &["evaluations", evaluation_name, "evaluators"])?;
    evaluators_table.insert(evaluator_name, evaluator_item);
    Ok(())
}

/// Navigate to a nested path in a TOML Item and return a mutable reference.
/// Returns None if any intermediate key doesn't exist or isn't a table.
fn get_nested_item_mut<'a>(item: &'a mut Item, keys: &[&str]) -> Option<&'a mut Item> {
    let mut current = item;
    for &key in keys {
        current = current.as_table_mut()?.get_mut(key)?;
    }
    Some(current)
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

/// Generate a canonical file path for a given key and base directory.
/// The extension is derived from the key name.
/// Returns None if suffix_keys is empty (shouldn't happen with valid patterns).
fn canonical_file_path(
    base_dir: &Path,
    suffix_keys: &[&str],
    original_path: &str,
) -> Option<PathBuf> {
    let terminal_key = suffix_keys.last()?;

    // Build the directory path from all keys except the terminal one
    let mut path = base_dir.to_path_buf();
    for &key in &suffix_keys[..suffix_keys.len() - 1] {
        path.push(key);
    }

    // Determine filename with appropriate extension
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

    path.push(filename);
    Some(path)
}

/// Process a single path suffix, extracting resolved path data if present.
fn process_path_suffix(
    item: &mut Item,
    suffix_keys: &[&str],
    base_dir: &Path,
    toml_file_dir: &Path,
) -> Result<Option<FileToWrite>, ConfigWriterError> {
    let key_path = suffix_keys.join(".");

    // Navigate to the nested item
    let Some(nested_item) = get_nested_item_mut(item, suffix_keys) else {
        return Ok(None);
    };

    // Check if it's a ResolvedTomlPathData
    let Some((content, original_path)) = extract_resolved_path_data(nested_item, &key_path)? else {
        return Ok(None);
    };

    // Generate the canonical output path (returns None only for empty suffix_keys, which shouldn't happen)
    let Some(absolute_path) = canonical_file_path(base_dir, suffix_keys, &original_path) else {
        return Ok(None);
    };
    let relative_path = compute_relative_path(toml_file_dir, &absolute_path)?;

    // Replace the table with a simple string value
    *nested_item = Item::Value(Value::from(relative_path.clone()));

    Ok(Some(FileToWrite {
        absolute_path,
        content,
    }))
}

/// Process a TOML Item for a function variant, extracting ResolvedTomlPathData fields
/// and converting them to relative file paths. Returns a list of files that need to be written.
///
/// This function iterates through known path patterns from TARGET_PATH_COMPONENTS,
/// checks if each path exists in the item, and extracts file content where found.
pub fn extract_resolved_paths(
    item: &mut Item,
    glob_base: &Path,
    toml_file_dir: &Path,
    function_name: &str,
    variant_name: &str,
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let mut files_to_write = Vec::new();

    // Build the base directory for this variant's files
    let variant_base = glob_base
        .join("functions")
        .join(function_name)
        .join("variants")
        .join(variant_name);

    // Iterate through all variant path suffixes from TARGET_PATH_COMPONENTS
    for suffix in variant_path_suffixes() {
        let Some(keys) = suffix_to_keys(suffix) else {
            continue; // Skip patterns with wildcards (shouldn't happen)
        };

        if let Some(file_to_write) = process_path_suffix(item, &keys, &variant_base, toml_file_dir)?
        {
            files_to_write.push(file_to_write);
        }
    }

    Ok(files_to_write)
}

/// Similar to extract_resolved_paths, but for evaluator variant contexts.
pub fn extract_resolved_paths_evaluator(
    item: &mut Item,
    glob_base: &Path,
    toml_file_dir: &Path,
    evaluation_name: &str,
    evaluator_name: &str,
    variant_name: &str,
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let mut files_to_write = Vec::new();

    // Build the base directory for this evaluator variant's files
    let evaluator_base = glob_base
        .join("evaluations")
        .join(evaluation_name)
        .join("evaluators")
        .join(evaluator_name)
        .join("variants")
        .join(variant_name);

    // Iterate through all evaluator path suffixes from TARGET_PATH_COMPONENTS
    for suffix in evaluator_path_suffixes() {
        let Some(keys) = suffix_to_keys(suffix) else {
            continue; // Skip patterns with wildcards (shouldn't happen)
        };

        if let Some(file_to_write) =
            process_path_suffix(item, &keys, &evaluator_base, toml_file_dir)?
        {
            files_to_write.push(file_to_write);
        }
    }

    Ok(files_to_write)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_resolved_paths_requires_data() {
        // Create a ResolvedTomlPathData table without __data (only __tensorzero_remapped_path)
        let mut resolved_path_table = Table::new();
        resolved_path_table.insert(
            "__tensorzero_remapped_path",
            Item::Value(Value::from("/path/to/template.minijinja")),
        );
        // Missing __data field

        // Put it under "system_template" key (a known variant path)
        let mut variant_table = Table::new();
        variant_table.insert("system_template", Item::Table(resolved_path_table));
        let mut item = Item::Table(variant_table);

        let result = extract_resolved_paths(
            &mut item,
            Path::new("/config"),
            Path::new("/config"),
            "my_function",
            "variant_a",
        );

        assert!(
            matches!(
                result,
                Err(ConfigWriterError::InvalidResolvedPathData { .. })
            ),
            "expected missing __data to return InvalidResolvedPathData"
        );
    }
}
