use std::path::Path;

use serde::Serialize;
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::error::ConfigWriterError;
use crate::path_resolver::{EvaluatorPathContext, FileToWrite, PathContext, VariantPathContext};

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

/// Internal implementation that extracts resolved paths using a PathContext.
/// This is the core logic shared by both variant and evaluator path extraction.
fn extract_resolved_paths_impl<C: PathContext>(
    item: &mut Item,
    ctx: &C,
    current_key: Option<&str>,
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let mut files_to_write = Vec::new();

    match item {
        Item::Table(table) => {
            // Check if this table is a ResolvedTomlPathData
            if table.contains_key("__tensorzero_remapped_path") {
                let data = table
                    .get("__data")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ConfigWriterError::InvalidResolvedPathData {
                        message: format!(
                            "`{}` must contain string `__data`",
                            current_key.unwrap_or("<root>")
                        ),
                    })?;

                // Try to resolve the key using the context, fall back to original path
                let (absolute_path, relative_path) =
                    if let Some(result) = current_key.and_then(|k| ctx.resolve_key(k)) {
                        result?
                    } else {
                        // Default to using the original path structure
                        let original_path = table
                            .get("__tensorzero_remapped_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let filename = Path::new(original_path)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("file");
                        ctx.fallback_path(filename)?
                    };

                files_to_write.push(FileToWrite {
                    absolute_path,
                    content: data.to_string(),
                });

                // Replace the table with a simple string value
                *item = Item::Value(Value::from(relative_path));
            } else {
                // Recursively process nested tables
                let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
                for key in keys {
                    if let Some(inner_item) = table.get_mut(&key) {
                        let nested_files =
                            extract_resolved_paths_impl(inner_item, ctx, Some(&key))?;
                        files_to_write.extend(nested_files);
                    }
                }
            }
        }
        Item::ArrayOfTables(array) => {
            for table in array.iter_mut() {
                let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
                for key in keys {
                    if let Some(inner_item) = table.get_mut(&key) {
                        let nested_files =
                            extract_resolved_paths_impl(inner_item, ctx, Some(&key))?;
                        files_to_write.extend(nested_files);
                    }
                }
            }
        }
        _ => {}
    }

    Ok(files_to_write)
}

/// Process a TOML Item, extracting ResolvedTomlPathData fields and converting them
/// to relative file paths. Returns a list of files that need to be written.
///
/// This function recursively walks the Item looking for tables with
/// `__tensorzero_remapped_path` and `__data` fields, extracts the content,
/// and replaces them with relative path strings.
pub fn extract_resolved_paths(
    item: &mut Item,
    glob_base: &Path,
    toml_file_dir: &Path,
    function_name: &str,
    variant_name: &str,
    current_key: Option<&str>,
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let ctx = VariantPathContext {
        glob_base,
        toml_file_dir,
        function_name,
        variant_name,
    };
    extract_resolved_paths_impl(item, &ctx, current_key)
}

/// Similar to extract_resolved_paths, but for evaluator contexts.
pub fn extract_resolved_paths_evaluator(
    item: &mut Item,
    glob_base: &Path,
    toml_file_dir: &Path,
    evaluation_name: &str,
    evaluator_name: &str,
    variant_name: &str,
    current_key: Option<&str>,
) -> Result<Vec<FileToWrite>, ConfigWriterError> {
    let ctx = EvaluatorPathContext {
        glob_base,
        toml_file_dir,
        evaluation_name,
        evaluator_name,
        variant_name,
    };
    extract_resolved_paths_impl(item, &ctx, current_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_resolved_paths_requires_data() {
        let mut table = Table::new();
        table.insert(
            "__tensorzero_remapped_path",
            Item::Value(Value::from("inline")),
        );
        let mut item = Item::Table(table);

        let result = extract_resolved_paths(
            &mut item,
            Path::new("."),
            Path::new("."),
            "my_function",
            "variant_a",
            Some("system_template"),
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
