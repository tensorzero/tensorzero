use std::path::Path;

use serde::Serialize;
use toml_edit::{DocumentMut, Item, Table, Value};

use crate::error::ConfigWriterError;
use crate::path_resolver::{FileToWrite, VariantPathContext};

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
#[expect(clippy::expect_used)]
pub fn ensure_table<'a>(doc: &'a mut DocumentMut, path: &[&str]) -> &'a mut Table {
    let mut current = doc.as_table_mut();

    for &key in path {
        if !current.contains_key(key) {
            current.insert(key, Item::Table(Table::new()));
        }
        // Safe: we just inserted an empty table if the key didn't exist,
        // so it will always be a table (unless the input is malformed).
        current = current[key]
            .as_table_mut()
            .expect("path component should be a table");
    }

    current
}

/// Upsert a variant into a function's variants table.
pub fn upsert_variant(
    doc: &mut DocumentMut,
    function_name: &str,
    variant_name: &str,
    variant_item: Item,
) {
    let variants_table = ensure_table(doc, &["functions", function_name, "variants"]);
    variants_table.insert(variant_name, variant_item);
}

/// Upsert an experimentation config into a function.
pub fn upsert_experimentation(
    doc: &mut DocumentMut,
    function_name: &str,
    experimentation_item: Item,
) {
    let function_table = ensure_table(doc, &["functions", function_name]);
    function_table.insert("experimentation", experimentation_item);
}

/// Upsert an evaluation into the evaluations table.
pub fn upsert_evaluation(doc: &mut DocumentMut, evaluation_name: &str, evaluation_item: Item) {
    let evaluations_table = ensure_table(doc, &["evaluations"]);
    evaluations_table.insert(evaluation_name, evaluation_item);
}

/// Upsert an evaluator into an evaluation's evaluators table.
pub fn upsert_evaluator(
    doc: &mut DocumentMut,
    evaluation_name: &str,
    evaluator_name: &str,
    evaluator_item: Item,
) {
    let evaluators_table = ensure_table(doc, &["evaluations", evaluation_name, "evaluators"]);
    evaluators_table.insert(evaluator_name, evaluator_item);
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
    let mut files_to_write = Vec::new();

    match item {
        Item::Table(table) => {
            // Check if this table is a ResolvedTomlPathData
            if table.contains_key("__tensorzero_remapped_path") && table.contains_key("__data") {
                let data = table
                    .get("__data")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // Determine the file type based on the current key
                let ctx = VariantPathContext {
                    glob_base,
                    toml_file_dir,
                    function_name,
                    variant_name,
                };

                let (absolute_path, relative_path) = match current_key {
                    Some("system_template") => ctx.template_path("system_template")?,
                    Some("user_template") => ctx.template_path("user_template")?,
                    Some("assistant_template") => ctx.template_path("assistant_template")?,
                    Some("system_instructions") => ctx.system_instructions_path()?,
                    Some("system_schema") => ctx.schema_path("system_schema")?,
                    Some("user_schema") => ctx.schema_path("user_schema")?,
                    Some("assistant_schema") => ctx.schema_path("assistant_schema")?,
                    Some("output_schema") => ctx.schema_path("output_schema")?,
                    _ => {
                        // Default to using the original path structure
                        let original_path = table
                            .get("__tensorzero_remapped_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let filename = Path::new(original_path)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("file");
                        let absolute = glob_base
                            .join("functions")
                            .join(function_name)
                            .join("variants")
                            .join(variant_name)
                            .join(filename);
                        let relative = pathdiff::diff_paths(&absolute, toml_file_dir)
                            .ok_or_else(|| ConfigWriterError::Path {
                                message: format!(
                                    "Cannot compute relative path from `{}` to `{}`",
                                    toml_file_dir.display(),
                                    absolute.display()
                                ),
                            })?
                            .to_str()
                            .ok_or_else(|| ConfigWriterError::Path {
                                message: "Path contains invalid UTF-8".to_string(),
                            })?
                            .to_string();
                        (absolute, relative)
                    }
                };

                files_to_write.push(FileToWrite {
                    absolute_path,
                    content: data,
                });

                // Replace the table with a simple string value
                *item = Item::Value(Value::from(relative_path));
            } else {
                // Recursively process nested tables
                let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
                for key in keys {
                    if let Some(inner_item) = table.get_mut(&key) {
                        let nested_files = extract_resolved_paths(
                            inner_item,
                            glob_base,
                            toml_file_dir,
                            function_name,
                            variant_name,
                            Some(&key),
                        )?;
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
                        let nested_files = extract_resolved_paths(
                            inner_item,
                            glob_base,
                            toml_file_dir,
                            function_name,
                            variant_name,
                            Some(&key),
                        )?;
                        files_to_write.extend(nested_files);
                    }
                }
            }
        }
        _ => {}
    }

    Ok(files_to_write)
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
    use crate::path_resolver::EvaluatorPathContext;

    let mut files_to_write = Vec::new();

    match item {
        Item::Table(table) => {
            // Check if this table is a ResolvedTomlPathData
            if table.contains_key("__tensorzero_remapped_path") && table.contains_key("__data") {
                let data = table
                    .get("__data")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let ctx = EvaluatorPathContext {
                    glob_base,
                    toml_file_dir,
                    evaluation_name,
                    evaluator_name,
                    variant_name,
                };

                let (absolute_path, relative_path) = match current_key {
                    Some("system_instructions") => ctx.system_instructions_path()?,
                    Some("system_template") => ctx.template_path("system_template")?,
                    Some("user_template") => ctx.template_path("user_template")?,
                    Some("assistant_template") => ctx.template_path("assistant_template")?,
                    _ => {
                        // Default to using the original path structure
                        let original_path = table
                            .get("__tensorzero_remapped_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let filename = Path::new(original_path)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("file");
                        let absolute = glob_base
                            .join("evaluations")
                            .join(evaluation_name)
                            .join("evaluators")
                            .join(evaluator_name)
                            .join("variants")
                            .join(variant_name)
                            .join(filename);
                        let relative = pathdiff::diff_paths(&absolute, toml_file_dir)
                            .ok_or_else(|| ConfigWriterError::Path {
                                message: format!(
                                    "Cannot compute relative path from `{}` to `{}`",
                                    toml_file_dir.display(),
                                    absolute.display()
                                ),
                            })?
                            .to_str()
                            .ok_or_else(|| ConfigWriterError::Path {
                                message: "Path contains invalid UTF-8".to_string(),
                            })?
                            .to_string();
                        (absolute, relative)
                    }
                };

                files_to_write.push(FileToWrite {
                    absolute_path,
                    content: data,
                });

                // Replace the table with a simple string value
                *item = Item::Value(Value::from(relative_path));
            } else {
                // Recursively process nested tables
                let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
                for key in keys {
                    if let Some(inner_item) = table.get_mut(&key) {
                        let nested_files = extract_resolved_paths_evaluator(
                            inner_item,
                            glob_base,
                            toml_file_dir,
                            evaluation_name,
                            evaluator_name,
                            variant_name,
                            Some(&key),
                        )?;
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
                        let nested_files = extract_resolved_paths_evaluator(
                            inner_item,
                            glob_base,
                            toml_file_dir,
                            evaluation_name,
                            evaluator_name,
                            variant_name,
                            Some(&key),
                        )?;
                        files_to_write.extend(nested_files);
                    }
                }
            }
        }
        _ => {}
    }

    Ok(files_to_write)
}
