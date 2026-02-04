use serde::Serialize;
use toml_edit::{DocumentMut, Item, Table};

use crate::error::ConfigWriterError;

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
