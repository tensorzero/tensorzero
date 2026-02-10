use serde::Serialize;
use toml_edit::{DocumentMut, InlineTable, Item, Table, Value};

use crate::error::ConfigApplierError;

/// Serialize a value to a TOML Item using toml_edit.
pub fn serialize_to_item<T: Serialize>(value: &T) -> Result<Item, ConfigApplierError> {
    // First serialize to toml string, then parse to toml_edit
    let toml_string = toml::to_string(value).map_err(|e| ConfigApplierError::TomlSerialize {
        message: e.to_string(),
    })?;

    let doc: DocumentMut = toml_string.parse().map_err(|e: toml_edit::TomlError| {
        ConfigApplierError::TomlSerialize {
            message: e.to_string(),
        }
    })?;

    // Convert the document to an Item::Table
    Ok(Item::Table(doc.into_table()))
}

/// Recursively convert all `Item::Table` entries within a table to inline tables.
/// This produces compact output like `retries = { num_retries = 3, max_delay_s = 10.0 }`
/// instead of separate `[header]` sections.
///
/// Must be called after `extract_resolved_paths` since path extraction needs regular tables.
pub fn convert_subtables_to_inline(table: &mut Table) {
    let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
    for key in keys {
        let Some(item) = table.get_mut(&key) else {
            continue;
        };
        if let Item::Table(sub_table) = item {
            // Recurse first (bottom-up conversion)
            convert_subtables_to_inline(sub_table);
            // Convert this Table to an InlineTable
            let inline = table_to_inline(sub_table);
            table.insert(&key, Item::Value(Value::InlineTable(inline)));
        }
    }
}

/// Convert a `Table` to an `InlineTable`, preserving all values.
fn table_to_inline(table: &Table) -> InlineTable {
    let mut inline = InlineTable::new();
    for (key, item) in table {
        if let Some(value) = item.as_value() {
            inline.insert(key, value.clone());
        }
    }
    inline
}

/// Ensure a table exists at the given path, creating it if necessary.
/// Returns a mutable reference to the table.
/// Returns an error if an existing entry is not a table.
pub fn ensure_table<'a>(
    doc: &'a mut DocumentMut,
    path: &[&str],
) -> Result<&'a mut Table, ConfigApplierError> {
    let mut current = doc.as_table_mut();

    for &key in path {
        if !current.contains_key(key) {
            current.insert(key, Item::Table(Table::new()));
        }
        current = current[key].as_table_mut().ok_or_else(|| {
            ConfigApplierError::InvalidConfigStructure {
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
) -> Result<(), ConfigApplierError> {
    let variants_table = ensure_table(doc, &["functions", function_name, "variants"])?;
    variants_table.insert(variant_name, variant_item);
    Ok(())
}

/// Upsert an experimentation config into a function.
pub fn upsert_experimentation(
    doc: &mut DocumentMut,
    function_name: &str,
    experimentation_item: Item,
) -> Result<(), ConfigApplierError> {
    let function_table = ensure_table(doc, &["functions", function_name])?;
    function_table.insert("experimentation", experimentation_item);
    Ok(())
}

/// Upsert an evaluation into the evaluations table.
pub fn upsert_evaluation(
    doc: &mut DocumentMut,
    evaluation_name: &str,
    evaluation_item: Item,
) -> Result<(), ConfigApplierError> {
    let evaluations_table = ensure_table(doc, &["evaluations"])?;
    evaluations_table.insert(evaluation_name, evaluation_item);
    Ok(())
}

/// Remove the given keys from a table.
pub fn strip_keys(table: &mut Table, keys: &[&str]) {
    for key in keys {
        table.remove(key);
    }
}

/// Remove entries from a table that are empty inline tables or contain only empty inline tables.
/// This recurses into inline tables first, stripping empty nested tables bottom-up.
/// For example, `timeouts = { non_streaming = {}, streaming = {} }` becomes empty and is removed.
pub fn strip_empty_tables(table: &mut Table) {
    // First, recursively strip empty sub-tables within inline tables
    let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
    for key in &keys {
        if let Some(Item::Value(Value::InlineTable(it))) = table.get_mut(key) {
            strip_empty_inline_tables(it);
        }
    }

    // Then remove any entries that are now empty
    let keys_to_remove: Vec<String> = table
        .iter()
        .filter_map(|(k, v)| match v {
            Item::Value(Value::InlineTable(it)) if it.is_empty() => Some(k.to_string()),
            Item::Table(t) if t.is_empty() => Some(k.to_string()),
            _ => None,
        })
        .collect();
    for key in keys_to_remove {
        table.remove(&key);
    }
}

/// Recursively strip empty inline tables within an inline table.
fn strip_empty_inline_tables(inline: &mut InlineTable) {
    // Recurse into nested inline tables first
    let keys: Vec<String> = inline.iter().map(|(k, _)| k.to_string()).collect();
    for key in &keys {
        if let Some(Value::InlineTable(nested)) = inline.get_mut(key) {
            strip_empty_inline_tables(nested);
        }
    }

    // Remove entries that are now empty inline tables
    let keys_to_remove: Vec<String> = inline
        .iter()
        .filter_map(|(k, v)| match v {
            Value::InlineTable(it) if it.is_empty() => Some(k.to_string()),
            _ => None,
        })
        .collect();
    for key in keys_to_remove {
        inline.remove(&key);
    }
}

/// Upsert an evaluator into an evaluation's evaluators table.
pub fn upsert_evaluator(
    doc: &mut DocumentMut,
    evaluation_name: &str,
    evaluator_name: &str,
    evaluator_item: Item,
) -> Result<(), ConfigApplierError> {
    let evaluators_table = ensure_table(doc, &["evaluations", evaluation_name, "evaluators"])?;
    evaluators_table.insert(evaluator_name, evaluator_item);
    Ok(())
}
