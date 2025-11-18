use std::collections::HashMap;

use crate::error::Error;

#[derive(Debug)]
pub struct ConfigSnapshot {
    pub config: String, // serialized as TOML
    pub extra_templates: HashMap<String, String>,
}

impl ConfigSnapshot {
    /// Compute a blake3 hash of this config snapshot
    pub fn hash(&self) -> Result<blake3::Hash, Error> {
        let mut hasher = blake3::Hasher::new();

        // Hash the config string
        hasher.update(self.config.as_bytes());
        hasher.update(&[0]); // null byte separator

        // Hash the extra templates in a deterministic order
        let mut template_names: Vec<_> = self.extra_templates.keys().collect();
        template_names.sort();

        for name in template_names {
            hasher.update(name.as_bytes());
            hasher.update(&[0]); // null byte separator
            hasher.update(self.extra_templates[name].as_bytes());
            hasher.update(&[0]); // null byte separator
        }

        Ok(hasher.finalize())
    }
}

/// Recursively sorts every sub-table of the toml::Table so that the config is
/// stable to rearranging on disk.
/// This should be done prior to hashing.
pub(super) fn prepare_table_for_snapshot(raw_config: toml::Table) -> toml::Table {
    let mut sorted_table = toml::Table::new();

    // Collect and sort keys
    let mut keys: Vec<_> = raw_config.keys().cloned().collect();
    keys.sort();

    // Insert entries in sorted order, recursively processing nested structures
    for key in keys {
        if let Some(value) = raw_config.get(&key) {
            let processed_value = match value {
                toml::Value::Table(table) => {
                    // Recursively sort nested tables
                    toml::Value::Table(prepare_table_for_snapshot(table.clone()))
                }
                toml::Value::Array(array) => {
                    // Process array elements, recursively sorting any tables within
                    let processed_array: Vec<_> = array
                        .iter()
                        .map(|element| match element {
                            toml::Value::Table(table) => {
                                toml::Value::Table(prepare_table_for_snapshot(table.clone()))
                            }
                            other => other.clone(),
                        })
                        .collect();
                    toml::Value::Array(processed_array)
                }
                other => other.clone(),
            };
            sorted_table.insert(key, processed_value);
        }
    }

    sorted_table
}
