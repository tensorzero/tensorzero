use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, ErrorDetails};

use super::stored::StoredConfig;
use super::UninitializedConfig;

/// A serializable snapshot of a config suitable for storage in the database.
///
/// This struct holds the parts of a config that need to be persisted to the `ConfigSnapshot`
/// table in ClickHouse. It serves two main purposes:
///
/// 1. **Config Version Tracking**: Each unique config gets a deterministic hash. This hash is
///    stored with each inference request, allowing you to correlate inference results with the
///    exact config that was active at the time.
///
/// 2. **Config History**: All historical configs are preserved in the database, enabling:
///    - Reproducing past behavior
///    - Understanding config evolution over time
///    - Debugging issues by comparing configs
///
/// # Fields
///
/// - `config`: The parsed config as a `StoredConfig` (will be serialized to TOML for storage)
/// - `hash`: A deterministic Blake3 hash computed from the TOML and templates
/// - `extra_templates`: Templates loaded from the filesystem (not in TOML)
///
/// # Templates in ConfigSnapshot
///
/// **IMPORTANT**: The `extra_templates` in this struct are **only used for database storage
/// and hash computation**. They are NOT used at runtime by the gateway.
///
/// - At runtime, the gateway uses `Config.templates` (a `TemplateConfig` with compiled MiniJinja templates)
/// - The `extra_templates` here are just the raw template strings that were loaded from disk
/// - They're stored in the database to preserve the complete config state for reproducibility
///
/// # Hash Computation
///
/// The hash is computed from:
/// 1. The TOML config (after sorting keys for determinism via `prepare_table_for_snapshot()`)
/// 2. The extra templates (sorted by name for determinism)
///
/// This ensures that any change to the config or templates produces a different hash.
///
/// # Usage
///
/// This is typically created during config loading and then written to the database:
///
/// ```ignore
/// // During config loading (in Config::load_from_toml)
/// let snapshot = ConfigSnapshot::new(sorted_table, extra_templates)?;
///
/// // Later, after database connection is established
/// write_config_snapshot(&clickhouse, snapshot).await?;
/// ```
#[expect(clippy::manual_non_exhaustive)]
#[derive(Debug)]
pub struct ConfigSnapshot {
    /// The config in a form suitable for serialization to TOML for database storage.
    /// Uses `StoredConfig` instead of `UninitializedConfig` to support backward-compatible
    /// deserialization of historical snapshots (see `stored.rs` for details).
    pub config: StoredConfig,

    /// A deterministic Blake3 hash of the config TOML and templates.
    /// This uniquely identifies this config version.
    pub hash: SnapshotHash,

    /// Templates that were loaded from the filesystem (e.g., prompt templates).
    /// These are stored separately from the TOML config itself.
    ///
    /// **NOTE**: These templates are for database storage only. At runtime, the gateway
    /// uses the compiled templates in `Config.templates`, not these raw strings.
    pub extra_templates: HashMap<String, String>,

    __private: (),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnapshotHash(Arc<str>);

impl std::fmt::Display for SnapshotHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl SnapshotHash {
    pub fn new_test() -> SnapshotHash {
        let hash = blake3::hash(&[]);
        let big_int = BigUint::from_bytes_be(hash.as_bytes());
        SnapshotHash(Arc::from(big_int.to_string()))
    }
}

impl ConfigSnapshot {
    pub fn new(
        sorted_config_toml: toml::Table,
        extra_templates: HashMap<String, String>,
    ) -> Result<Self, Error> {
        let config = UninitializedConfig::try_from(sorted_config_toml.clone())?;
        let hash = ConfigSnapshot::hash(&sorted_config_toml, &extra_templates)?;
        Ok(Self {
            config: config.into(),
            hash,
            extra_templates,
            __private: (),
        })
    }

    /// Create a ConfigSnapshot from a TOML string for testing.
    /// Parses the string, computes the hash, and stores the config.
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn new_from_toml_string(
        config_toml: &str,
        extra_templates: HashMap<String, String>,
    ) -> Result<Self, Error> {
        use super::snapshot::prepare_table_for_snapshot;
        let table: toml::Table = config_toml.parse().map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to parse TOML: {e}"),
            })
        })?;
        let sorted_table = prepare_table_for_snapshot(table);
        Self::new(sorted_table, extra_templates)
    }

    /// Create an empty ConfigSnapshot for testing when the actual config doesn't matter.
    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn new_empty_for_test() -> Self {
        Self {
            config: StoredConfig::default(),
            hash: SnapshotHash::new_test(),
            extra_templates: HashMap::new(),
            __private: (),
        }
    }

    /// Compute a blake3 hash of this config snapshot
    fn hash(
        sorted_config_toml: &toml::Table,
        extra_templates: &HashMap<String, String>,
    ) -> Result<SnapshotHash, Error> {
        let mut hasher = blake3::Hasher::new();

        // Serialize and hash the TOML config
        let serialized_config = toml::to_string(sorted_config_toml).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize config for hashing: {e}"),
            })
        })?;
        hasher.update(serialized_config.as_bytes());
        hasher.update(&[0]); // null byte separator

        // Hash the extra templates in a deterministic order
        let mut template_names: Vec<_> = extra_templates.keys().collect();
        template_names.sort();

        for name in template_names {
            hasher.update(name.as_bytes());
            hasher.update(&[0]); // null byte separator
            hasher.update(extra_templates[name].as_bytes());
            hasher.update(&[0]); // null byte separator
        }

        let hash = hasher.finalize();
        // Convert the 32-byte hash to a decimal string
        let big_int = BigUint::from_bytes_be(hash.as_bytes());
        Ok(SnapshotHash(Arc::from(big_int.to_string())))
    }
}

/// Recursively sorts every sub-table of the toml::Table so that the config is
/// stable to rearranging on disk.
/// This should be done prior to hashing.
pub fn prepare_table_for_snapshot(raw_config: toml::Table) -> toml::Table {
    internal_prepare_table_for_snapshot(&raw_config)
}

fn internal_prepare_table_for_snapshot(raw_config: &toml::Table) -> toml::Table {
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
                    toml::Value::Table(internal_prepare_table_for_snapshot(table))
                }
                toml::Value::Array(array) => {
                    // Process array elements, recursively sorting any tables within
                    let processed_array: Vec<_> = array
                        .iter()
                        .map(|element| match element {
                            toml::Value::Table(table) => {
                                toml::Value::Table(internal_prepare_table_for_snapshot(table))
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

#[cfg(test)]
mod tests {
    use super::*;
    use toml::Table;

    #[test]
    fn test_prepare_table_for_snapshot_basic_sorting() {
        let mut table = Table::new();
        table.insert("zebra".to_string(), toml::Value::String("z".to_string()));
        table.insert("alpha".to_string(), toml::Value::String("a".to_string()));
        table.insert("beta".to_string(), toml::Value::String("b".to_string()));

        let sorted = prepare_table_for_snapshot(table);

        let keys: Vec<_> = sorted.keys().collect();
        assert_eq!(keys, vec!["alpha", "beta", "zebra"]);
    }

    #[test]
    fn test_prepare_table_for_snapshot_nested_tables() {
        let mut inner_table = Table::new();
        inner_table.insert("zoo".to_string(), toml::Value::Integer(3));
        inner_table.insert("apple".to_string(), toml::Value::Integer(1));
        inner_table.insert("mango".to_string(), toml::Value::Integer(2));

        let mut outer_table = Table::new();
        outer_table.insert("zebra".to_string(), toml::Value::Table(inner_table));
        outer_table.insert("alpha".to_string(), toml::Value::Integer(42));

        let sorted = prepare_table_for_snapshot(outer_table);

        // Check outer keys are sorted
        let outer_keys: Vec<_> = sorted.keys().collect();
        assert_eq!(outer_keys, vec!["alpha", "zebra"]);

        // Check inner keys are sorted
        if let Some(toml::Value::Table(inner)) = sorted.get("zebra") {
            let inner_keys: Vec<_> = inner.keys().collect();
            assert_eq!(inner_keys, vec!["apple", "mango", "zoo"]);
        } else {
            panic!("Expected nested table");
        }
    }

    #[test]
    fn test_prepare_table_for_snapshot_array_with_tables() {
        let mut table1 = Table::new();
        table1.insert("zulu".to_string(), toml::Value::Integer(1));
        table1.insert("alpha".to_string(), toml::Value::Integer(2));

        let mut table2 = Table::new();
        table2.insert("yankee".to_string(), toml::Value::Integer(3));
        table2.insert("bravo".to_string(), toml::Value::Integer(4));

        let array = vec![toml::Value::Table(table1), toml::Value::Table(table2)];

        let mut outer_table = Table::new();
        outer_table.insert("items".to_string(), toml::Value::Array(array));

        let sorted = prepare_table_for_snapshot(outer_table);

        if let Some(toml::Value::Array(arr)) = sorted.get("items") {
            // Check first table in array is sorted
            if let toml::Value::Table(t1) = &arr[0] {
                let keys1: Vec<_> = t1.keys().collect();
                assert_eq!(keys1, vec!["alpha", "zulu"]);
            } else {
                panic!("Expected table in array");
            }

            // Check second table in array is sorted
            if let toml::Value::Table(t2) = &arr[1] {
                let keys2: Vec<_> = t2.keys().collect();
                assert_eq!(keys2, vec!["bravo", "yankee"]);
            } else {
                panic!("Expected table in array");
            }
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_prepare_table_for_snapshot_mixed_types() {
        let mut table = Table::new();
        table.insert(
            "string".to_string(),
            toml::Value::String("hello".to_string()),
        );
        table.insert("integer".to_string(), toml::Value::Integer(42));
        table.insert("float".to_string(), toml::Value::Float(2.71));
        table.insert("boolean".to_string(), toml::Value::Boolean(true));

        let sorted = prepare_table_for_snapshot(table);

        // Check keys are sorted
        let keys: Vec<_> = sorted.keys().collect();
        assert_eq!(keys, vec!["boolean", "float", "integer", "string"]);

        // Check values are preserved
        assert_eq!(
            sorted.get("string"),
            Some(&toml::Value::String("hello".to_string()))
        );
        assert_eq!(sorted.get("integer"), Some(&toml::Value::Integer(42)));
        assert_eq!(sorted.get("float"), Some(&toml::Value::Float(2.71)));
        assert_eq!(sorted.get("boolean"), Some(&toml::Value::Boolean(true)));
    }

    #[test]
    fn test_prepare_table_for_snapshot_empty_table() {
        let table = Table::new();
        let sorted = prepare_table_for_snapshot(table);
        assert_eq!(sorted.len(), 0);
    }

    #[test]
    fn test_prepare_table_for_snapshot_single_key() {
        let mut table = Table::new();
        table.insert("only".to_string(), toml::Value::String("one".to_string()));

        let sorted = prepare_table_for_snapshot(table);

        assert_eq!(sorted.len(), 1);
        assert_eq!(
            sorted.get("only"),
            Some(&toml::Value::String("one".to_string()))
        );
    }

    #[test]
    fn test_prepare_table_for_snapshot_deep_nesting() {
        let mut level3 = Table::new();
        level3.insert("z3".to_string(), toml::Value::Integer(3));
        level3.insert("a3".to_string(), toml::Value::Integer(1));

        let mut level2 = Table::new();
        level2.insert("z2".to_string(), toml::Value::Table(level3));
        level2.insert("a2".to_string(), toml::Value::Integer(2));

        let mut level1 = Table::new();
        level1.insert("z1".to_string(), toml::Value::Table(level2));
        level1.insert("a1".to_string(), toml::Value::Integer(1));

        let sorted = prepare_table_for_snapshot(level1);

        // Check level 1
        let keys1: Vec<_> = sorted.keys().collect();
        assert_eq!(keys1, vec!["a1", "z1"]);

        // Check level 2
        if let Some(toml::Value::Table(l2)) = sorted.get("z1") {
            let keys2: Vec<_> = l2.keys().collect();
            assert_eq!(keys2, vec!["a2", "z2"]);

            // Check level 3
            if let Some(toml::Value::Table(l3)) = l2.get("z2") {
                let keys3: Vec<_> = l3.keys().collect();
                assert_eq!(keys3, vec!["a3", "z3"]);
            } else {
                panic!("Expected level 3 table");
            }
        } else {
            panic!("Expected level 2 table");
        }
    }

    #[test]
    fn test_prepare_table_for_snapshot_complex_structure() {
        // Simulate a realistic config structure
        let mut provider1 = Table::new();
        provider1.insert(
            "type".to_string(),
            toml::Value::String("openai".to_string()),
        );
        provider1.insert(
            "api_key".to_string(),
            toml::Value::String("secret".to_string()),
        );

        let mut provider2 = Table::new();
        provider2.insert(
            "url".to_string(),
            toml::Value::String("http://localhost".to_string()),
        );
        provider2.insert(
            "type".to_string(),
            toml::Value::String("custom".to_string()),
        );

        let mut config = Table::new();
        config.insert("zebra_provider".to_string(), toml::Value::Table(provider1));
        config.insert("alpha_provider".to_string(), toml::Value::Table(provider2));
        config.insert("timeout".to_string(), toml::Value::Integer(30));
        config.insert(
            "models".to_string(),
            toml::Value::Array(vec![
                toml::Value::String("gpt-4".to_string()),
                toml::Value::String("gpt-3.5".to_string()),
            ]),
        );

        let sorted = prepare_table_for_snapshot(config);

        // Check top-level sorting
        let keys: Vec<_> = sorted.keys().collect();
        assert_eq!(
            keys,
            vec!["alpha_provider", "models", "timeout", "zebra_provider"]
        );

        // Check nested table sorting
        if let Some(toml::Value::Table(alpha)) = sorted.get("alpha_provider") {
            let alpha_keys: Vec<_> = alpha.keys().collect();
            assert_eq!(alpha_keys, vec!["type", "url"]);
        } else {
            panic!("Expected alpha_provider table");
        }

        if let Some(toml::Value::Table(zebra)) = sorted.get("zebra_provider") {
            let zebra_keys: Vec<_> = zebra.keys().collect();
            assert_eq!(zebra_keys, vec!["api_key", "type"]);
        } else {
            panic!("Expected zebra_provider table");
        }
    }

    #[test]
    fn test_prepare_table_for_snapshot_deterministic() {
        // Test that the same input produces the same output (deterministic hashing)
        let mut table1 = Table::new();
        table1.insert("c".to_string(), toml::Value::Integer(3));
        table1.insert("a".to_string(), toml::Value::Integer(1));
        table1.insert("b".to_string(), toml::Value::Integer(2));

        let mut table2 = Table::new();
        table2.insert("b".to_string(), toml::Value::Integer(2));
        table2.insert("c".to_string(), toml::Value::Integer(3));
        table2.insert("a".to_string(), toml::Value::Integer(1));

        let sorted1 = prepare_table_for_snapshot(table1);
        let sorted2 = prepare_table_for_snapshot(table2);

        // Both should produce the same key order
        let keys1: Vec<_> = sorted1.keys().collect();
        let keys2: Vec<_> = sorted2.keys().collect();
        assert_eq!(keys1, keys2);
        assert_eq!(keys1, vec!["a", "b", "c"]);

        // Values should match
        assert_eq!(sorted1.get("a"), sorted2.get("a"));
        assert_eq!(sorted1.get("b"), sorted2.get("b"));
        assert_eq!(sorted1.get("c"), sorted2.get("c"));
    }
}
