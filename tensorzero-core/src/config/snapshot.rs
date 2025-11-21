use std::collections::HashMap;

#[derive(Debug)]
pub struct ConfigSnapshot {
    pub config: String, // serialized as TOML
    pub extra_templates: HashMap<String, String>,
}

impl ConfigSnapshot {
    /// Compute a blake3 hash of this config snapshot
    pub fn hash(&self) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        let ConfigSnapshot {
            config,
            extra_templates,
        } = self;

        // Hash the config string
        hasher.update(config.as_bytes());
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

        hasher.finalize()
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
