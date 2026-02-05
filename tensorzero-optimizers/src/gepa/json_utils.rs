//! JSON utility functions for GEPA optimization.
//!
//! Provides helpers for deterministic JSON serialization.

use serde_json::{Map, Value};

/// Recursively sorts all object keys in a JSON value for deterministic serialization.
///
/// This is necessary because HashMap iteration order is non-deterministic,
/// which can cause cache misses when serializing the same data structure.
///
/// - Objects: Keys are sorted alphabetically, values are recursively processed
/// - Arrays: Elements are recursively processed (order preserved)
/// - Primitives: Returned unchanged
pub fn sort_json_keys(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted_map = Map::new();
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            for key in keys {
                if let Some(v) = map.get(&key) {
                    sorted_map.insert(key, sort_json_keys(v.clone()));
                }
            }
            Value::Object(sorted_map)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(sort_json_keys).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_sort_json_keys_flat_object() {
        let input = json!({
            "zebra": 1,
            "apple": 2,
            "mango": 3
        });

        let sorted = sort_json_keys(input);
        let keys: Vec<_> = sorted.as_object().unwrap().keys().collect();

        assert_eq!(
            keys,
            vec!["apple", "mango", "zebra"],
            "keys should be sorted alphabetically"
        );
    }

    #[test]
    fn test_sort_json_keys_nested_object() {
        let input = json!({
            "outer_z": {
                "inner_b": 1,
                "inner_a": 2
            },
            "outer_a": {
                "inner_z": 3,
                "inner_m": 4
            }
        });

        let sorted = sort_json_keys(input);

        // Check outer keys
        let outer_keys: Vec<_> = sorted.as_object().unwrap().keys().collect();
        assert_eq!(
            outer_keys,
            vec!["outer_a", "outer_z"],
            "outer keys should be sorted"
        );

        // Check inner keys of outer_a
        let outer_a = sorted.get("outer_a").unwrap().as_object().unwrap();
        let inner_a_keys: Vec<_> = outer_a.keys().collect();
        assert_eq!(
            inner_a_keys,
            vec!["inner_m", "inner_z"],
            "inner keys of outer_a should be sorted"
        );

        // Check inner keys of outer_z
        let outer_z = sorted.get("outer_z").unwrap().as_object().unwrap();
        let inner_z_keys: Vec<_> = outer_z.keys().collect();
        assert_eq!(
            inner_z_keys,
            vec!["inner_a", "inner_b"],
            "inner keys of outer_z should be sorted"
        );
    }

    #[test]
    fn test_sort_json_keys_array_with_objects() {
        let input = json!([
            {"z": 1, "a": 2},
            {"m": 3, "b": 4}
        ]);

        let sorted = sort_json_keys(input);
        let arr = sorted.as_array().unwrap();

        // Check first object keys
        let first_keys: Vec<_> = arr[0].as_object().unwrap().keys().collect();
        assert_eq!(
            first_keys,
            vec!["a", "z"],
            "first object keys should be sorted"
        );

        // Check second object keys
        let second_keys: Vec<_> = arr[1].as_object().unwrap().keys().collect();
        assert_eq!(
            second_keys,
            vec!["b", "m"],
            "second object keys should be sorted"
        );
    }

    #[test]
    fn test_sort_json_keys_primitives_unchanged() {
        assert_eq!(
            sort_json_keys(json!(42)),
            json!(42),
            "numbers should be unchanged"
        );
        assert_eq!(
            sort_json_keys(json!("hello")),
            json!("hello"),
            "strings should be unchanged"
        );
        assert_eq!(
            sort_json_keys(json!(true)),
            json!(true),
            "booleans should be unchanged"
        );
        assert_eq!(
            sort_json_keys(json!(null)),
            json!(null),
            "null should be unchanged"
        );
    }

    #[test]
    fn test_sort_json_keys_deeply_nested() {
        let input = json!({
            "level1_b": {
                "level2_z": {
                    "level3_m": "value",
                    "level3_a": "value"
                },
                "level2_a": "value"
            },
            "level1_a": "value"
        });

        let sorted = sort_json_keys(input);

        // Check all levels are sorted
        let l1_keys: Vec<_> = sorted.as_object().unwrap().keys().collect();
        assert_eq!(
            l1_keys,
            vec!["level1_a", "level1_b"],
            "level 1 keys should be sorted"
        );

        let l2 = sorted.get("level1_b").unwrap().as_object().unwrap();
        let l2_keys: Vec<_> = l2.keys().collect();
        assert_eq!(
            l2_keys,
            vec!["level2_a", "level2_z"],
            "level 2 keys should be sorted"
        );

        let l3 = l2.get("level2_z").unwrap().as_object().unwrap();
        let l3_keys: Vec<_> = l3.keys().collect();
        assert_eq!(
            l3_keys,
            vec!["level3_a", "level3_m"],
            "level 3 keys should be sorted"
        );
    }

    #[test]
    fn test_sort_json_keys_empty_structures() {
        assert_eq!(
            sort_json_keys(json!({})),
            json!({}),
            "empty object should remain empty"
        );
        assert_eq!(
            sort_json_keys(json!([])),
            json!([]),
            "empty array should remain empty"
        );
    }

    #[test]
    fn test_sort_json_keys_mixed_array() {
        let input = json!([
            {"z": 1, "a": 2},
            "string",
            42,
            null,
            [{"b": 1, "a": 2}]
        ]);

        let sorted = sort_json_keys(input);
        let arr = sorted.as_array().unwrap();

        // Object in array should have sorted keys
        let first_keys: Vec<_> = arr[0].as_object().unwrap().keys().collect();
        assert_eq!(
            first_keys,
            vec!["a", "z"],
            "object in array should have sorted keys"
        );

        // Primitives unchanged
        assert_eq!(arr[1], "string");
        assert_eq!(arr[2], 42);
        assert_eq!(arr[3], json!(null));

        // Nested array with object
        let nested_obj_keys: Vec<_> = arr[4].as_array().unwrap()[0]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        assert_eq!(
            nested_obj_keys,
            vec!["a", "b"],
            "nested object in array should have sorted keys"
        );
    }
}
