//! Transform JSON schemas to be compatible with OpenAI Structured Outputs.
//!
//! OpenAI's Structured Outputs feature requires schemas to conform to a specific subset
//! of JSON Schema. This module provides functions to transform arbitrary schemas into
//! compliant ones.
//!
//! ## OpenAI Requirements
//!
//! - `additionalProperties: false` must be set on all objects
//! - All properties must be listed in `required`
//! - Unsupported keywords must be removed: `allOf`, `oneOf`, `not`, `if`, `then`, `else`,
//!   `dependentRequired`, `dependentSchemas`
//!
//! ## Supported Keywords (preserved)
//!
//! - `pattern`, `format` (date-time, email, uuid, etc.)
//! - `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`
//! - `minItems`, `maxItems`
//! - `enum`, `const`
//! - `$ref`, `$defs`/`definitions`
//! - `anyOf`

use serde_json::Value;

/// Transform a JSON schema to be compatible with OpenAI Structured Outputs.
///
/// This function modifies the schema in place, applying all necessary transformations:
/// 1. Removes unsupported composition keywords (`allOf`, `oneOf` → `anyOf`, `not`, etc.)
/// 2. Sets `additionalProperties: false` on all objects
/// 3. Ensures all properties are listed in `required`
pub fn to_openai_compatible(schema: &mut Value) {
    remove_unsupported_keywords(schema);
    set_additional_properties_false(schema);
    enforce_all_required(schema);
}

/// Remove unsupported JSON Schema keywords and convert others to supported equivalents.
///
/// - Converts `allOf` and `oneOf` to `anyOf` (best-effort, may lose semantics)
/// - Removes `not`, `if`, `then`, `else`, `dependentRequired`, `dependentSchemas`
fn remove_unsupported_keywords(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            // Convert allOf/oneOf to anyOf
            for key in ["allOf", "oneOf"] {
                if let Some(v) = obj.remove(key) {
                    obj.insert("anyOf".to_string(), v);
                }
            }

            // Remove unsupported composition keywords
            for key in [
                "not",
                "if",
                "then",
                "else",
                "dependentRequired",
                "dependentSchemas",
            ] {
                obj.remove(key);
            }

            // Recurse into all values
            for v in obj.values_mut() {
                remove_unsupported_keywords(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                remove_unsupported_keywords(v);
            }
        }
        _ => {}
    }
}

/// Set `additionalProperties: false` on all objects with `type: "object"`.
///
/// This is required by OpenAI Structured Outputs to ensure the model only generates
/// specified keys.
fn set_additional_properties_false(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            // Check if this is an object type schema
            let is_object_type = obj
                .get("type")
                .is_some_and(|t| t == &Value::String("object".to_string()));

            if is_object_type {
                obj.insert("additionalProperties".to_string(), Value::Bool(false));
            }

            // Recurse into all values
            for v in obj.values_mut() {
                set_additional_properties_false(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                set_additional_properties_false(v);
            }
        }
        _ => {}
    }
}

/// Ensure all properties are listed in the `required` array.
///
/// OpenAI Structured Outputs requires all fields to be required. Optional fields
/// should use `type: ["string", "null"]` or `anyOf` with null instead.
fn enforce_all_required(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            // If this object has properties, ensure all are required
            if let Some(Value::Object(properties)) = obj.get("properties") {
                let property_names: Vec<Value> = properties
                    .keys()
                    .map(|k| Value::String(k.clone()))
                    .collect();

                if !property_names.is_empty() {
                    // Get or create the required array
                    let required = obj
                        .entry("required")
                        .or_insert_with(|| Value::Array(Vec::new()));

                    if let Value::Array(required_arr) = required {
                        for name in property_names {
                            if !required_arr.contains(&name) {
                                required_arr.push(name);
                            }
                        }
                    }
                }
            }

            // Recurse into all values
            for v in obj.values_mut() {
                enforce_all_required(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                enforce_all_required(v);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_adds_additional_properties_false() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        to_openai_compatible(&mut schema);

        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn test_nested_objects_get_additional_properties() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            }
        });

        to_openai_compatible(&mut schema);

        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(
            schema["properties"]["user"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn test_enforces_all_required() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name"]
        });

        to_openai_compatible(&mut schema);

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("name")));
        assert!(required.contains(&json!("age")));
    }

    #[test]
    fn test_creates_required_if_missing() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        to_openai_compatible(&mut schema);

        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("name")));
    }

    #[test]
    fn test_converts_allof_to_anyof() {
        let mut schema = json!({
            "allOf": [
                { "type": "object", "properties": { "a": { "type": "string" } } },
                { "type": "object", "properties": { "b": { "type": "string" } } }
            ]
        });

        to_openai_compatible(&mut schema);

        assert!(schema.get("allOf").is_none());
        assert!(schema.get("anyOf").is_some());
    }

    #[test]
    fn test_converts_oneof_to_anyof() {
        let mut schema = json!({
            "oneOf": [
                { "type": "string" },
                { "type": "number" }
            ]
        });

        to_openai_compatible(&mut schema);

        assert!(schema.get("oneOf").is_none());
        assert!(schema.get("anyOf").is_some());
    }

    #[test]
    fn test_removes_unsupported_keywords() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "if": { "properties": { "name": { "const": "test" } } },
            "then": { "required": ["extra"] },
            "else": {},
            "not": { "type": "null" },
            "dependentRequired": { "name": ["age"] },
            "dependentSchemas": {}
        });

        to_openai_compatible(&mut schema);

        assert!(schema.get("if").is_none());
        assert!(schema.get("then").is_none());
        assert!(schema.get("else").is_none());
        assert!(schema.get("not").is_none());
        assert!(schema.get("dependentRequired").is_none());
        assert!(schema.get("dependentSchemas").is_none());
    }

    #[test]
    fn test_preserves_supported_keywords() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "pattern": "^[a-z]+@[a-z]+\\.[a-z]+$"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "minItems": 1,
                    "maxItems": 10
                }
            }
        });

        to_openai_compatible(&mut schema);

        // Supported keywords should be preserved
        assert_eq!(schema["properties"]["email"]["format"], json!("email"));
        assert!(schema["properties"]["email"]["pattern"].as_str().is_some());
        assert_eq!(schema["properties"]["age"]["minimum"], json!(0));
        assert_eq!(schema["properties"]["age"]["maximum"], json!(150));
        assert_eq!(schema["properties"]["tags"]["minItems"], json!(1));
        assert_eq!(schema["properties"]["tags"]["maxItems"], json!(10));
    }

    #[test]
    fn test_handles_defs() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "user": { "$ref": "#/$defs/User" }
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            }
        });

        to_openai_compatible(&mut schema);

        // $defs should also be transformed
        assert_eq!(
            schema["$defs"]["User"]["additionalProperties"],
            json!(false)
        );
        let user_required = schema["$defs"]["User"]["required"].as_array().unwrap();
        assert!(user_required.contains(&json!("name")));
    }

    #[test]
    fn test_handles_anyof_branches() {
        let mut schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                },
                {
                    "type": "null"
                }
            ]
        });

        to_openai_compatible(&mut schema);

        // Each branch should be transformed
        assert_eq!(schema["anyOf"][0]["additionalProperties"], json!(false));
        let required = schema["anyOf"][0]["required"].as_array().unwrap();
        assert!(required.contains(&json!("name")));
    }

    #[test]
    fn test_handles_array_items() {
        let mut schema = json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": { "type": "integer" }
                }
            }
        });

        to_openai_compatible(&mut schema);

        assert_eq!(schema["items"]["additionalProperties"], json!(false));
        let required = schema["items"]["required"].as_array().unwrap();
        assert!(required.contains(&json!("id")));
    }
}
