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
//! - `$ref` cannot have sibling keywords (`type`, `properties`, `required`, `additionalProperties`)
//!
//! ## Supported Keywords (preserved)
//!
//! - `pattern`, `format` (date-time, email, uuid, etc.)
//! - `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`
//! - `minItems`, `maxItems`
//! - `enum`, `const`
//! - `$ref`, `$defs`/`definitions`
//! - `anyOf`
//!
//! ## Unsupported Metadata Keywords (removed)
//!
//! - `$schema`, `title`, `default`, `deprecated`
//! - Custom extensions (`x-*`)

use serde_json::{Map, Value};

/// Transform a JSON schema to be compatible with OpenAI Structured Outputs.
///
/// This function modifies the schema in place, applying all necessary transformations:
/// 1. Resolves `$ref` with sibling keywords by merging the referenced definition
/// 2. Removes unsupported composition keywords (`allOf`, `oneOf` → `anyOf`, `not`, etc.)
/// 3. Removes unsupported metadata keywords (`$schema`, `title`, `default`, `deprecated`, `x-*`)
/// 4. Sets `additionalProperties: false` on all objects
/// 5. Ensures all properties are listed in `required`
pub fn to_openai_compatible(schema: &mut Value) {
    // Extract $defs for reference resolution (need access during transformation)
    let defs = schema
        .get("$defs")
        .cloned()
        .or_else(|| schema.get("definitions").cloned());

    resolve_refs_with_siblings(schema, defs.as_ref());
    remove_unsupported_keywords(schema);
    set_additional_properties_false(schema);
    enforce_all_required(schema);
}

/// Resolve `$ref` that have sibling keywords by merging the referenced definition.
///
/// OpenAI's Structured Outputs does not support `$ref` with sibling keywords like
/// `type`, `properties`, `required`, or `additionalProperties`. When we encounter
/// such cases, we resolve the reference and merge the properties inline.
fn resolve_refs_with_siblings(value: &mut Value, defs: Option<&Value>) {
    match value {
        Value::Object(obj) => {
            // Check if this object has $ref with siblings that OpenAI doesn't support
            if let Some(Value::String(ref_path)) = obj.get("$ref").cloned() {
                let has_problematic_siblings = obj.keys().any(|k| {
                    matches!(
                        k.as_str(),
                        "type" | "properties" | "required" | "additionalProperties"
                    )
                });

                if has_problematic_siblings {
                    // Resolve the ref and merge
                    if let Some(resolved) = resolve_ref(&ref_path, defs) {
                        merge_schema(obj, resolved);
                    }
                    obj.remove("$ref");
                }
            }

            // Recurse into all values
            for v in obj.values_mut() {
                resolve_refs_with_siblings(v, defs);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                resolve_refs_with_siblings(v, defs);
            }
        }
        _ => {}
    }
}

/// Resolve a `$ref` path like `#/$defs/ChatRequest` or `#/definitions/ChatRequest`.
fn resolve_ref(ref_path: &str, defs: Option<&Value>) -> Option<Value> {
    let defs = defs?;

    // Parse paths like "#/$defs/Name" or "#/definitions/Name"
    let path = ref_path.strip_prefix('#')?;

    // Handle "/$defs/Name" or "/definitions/Name"
    let name = path
        .strip_prefix("/$defs/")
        .or_else(|| path.strip_prefix("/definitions/"))?;

    defs.get(name).cloned()
}

/// Merge a resolved schema into the target object.
///
/// Properties from the source are added to the target (target properties take precedence).
/// Required arrays are merged (deduplicated).
fn merge_schema(target: &mut Map<String, Value>, source: Value) {
    let Value::Object(source_obj) = source else {
        return;
    };

    // Merge properties
    if let Some(Value::Object(source_props)) = source_obj.get("properties") {
        let target_props = target
            .entry("properties")
            .or_insert_with(|| Value::Object(Map::new()));

        if let Value::Object(target_props_obj) = target_props {
            for (key, value) in source_props {
                // Only add if not already present (target takes precedence)
                target_props_obj.entry(key.clone()).or_insert(value.clone());
            }
        }
    }

    // Merge required arrays
    if let Some(Value::Array(source_required)) = source_obj.get("required") {
        let target_required = target
            .entry("required")
            .or_insert_with(|| Value::Array(Vec::new()));

        if let Value::Array(target_required_arr) = target_required {
            for item in source_required {
                if !target_required_arr.contains(item) {
                    target_required_arr.push(item.clone());
                }
            }
        }
    }

    // Copy other fields from source if not present in target
    // (except $ref-specific fields we don't want to carry over)
    for (key, value) in source_obj {
        if !matches!(key.as_str(), "properties" | "required" | "$ref") {
            target.entry(key).or_insert(value);
        }
    }
}

/// Remove unsupported JSON Schema keywords and convert others to supported equivalents.
///
/// - Converts `allOf` and `oneOf` to `anyOf` (best-effort, may lose semantics)
/// - Removes `not`, `if`, `then`, `else`, `dependentRequired`, `dependentSchemas`
/// - Removes metadata keywords: `$schema`, `title`, `default`, `deprecated`
/// - Removes custom extensions (`x-*` keys)
fn remove_unsupported_keywords(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            // Convert allOf/oneOf to anyOf
            for key in ["allOf", "oneOf"] {
                if let Some(v) = obj.remove(key) {
                    obj.insert("anyOf".to_string(), v);
                }
            }

            // Remove unsupported composition keywords and metadata
            for key in [
                // Composition keywords
                "not",
                "if",
                "then",
                "else",
                "dependentRequired",
                "dependentSchemas",
                // Metadata keywords not supported by OpenAI
                "$schema",
                "title",
                "default",
                "deprecated",
            ] {
                obj.remove(key);
            }

            // Remove custom extensions (x-* keys)
            obj.retain(|k, _| !k.starts_with("x-"));

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

    #[test]
    fn test_removes_metadata_keywords() {
        let mut schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "TestSchema",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Name Field",
                    "default": "unnamed",
                    "deprecated": true
                }
            },
            "default": {}
        });

        to_openai_compatible(&mut schema);

        assert!(schema.get("$schema").is_none());
        assert!(schema.get("title").is_none());
        assert!(schema.get("default").is_none());
        assert!(schema["properties"]["name"].get("title").is_none());
        assert!(schema["properties"]["name"].get("default").is_none());
        assert!(schema["properties"]["name"].get("deprecated").is_none());
    }

    #[test]
    fn test_removes_custom_extensions() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "x-double-option": true,
                    "x-custom-field": "test"
                }
            },
            "x-some-extension": "value"
        });

        to_openai_compatible(&mut schema);

        assert!(schema.get("x-some-extension").is_none());
        assert!(
            schema["properties"]["value"]
                .get("x-double-option")
                .is_none()
        );
        assert!(
            schema["properties"]["value"]
                .get("x-custom-field")
                .is_none()
        );
    }

    #[test]
    fn test_resolves_ref_with_siblings() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "request": {
                    "$ref": "#/$defs/Request"
                }
            },
            "$defs": {
                "Request": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": { "type": "string", "const": "chat" }
                            },
                            "$ref": "#/$defs/ChatRequest",
                            "required": ["type"],
                            "additionalProperties": false
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": { "type": "string", "const": "json" }
                            },
                            "$ref": "#/$defs/JsonRequest",
                            "required": ["type"],
                            "additionalProperties": false
                        }
                    ]
                },
                "ChatRequest": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" }
                    },
                    "required": ["id", "name"],
                    "additionalProperties": false
                },
                "JsonRequest": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "value": { "type": "number" }
                    },
                    "required": ["id", "value"],
                    "additionalProperties": false
                }
            }
        });

        to_openai_compatible(&mut schema);

        // The $ref should be removed and properties merged
        let chat_branch = &schema["$defs"]["Request"]["anyOf"][0];
        assert!(chat_branch.get("$ref").is_none());

        // Should have both the inline property and merged properties
        assert!(chat_branch["properties"].get("type").is_some());
        assert!(chat_branch["properties"].get("id").is_some());
        assert!(chat_branch["properties"].get("name").is_some());

        // Required should be merged
        let required = chat_branch["required"].as_array().unwrap();
        assert!(required.contains(&json!("type")));
        assert!(required.contains(&json!("id")));
        assert!(required.contains(&json!("name")));

        // Same for json branch
        let json_branch = &schema["$defs"]["Request"]["anyOf"][1];
        assert!(json_branch.get("$ref").is_none());
        assert!(json_branch["properties"].get("type").is_some());
        assert!(json_branch["properties"].get("id").is_some());
        assert!(json_branch["properties"].get("value").is_some());
    }

    #[test]
    fn test_ref_without_siblings_preserved() {
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

        // $ref without problematic siblings should be preserved
        assert!(schema["properties"]["user"].get("$ref").is_some());
    }

    #[test]
    fn test_ref_with_definitions_key() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "tag": { "type": "string", "const": "test" }
                    },
                    "$ref": "#/definitions/DataFields",
                    "required": ["tag"]
                }
            },
            "definitions": {
                "DataFields": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "value": { "type": "number" }
                    },
                    "required": ["id", "value"]
                }
            }
        });

        to_openai_compatible(&mut schema);

        // Should work with "definitions" as well as "$defs"
        let data = &schema["properties"]["data"];
        assert!(data.get("$ref").is_none());
        assert!(data["properties"].get("tag").is_some());
        assert!(data["properties"].get("id").is_some());
        assert!(data["properties"].get("value").is_some());
    }

    #[test]
    fn test_inline_properties_take_precedence() {
        // When both inline and referenced schema have the same property,
        // the inline property should take precedence
        let mut schema = json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Inline ID" }
                    },
                    "$ref": "#/$defs/DataFields",
                    "required": ["id"]
                }
            },
            "$defs": {
                "DataFields": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer", "description": "Ref ID" },
                        "value": { "type": "number" }
                    },
                    "required": ["id", "value"]
                }
            }
        });

        to_openai_compatible(&mut schema);

        let data = &schema["properties"]["data"];
        // Inline property should win - type should be "string", not "integer"
        assert_eq!(data["properties"]["id"]["type"], json!("string"));
        assert_eq!(data["properties"]["id"]["description"], json!("Inline ID"));
        // But the additional property from ref should be merged
        assert!(data["properties"].get("value").is_some());
    }

    #[test]
    fn test_unresolvable_ref_is_removed() {
        // When $ref points to a non-existent definition, it should still be
        // removed if it has problematic siblings (to avoid OpenAI errors)
        let mut schema = json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" }
                    },
                    "$ref": "#/$defs/NonExistent",
                    "required": ["id"]
                }
            },
            "$defs": {}
        });

        to_openai_compatible(&mut schema);

        let data = &schema["properties"]["data"];
        // $ref should be removed even though it couldn't be resolved
        assert!(data.get("$ref").is_none());
        // Original properties should remain
        assert!(data["properties"].get("id").is_some());
    }

    #[test]
    fn test_ref_with_description_only_preserved() {
        // $ref with only non-problematic siblings like "description" should be preserved
        let mut schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "$ref": "#/$defs/User",
                    "description": "The user object"
                }
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

        // $ref should be preserved since "description" is not a problematic sibling
        assert!(schema["properties"]["user"].get("$ref").is_some());
        assert_eq!(
            schema["properties"]["user"]["description"],
            json!("The user object")
        );
    }

    #[test]
    fn test_description_merged_from_ref() {
        // When resolving $ref, description from the referenced schema should be merged
        let mut schema = json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "tag": { "type": "string" }
                    },
                    "$ref": "#/$defs/DataFields",
                    "required": ["tag"]
                }
            },
            "$defs": {
                "DataFields": {
                    "type": "object",
                    "description": "Data fields from the definition",
                    "properties": {
                        "id": { "type": "string" }
                    },
                    "required": ["id"]
                }
            }
        });

        to_openai_compatible(&mut schema);

        let data = &schema["properties"]["data"];
        // Description from the ref should be merged
        assert_eq!(
            data["description"],
            json!("Data fields from the definition")
        );
    }
}
