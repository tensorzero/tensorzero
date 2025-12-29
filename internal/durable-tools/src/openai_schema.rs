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
    normalize_any_type_schemas(schema);
    remove_unsupported_formats(schema);
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
///
/// Special handling for `anyOf`: When the source contains `anyOf`, we can't just copy it
/// to the target because having both `properties` AND `anyOf` at the same level is invalid
/// for OpenAI Structured Outputs. Instead, we expand each `anyOf` branch to include the
/// target's inline properties, then replace the target with just the expanded `anyOf`.
fn merge_schema(target: &mut Map<String, Value>, source: Value) {
    let Value::Object(source_obj) = source else {
        return;
    };

    // Special case: if source has `anyOf`, we need to expand each branch with the target's
    // inline properties, then replace the target entirely with the expanded `anyOf`.
    // This is because OpenAI doesn't allow `properties` and `anyOf` at the same level.
    if let Some(Value::Array(any_of_branches)) = source_obj.get("anyOf") {
        let expanded_branches = expand_anyof_branches(target, any_of_branches);

        // Clear the target and replace with just the expanded anyOf
        // Keep only description if present (it's valid metadata for anyOf schemas)
        let description = target.remove("description");
        target.clear();
        if let Some(desc) = description {
            target.insert("description".to_string(), desc);
        }
        target.insert("anyOf".to_string(), Value::Array(expanded_branches));
        return;
    }

    // Standard merge for non-anyOf sources
    merge_properties(target, &source_obj);
    merge_required(target, &source_obj);

    // Copy other fields from source if not present in target
    // (except $ref-specific fields we don't want to carry over)
    for (key, value) in source_obj {
        if !matches!(key.as_str(), "properties" | "required" | "$ref") {
            target.entry(key).or_insert(value);
        }
    }
}

/// Expand each branch of an `anyOf` with the target's inline properties.
///
/// For each branch, we merge in the target's `type`, `properties`, `required`,
/// and `additionalProperties` to create a complete schema.
/// Inline properties (from target) take precedence over branch properties.
fn expand_anyof_branches(target: &Map<String, Value>, branches: &[Value]) -> Vec<Value> {
    branches
        .iter()
        .map(|branch| {
            let mut expanded = if let Value::Object(obj) = branch {
                obj.clone()
            } else {
                // Non-object branch (e.g., {"type": "null"}), just clone it
                return branch.clone();
            };

            // Merge properties: inline properties (from target) take precedence over branch
            if let Some(Value::Object(target_props)) = target.get("properties") {
                let branch_props = expanded
                    .entry("properties")
                    .or_insert_with(|| Value::Object(Map::new()));

                if let Value::Object(branch_props_obj) = branch_props {
                    // Overwrite branch properties with target properties (target takes precedence)
                    for (key, value) in target_props {
                        branch_props_obj.insert(key.clone(), value.clone());
                    }
                }
            }

            // Merge target's required into this branch
            if let Some(target_required) = target.get("required") {
                merge_required(
                    &mut expanded,
                    &map_with_key("required", target_required.clone()),
                );
            }

            // Copy target's type if branch doesn't have one
            if let Some(target_type) = target.get("type") {
                expanded.entry("type").or_insert(target_type.clone());
            }

            // Copy target's additionalProperties if branch doesn't have one
            if let Some(target_additional) = target.get("additionalProperties") {
                expanded
                    .entry("additionalProperties")
                    .or_insert(target_additional.clone());
            }

            Value::Object(expanded)
        })
        .collect()
}

/// Helper to create a Map with a single key-value pair.
fn map_with_key(key: &str, value: Value) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert(key.to_string(), value);
    map
}

/// Merge properties from source into target (target properties take precedence).
fn merge_properties(target: &mut Map<String, Value>, source: &Map<String, Value>) {
    if let Some(Value::Object(source_props)) = source.get("properties") {
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
}

/// Merge required arrays from source into target (deduplicated).
fn merge_required(target: &mut Map<String, Value>, source: &Map<String, Value>) {
    if let Some(Value::Array(source_required)) = source.get("required") {
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

/// Normalize schemas that accept "any" type (represented as `true` or missing type).
///
/// OpenAI Structured Outputs requires explicit types. When we encounter:
/// - `true` (JSON Schema shorthand for "accept anything")
/// - A schema object without a `type` field (and no `$ref`, `anyOf`, etc.)
///
/// We convert these to an empty object type `{"type": "object", "additionalProperties": true}`,
/// which OpenAI interprets as an arbitrary JSON object.
///
/// Note: We keep `additionalProperties: true` here because `set_additional_properties_false`
/// will set it to `false` later, but only for schemas that are actual object types with properties.
fn normalize_any_type_schemas(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            // Check each property value for `true` schema or missing type
            if let Some(Value::Object(properties)) = obj.get_mut("properties") {
                for prop_value in properties.values_mut() {
                    // Handle `true` schema (accept anything)
                    if *prop_value == Value::Bool(true) {
                        *prop_value = serde_json::json!({});
                        continue;
                    }

                    // Handle schema objects without type
                    if let Value::Object(prop_obj) = prop_value {
                        let has_type = prop_obj.contains_key("type");
                        let has_ref = prop_obj.contains_key("$ref");
                        let has_any_of = prop_obj.contains_key("anyOf");
                        let has_one_of = prop_obj.contains_key("oneOf");
                        let has_all_of = prop_obj.contains_key("allOf");
                        let has_enum = prop_obj.contains_key("enum");
                        let has_const = prop_obj.contains_key("const");

                        // If no type-defining keyword is present, this is effectively "any"
                        if !has_type
                            && !has_ref
                            && !has_any_of
                            && !has_one_of
                            && !has_all_of
                            && !has_enum
                            && !has_const
                        {
                            // Convert to empty schema (matches anything)
                            let description = prop_obj.remove("description");
                            prop_obj.clear();
                            if let Some(desc) = description {
                                prop_obj.insert("description".to_string(), desc);
                            }
                        }
                    }
                }
            }

            // Handle items in arrays
            if let Some(items) = obj.get_mut("items") {
                if *items == Value::Bool(true) {
                    *items = serde_json::json!({});
                }
            }

            // Recurse into all values
            for v in obj.values_mut() {
                normalize_any_type_schemas(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                normalize_any_type_schemas(v);
            }
        }
        _ => {}
    }
}

/// Supported format values per OpenAI documentation:
/// - date-time, time, date, duration
/// - email, hostname
/// - ipv4, ipv6, uuid
const SUPPORTED_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
];

/// Remove unsupported `format` values from string properties.
///
/// OpenAI only supports a specific set of format values. Other formats like
/// `float`, `int32`, `uint32` (commonly used in OpenAPI/JSON Schema for numbers)
/// are not supported and must be removed.
fn remove_unsupported_formats(value: &mut Value) {
    match value {
        Value::Object(obj) => {
            // Check if this object has an unsupported format
            if let Some(Value::String(format)) = obj.get("format") {
                if !SUPPORTED_FORMATS.contains(&format.as_str()) {
                    obj.remove("format");
                }
            }

            // Recurse into all values
            for v in obj.values_mut() {
                remove_unsupported_formats(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                remove_unsupported_formats(v);
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

    #[test]
    fn test_ref_to_anyof_expands_branches() {
        // When $ref points to an anyOf schema, we need to expand each branch
        // with the inline properties instead of creating invalid schema with
        // both properties and anyOf at the same level
        let mut schema = json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "object",
                    "properties": {
                        "type": { "type": "string", "const": "tool_call" }
                    },
                    "$ref": "#/$defs/ToolCallWrapper",
                    "required": ["type"],
                    "additionalProperties": false
                }
            },
            "$defs": {
                "ToolCallWrapper": {
                    "anyOf": [
                        { "$ref": "#/$defs/ToolCall" },
                        { "$ref": "#/$defs/InferenceResponseToolCall" }
                    ]
                },
                "ToolCall": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" }
                    },
                    "required": ["id", "name"],
                    "additionalProperties": false
                },
                "InferenceResponseToolCall": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "raw_name": { "type": "string" }
                    },
                    "required": ["id", "raw_name"],
                    "additionalProperties": false
                }
            }
        });

        to_openai_compatible(&mut schema);

        let content = &schema["properties"]["content"];

        // The result should be an anyOf with expanded branches, NOT properties + anyOf
        assert!(
            content.get("properties").is_none(),
            "Should not have both properties and anyOf at same level"
        );
        assert!(content.get("anyOf").is_some(), "Should have anyOf");

        let any_of = content["anyOf"].as_array().unwrap();
        assert_eq!(any_of.len(), 2);

        // Each branch should have the inline "type" property merged in
        for branch in any_of {
            assert!(
                branch["properties"].get("type").is_some(),
                "Each branch should have the 'type' property"
            );
            // Each branch should have additionalProperties: false
            assert_eq!(
                branch["additionalProperties"],
                json!(false),
                "Each branch should have additionalProperties: false"
            );
        }

        // First branch should have ToolCall properties + inline type
        assert!(any_of[0]["properties"].get("id").is_some());
        assert!(any_of[0]["properties"].get("name").is_some());
        assert!(any_of[0]["properties"].get("type").is_some());

        // Second branch should have InferenceResponseToolCall properties + inline type
        assert!(any_of[1]["properties"].get("id").is_some());
        assert!(any_of[1]["properties"].get("raw_name").is_some());
        assert!(any_of[1]["properties"].get("type").is_some());
    }

    #[test]
    fn test_ref_to_anyof_merges_required() {
        // When expanding anyOf branches, required arrays should be merged
        let mut schema = json!({
            "type": "object",
            "properties": {
                "item": {
                    "type": "object",
                    "properties": {
                        "tag": { "type": "string", "const": "special" }
                    },
                    "$ref": "#/$defs/TypeUnion",
                    "required": ["tag"],
                    "additionalProperties": false
                }
            },
            "$defs": {
                "TypeUnion": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "a": { "type": "string" }
                            },
                            "required": ["a"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "b": { "type": "number" }
                            },
                            "required": ["b"]
                        }
                    ]
                }
            }
        });

        to_openai_compatible(&mut schema);

        let item = &schema["properties"]["item"];
        let any_of = item["anyOf"].as_array().unwrap();

        // First branch should require both "tag" and "a"
        let required_0 = any_of[0]["required"].as_array().unwrap();
        assert!(required_0.contains(&json!("tag")));
        assert!(required_0.contains(&json!("a")));

        // Second branch should require both "tag" and "b"
        let required_1 = any_of[1]["required"].as_array().unwrap();
        assert!(required_1.contains(&json!("tag")));
        assert!(required_1.contains(&json!("b")));
    }

    #[test]
    fn test_ref_to_anyof_preserves_description() {
        // Description on the inline schema should be preserved
        let mut schema = json!({
            "type": "object",
            "properties": {
                "choice": {
                    "description": "A choice between options",
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string" }
                    },
                    "$ref": "#/$defs/Options",
                    "required": ["kind"]
                }
            },
            "$defs": {
                "Options": {
                    "anyOf": [
                        { "type": "object", "properties": { "x": { "type": "number" } } },
                        { "type": "object", "properties": { "y": { "type": "number" } } }
                    ]
                }
            }
        });

        to_openai_compatible(&mut schema);

        let choice = &schema["properties"]["choice"];
        // Description should be preserved on the anyOf schema
        assert_eq!(choice["description"], json!("A choice between options"));
        assert!(choice.get("anyOf").is_some());
    }

    #[test]
    fn test_ref_to_anyof_inline_properties_take_precedence() {
        // When both inline and branch have same property, inline should win
        let mut schema = json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Inline ID" }
                    },
                    "$ref": "#/$defs/TypedData",
                    "required": ["id"],
                    "additionalProperties": false
                }
            },
            "$defs": {
                "TypedData": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "id": { "type": "integer", "description": "Branch ID" },
                                "value": { "type": "string" }
                            }
                        }
                    ]
                }
            }
        });

        to_openai_compatible(&mut schema);

        let data = &schema["properties"]["data"];
        let branch = &data["anyOf"][0];

        // Inline property should take precedence
        assert_eq!(branch["properties"]["id"]["type"], json!("string"));
        assert_eq!(
            branch["properties"]["id"]["description"],
            json!("Inline ID")
        );
        // But branch's unique property should be merged
        assert!(branch["properties"].get("value").is_some());
    }

    #[test]
    fn test_nested_ref_to_anyof() {
        // Test deeply nested $ref to anyOf - this is the real-world ToolCallWrapper case
        let mut schema = json!({
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$defs/Message"
                    }
                }
            },
            "$defs": {
                "Message": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "array",
                            "items": {
                                "$ref": "#/$defs/ContentBlock"
                            }
                        }
                    }
                },
                "ContentBlock": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": { "const": "text" },
                                "text": { "type": "string" }
                            },
                            "required": ["type", "text"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": { "const": "tool_call" }
                            },
                            "$ref": "#/$defs/ToolCallData",
                            "required": ["type"],
                            "additionalProperties": false
                        }
                    ]
                },
                "ToolCallData": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "name": { "type": "string" }
                            },
                            "required": ["id", "name"]
                        }
                    ]
                }
            }
        });

        to_openai_compatible(&mut schema);

        // Navigate to the tool_call content block
        let content_block = &schema["$defs"]["ContentBlock"];
        let tool_call_branch = &content_block["anyOf"][1];

        // The tool_call branch should now be an anyOf (expanded from ToolCallData)
        assert!(
            tool_call_branch.get("anyOf").is_some(),
            "tool_call branch should be expanded to anyOf"
        );
        assert!(
            tool_call_branch.get("properties").is_none(),
            "tool_call branch should not have properties at top level"
        );

        // The nested anyOf branch should have both inline "type" and ToolCallData properties
        let nested_branch = &tool_call_branch["anyOf"][0];
        assert!(nested_branch["properties"].get("type").is_some());
        assert!(nested_branch["properties"].get("id").is_some());
        assert!(nested_branch["properties"].get("name").is_some());
    }

    #[test]
    fn test_normalizes_true_schema_to_empty_object() {
        // Properties with `true` as schema (accept any value) should be normalized
        let mut schema = json!({
            "type": "object",
            "properties": {
                "parameters": true,
                "data": true
            }
        });

        to_openai_compatible(&mut schema);

        // `true` should be converted to an empty schema `{}`
        assert_eq!(schema["properties"]["parameters"], json!({}));
        assert_eq!(schema["properties"]["data"], json!({}));
    }

    #[test]
    fn test_normalizes_schema_missing_type() {
        // Properties with only description (no type) should be normalized
        let mut schema = json!({
            "type": "object",
            "properties": {
                "data": {
                    "description": "Some arbitrary data"
                },
                "output": {
                    "description": "Output value that can be anything"
                }
            }
        });

        to_openai_compatible(&mut schema);

        // Should be converted to just description (empty schema)
        assert_eq!(
            schema["properties"]["data"],
            json!({ "description": "Some arbitrary data" })
        );
        assert_eq!(
            schema["properties"]["output"],
            json!({ "description": "Output value that can be anything" })
        );
    }

    #[test]
    fn test_preserves_schema_with_ref() {
        // Properties with $ref should not be modified
        let mut schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "$ref": "#/$defs/User",
                    "description": "A user reference"
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

        // $ref should be preserved
        assert!(schema["properties"]["user"].get("$ref").is_some());
    }

    #[test]
    fn test_preserves_schema_with_anyof() {
        // Properties with anyOf should not be modified
        let mut schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ],
                    "description": "A nullable string"
                }
            }
        });

        to_openai_compatible(&mut schema);

        // anyOf should be preserved
        assert!(schema["properties"]["value"].get("anyOf").is_some());
        assert_eq!(
            schema["properties"]["value"]["description"],
            json!("A nullable string")
        );
    }

    #[test]
    fn test_removes_unsupported_format_float() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "format": "float"
                },
                "count": {
                    "type": "integer",
                    "format": "int32"
                },
                "size": {
                    "type": "integer",
                    "format": "uint32"
                }
            }
        });

        to_openai_compatible(&mut schema);

        // Unsupported formats should be removed
        assert!(schema["properties"]["temperature"].get("format").is_none());
        assert!(schema["properties"]["count"].get("format").is_none());
        assert!(schema["properties"]["size"].get("format").is_none());
    }

    #[test]
    fn test_preserves_supported_formats() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "format": "uuid"
                },
                "email": {
                    "type": "string",
                    "format": "email"
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time"
                },
                "date": {
                    "type": "string",
                    "format": "date"
                },
                "time": {
                    "type": "string",
                    "format": "time"
                },
                "duration": {
                    "type": "string",
                    "format": "duration"
                },
                "hostname": {
                    "type": "string",
                    "format": "hostname"
                },
                "ipv4": {
                    "type": "string",
                    "format": "ipv4"
                },
                "ipv6": {
                    "type": "string",
                    "format": "ipv6"
                }
            }
        });

        to_openai_compatible(&mut schema);

        // All supported formats should be preserved
        assert_eq!(schema["properties"]["id"]["format"], json!("uuid"));
        assert_eq!(schema["properties"]["email"]["format"], json!("email"));
        assert_eq!(
            schema["properties"]["created_at"]["format"],
            json!("date-time")
        );
        assert_eq!(schema["properties"]["date"]["format"], json!("date"));
        assert_eq!(schema["properties"]["time"]["format"], json!("time"));
        assert_eq!(
            schema["properties"]["duration"]["format"],
            json!("duration")
        );
        assert_eq!(
            schema["properties"]["hostname"]["format"],
            json!("hostname")
        );
        assert_eq!(schema["properties"]["ipv4"]["format"], json!("ipv4"));
        assert_eq!(schema["properties"]["ipv6"]["format"], json!("ipv6"));
    }

    #[test]
    fn test_removes_nested_unsupported_formats() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "properties": {
                        "temp": {
                            "type": "number",
                            "format": "float"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "format": "uint32",
                            "minimum": 0
                        }
                    }
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "number",
                                "format": "double"
                            }
                        }
                    }
                }
            }
        });

        to_openai_compatible(&mut schema);

        // Unsupported formats should be removed at all nesting levels
        assert!(
            schema["properties"]["params"]["properties"]["temp"]
                .get("format")
                .is_none()
        );
        assert!(
            schema["properties"]["params"]["properties"]["max_tokens"]
                .get("format")
                .is_none()
        );
        // minimum should be preserved
        assert_eq!(
            schema["properties"]["params"]["properties"]["max_tokens"]["minimum"],
            json!(0)
        );
        assert!(
            schema["properties"]["items"]["items"]["properties"]["value"]
                .get("format")
                .is_none()
        );
    }

    #[test]
    fn test_normalizes_true_in_nested_properties() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "tool": {
                    "type": "object",
                    "properties": {
                        "parameters": true,
                        "name": { "type": "string" }
                    }
                }
            },
            "$defs": {
                "Config": {
                    "type": "object",
                    "properties": {
                        "data": true
                    }
                }
            }
        });

        to_openai_compatible(&mut schema);

        // `true` should be normalized at all levels
        assert_eq!(
            schema["properties"]["tool"]["properties"]["parameters"],
            json!({})
        );
        assert_eq!(schema["$defs"]["Config"]["properties"]["data"], json!({}));
    }

    #[test]
    fn test_normalizes_true_in_array_items() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": true
                }
            }
        });

        to_openai_compatible(&mut schema);

        // `true` in items should be normalized
        assert_eq!(schema["properties"]["items"]["items"], json!({}));
    }
}
