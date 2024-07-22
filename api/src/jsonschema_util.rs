use jsonschema::JSONSchema;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct JSONSchemaFromPath(pub Arc<JSONSchema>);

impl JSONSchemaFromPath {
    /// Load a JSON schema from a file path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref().to_owned();
        let content = fs::read_to_string(path)?;
        let schema: serde_json::Value = serde_json::from_str(&content)?;
        // We can 'leak' memory here because we want the schema to exist for the duration of the process
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(schema));
        let compiled_schema = JSONSchema::compile(schema_boxed)?;
        Ok(Self(Arc::new(compiled_schema)))
    }
}

impl<'de> Deserialize<'de> for JSONSchemaFromPath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path = PathBuf::deserialize(deserializer)?;
        JSONSchemaFromPath::new(path).map_err(serde::de::Error::custom)
    }
}

// Implement Deref for convenience
impl std::ops::Deref for JSONSchemaFromPath {
    type Target = JSONSchema;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_valid_schema() {
        let schema = r#"
        {
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name"],
            "additionalProperties": false
        }
        "#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", schema).expect("Failed to write schema to temporary file");

        let schema =
            JSONSchemaFromPath::new(temp_file.path()).expect("Failed to create JSONSchemaFromPath");

        let instance = serde_json::json!({
            "name": "John Doe",
        });
        assert!(schema.is_valid(&instance));

        let instance = serde_json::json!({
            "name": "John Doe",
            "age": 30,
        });
        assert!(schema.is_valid(&instance));

        let instance = serde_json::json!({
            "name": "John Doe",
            "age": 30,
            "role": "admin"
        });
        assert!(!schema.is_valid(&instance));

        let instance = serde_json::json!({
            "age": "not a number"
        });
        assert!(!schema.is_valid(&instance));

        let instance = serde_json::json!({});
        assert!(!schema.is_valid(&instance));
    }

    #[test]
    fn test_invalid_schema() {
        let invalid_schema = r#"
        {
            "type": "invalid",
            "properties": {
                "name": { "type": "string" }
            }
        }
        "#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{}", invalid_schema)
            .expect("Failed to write invalid schema to temporary file");

        let result = JSONSchemaFromPath::new(temp_file.path());
        assert!(result.is_err()); // TODO: fix
    }

    #[test]
    fn test_nonexistent_file() {
        let result = JSONSchemaFromPath::new("nonexistent_file.json");
        assert!(result.is_err()); // TODO: fix
    }
}
