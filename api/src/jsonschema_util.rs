use jsonschema::JSONSchema;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::Error;

#[derive(Clone, Debug)]
pub struct JSONSchemaFromPath {
    compiled: Arc<JSONSchema>,
    pub value: &'static serde_json::Value,
}

impl JSONSchemaFromPath {
    /// Load a JSON schema from a file path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref().to_owned();
        let content = fs::read_to_string(path)?;
        let schema: serde_json::Value = serde_json::from_str(&content)?;
        // We can 'leak' memory here because we want the schema to exist for the duration of the process
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(schema));
        let compiled_schema = JSONSchema::compile(schema_boxed)?;
        Ok(Self {
            compiled: Arc::new(compiled_schema),
            value: schema_boxed,
        })
    }

    pub fn validate(&self, instance: &serde_json::Value) -> Result<(), Error> {
        self.compiled
            .validate(instance)
            .map_err(|e| Error::JsonSchemaValidation {
                messages: e
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<String>>(),
                data: instance.clone(),
                schema: self.value.clone(),
            })
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
        assert!(schema.validate(&instance).is_ok());

        let instance = serde_json::json!({
            "name": "John Doe",
            "age": 30,
        });
        assert!(schema.validate(&instance).is_ok());

        let instance = serde_json::json!({
            "name": "John Doe",
            "age": 30,
            "role": "admin"
        });
        assert!(schema.validate(&instance).is_err());

        let instance = serde_json::json!({
            "age": "not a number"
        });
        assert!(schema.validate(&instance).is_err());

        let instance = serde_json::json!({});
        assert!(schema.validate(&instance).is_err());
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
        assert_eq!(
            result.unwrap_err().to_string(),
            "\"invalid\" is not valid under any of the schemas listed in the 'anyOf' keyword"
                .to_string()
        )
    }

    #[test]
    fn test_nonexistent_file() {
        let result = JSONSchemaFromPath::new("nonexistent_file.json");
        assert_eq!(
            result.unwrap_err().to_string(),
            "No such file or directory (os error 2)".to_string()
        )
    }
}
