use jsonschema::JSONSchema;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::Error;

#[derive(Debug, Serialize)]
pub struct JSONSchemaFromPath {
    #[serde(skip)]
    pub compiled: JSONSchema,
    pub value: &'static serde_json::Value,
}

impl PartialEq for JSONSchemaFromPath {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl JSONSchemaFromPath {
    /// Just instantiates the struct, does not load the schema
    /// You should call `load` to load the schema
    pub fn new<P: AsRef<Path>>(path: PathBuf, base_path: P) -> Result<Self, Error> {
        let path = base_path.as_ref().join(path);
        let content = fs::read_to_string(&path).map_err(|e| Error::JsonSchema {
            message: format!("Failed to read JSON Schema `{}`: {}", path.display(), e),
        })?;

        let schema: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| Error::JsonSchema {
                message: format!("Failed to parse JSON Schema `{}`: {}", path.display(), e),
            })?;
        // We can 'leak' memory here because we want the schema to exist for the duration of the process
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(schema));
        let compiled_schema = JSONSchema::compile(schema_boxed).map_err(|e| Error::JsonSchema {
            message: format!("Failed to compile JSON Schema `{}`: {}", path.display(), e),
        })?;
        let compiled = compiled_schema;
        let value = schema_boxed;
        Ok(Self { compiled, value })
    }

    #[cfg(any(test, feature = "integration_tests"))]
    pub fn from_value(value: &serde_json::Value) -> Self {
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(value.clone()));
        #[allow(clippy::unwrap_used)]
        let compiled_schema = JSONSchema::compile(schema_boxed).unwrap();
        Self {
            compiled: compiled_schema,
            value: schema_boxed,
        }
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

        let schema = JSONSchemaFromPath::new(temp_file.path().to_owned(), PathBuf::from(""))
            .expect("Failed to load schema");

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

        let result = JSONSchemaFromPath::new(temp_file.path().to_owned(), PathBuf::from(""));
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "Failed to compile JSON Schema `{}`: \"invalid\" is not valid under any of the schemas listed in the 'anyOf' keyword",
                temp_file.path().display()
            )
        )
    }

    #[test]
    fn test_nonexistent_file() {
        let result =
            JSONSchemaFromPath::new(PathBuf::from("nonexistent_file.json"), PathBuf::from(""));
        assert_eq!(
            result.unwrap_err().to_string(),
            "Failed to read JSON Schema `nonexistent_file.json`: No such file or directory (os error 2)".to_string()
        )
    }
}
