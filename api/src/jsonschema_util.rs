use jsonschema::JSONSchema;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::Error;

#[derive(Clone, Debug)]
pub struct JSONSchemaFromPath {
    path: PathBuf,
    compiled: Option<Arc<JSONSchema>>,
    pub value: Option<&'static serde_json::Value>,
}

impl JSONSchemaFromPath {
    /// Just instantiates the struct, does not load the schema
    /// You should call `load` to load the schema
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            compiled: None,
            value: None,
        }
    }

    #[cfg(test)]
    pub fn from_value(value: &serde_json::Value) -> Self {
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(value.clone()));
        let compiled_schema = JSONSchema::compile(schema_boxed).unwrap();
        Self {
            path: PathBuf::new(),
            compiled: Some(Arc::new(compiled_schema)),
            value: Some(schema_boxed),
        }
    }

    pub fn load<P: AsRef<Path>>(&mut self, base_path: Option<P>) -> Result<(), Error> {
        let path = match base_path {
            Some(base_path) => base_path.as_ref().join(&self.path),
            None => self.path.clone(),
        };
        let content = fs::read_to_string(path).map_err(|e| Error::JsonSchema {
            message: format!(
                "Failed to read JSON Schema `{}`: {}",
                self.path.display(),
                e
            ),
        })?;

        let schema: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| Error::JsonSchema {
                message: format!(
                    "Failed to parse JSON Schema `{}`: {}",
                    self.path.display(),
                    e
                ),
            })?;
        // We can 'leak' memory here because we want the schema to exist for the duration of the process
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(schema));
        let compiled_schema = JSONSchema::compile(schema_boxed).map_err(|e| Error::JsonSchema {
            message: format!(
                "Failed to compile JSON Schema `{}`: {}",
                self.path.display(),
                e
            ),
        })?;
        self.compiled = Some(Arc::new(compiled_schema));
        self.value = Some(schema_boxed);
        Ok(())
    }

    pub fn validate(&self, instance: &serde_json::Value) -> Result<(), Error> {
        match (&self.compiled, self.value) {
            (Some(compiled), Some(value)) => {
                compiled
                    .validate(instance)
                    .map_err(|e| Error::JsonSchemaValidation {
                        messages: e
                            .into_iter()
                            .map(|error| error.to_string())
                            .collect::<Vec<String>>(),
                        data: instance.clone(),
                        schema: value.clone(),
                    })
            }
            _ => Err(Error::JsonSchema {
                message: format!("JSON Schema `{}` not loaded", self.path.display()),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for JSONSchemaFromPath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path = PathBuf::deserialize(deserializer)?;
        Ok(JSONSchemaFromPath {
            path,
            compiled: None,
            value: None,
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

        let mut schema = JSONSchemaFromPath::new(temp_file.path().to_owned());
        schema
            .load::<&std::path::Path>(None)
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

        let mut schema = JSONSchemaFromPath::new(temp_file.path().to_owned());
        let result = schema.load::<&std::path::Path>(None);
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
        let mut schema = JSONSchemaFromPath::new(PathBuf::from("nonexistent_file.json"));
        let result = schema.load::<&std::path::Path>(None);
        assert_eq!(
            result.unwrap_err().to_string(),
            "Failed to read JSON Schema `nonexistent_file.json`: No such file or directory (os error 2)".to_string()
        )
    }
}
