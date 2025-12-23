use jsonschema::Validator;
use once_cell::sync::OnceCell as SyncOnceCell;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::instrument;

use crate::config::path::ResolvedTomlPathData;
use crate::error::{Error, ErrorDetails};
use crate::utils::spawn_ignoring_shutdown;

#[derive(Debug, Serialize)]
pub enum JsonSchemaRef<'a> {
    Static(&'a StaticJSONSchema),
    Dynamic(&'a DynamicJSONSchema),
}

impl<'a> JsonSchemaRef<'a> {
    pub async fn validate(&self, instance: &Value) -> Result<(), Error> {
        match self {
            JsonSchemaRef::Static(schema) => schema.validate(instance),
            JsonSchemaRef::Dynamic(schema) => schema.validate(instance).await,
        }
    }

    pub fn value(&'a self) -> &'a Value {
        match self {
            JsonSchemaRef::Static(schema) => &schema.value,
            JsonSchemaRef::Dynamic(schema) => &schema.value,
        }
    }
}

/// A JSON schema with a lazily-compiled validator.
///
/// The validator is compiled when constructed from a config from disk, or on first access (via
/// `validate()` or `compiled()`) if not compiled. If it's deserialized from JSON, it's not compiled.
///
/// TODO(#5016): remove the distinction between Static and Dynamic JSONSchemas
#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct StaticJSONSchema {
    /// Lazily-compiled validator. Use `compiled()` method to access.
    #[serde(skip)]
    compiled: SyncOnceCell<Arc<Validator>>,
    pub value: serde_json::Value,
}

impl PartialEq for StaticJSONSchema {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Default for StaticJSONSchema {
    fn default() -> Self {
        Self {
            compiled: SyncOnceCell::new(),
            value: serde_json::json!({}),
        }
    }
}

impl StaticJSONSchema {
    /// Creates a StaticJSONSchema from a file path, eagerly compiling the schema.
    /// Returns an error if the JSON is invalid or the schema fails to compile.
    pub fn from_path(path: ResolvedTomlPathData) -> Result<Self, Error> {
        let content = path.data();

        let schema: serde_json::Value = serde_json::from_str(content).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!(
                    "Failed to parse JSON Schema `{}`: {}",
                    path.get_template_key(),
                    e
                ),
            })
        })?;
        let compiled_schema = jsonschema::validator_for(&schema).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!(
                    "Failed to compile JSON Schema `{}`: {}",
                    path.get_template_key(),
                    e
                ),
            })
        })?;
        let compiled = SyncOnceCell::new();
        // Eagerly initialize since we just compiled it
        let _ = compiled.set(Arc::new(compiled_schema));
        Ok(Self {
            compiled,
            value: schema,
        })
    }

    /// Creates a StaticJSONSchema from a JSON value, eagerly compiling the schema.
    /// Returns an error if the schema fails to compile.
    pub fn from_value(value: serde_json::Value) -> Result<Self, Error> {
        let compiled_schema = jsonschema::validator_for(&value).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!("Failed to compile JSON Schema: {e}"),
            })
        })?;
        let compiled = SyncOnceCell::new();
        // Eagerly initialize since we just compiled it
        let _ = compiled.set(Arc::new(compiled_schema));
        Ok(Self { compiled, value })
    }

    /// Gets the compiled validator, compiling lazily if needed.
    /// Returns an error if the schema is invalid.
    fn compiled(&self) -> Result<&Validator, Error> {
        self.compiled
            .get_or_try_init(|| {
                let validator = jsonschema::validator_for(&self.value).map_err(|e| {
                    Error::new(ErrorDetails::JsonSchema {
                        message: format!("Failed to compile JSON Schema: {e}"),
                    })
                })?;
                Ok(Arc::new(validator))
            })
            .map(|arc| arc.as_ref())
    }

    /// Validates an instance against this schema.
    /// The schema is compiled lazily on first validation if it hasn't been compiled yet.
    pub fn validate(&self, instance: &serde_json::Value) -> Result<(), Error> {
        self.compiled()?.validate(instance).map_err(|e| {
            Error::new(ErrorDetails::JsonSchemaValidation {
                messages: vec![e.to_string()],
                data: Box::new(instance.clone()),
                schema: Box::new(self.value.clone()),
            })
        })
    }
}

/// Wraps a schema with metadata indicating whether it was defined using legacy syntax
/// (e.g., `user_schema`, `assistant_schema`, `system_schema`) or new syntax (e.g., `schemas.<name>`).
/// This is used to determine whether to show a "Legacy" badge in the UI.
#[derive(Clone, Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct SchemaWithMetadata {
    pub schema: StaticJSONSchema,
    pub legacy_definition: bool,
}

/// This is a JSONSchema that is compiled on the fly.
/// This is useful for schemas that are not known at compile time, in particular, for dynamic tool definitions.
/// In order to avoid blocking the inference, we compile the schema asynchronously as the inference runs.
/// We use a tokio::sync::OnceCell to ensure that the schema is compiled only once
///
/// The public API of this struct should look very normal except validation is `async`
/// There are just `new` and `validate` methods.
///
/// TODO(#5016): remove the distinction between Static and Dynamic JSONSchemas
#[derive(Debug, Serialize, Clone, ts_rs::TS)]
#[ts(export)]
pub struct DynamicJSONSchema {
    pub value: Value,
    #[serde(skip)]
    compiled_schema: Arc<OnceCell<Validator>>,
}

impl PartialEq for DynamicJSONSchema {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl DynamicJSONSchema {
    pub fn new(schema: Value) -> Self {
        let compiled_schema = Arc::new(OnceCell::new());
        let this = Self {
            value: schema,
            compiled_schema,
        };
        let this_clone = this.clone();
        // Kick off the schema compilation in the background.
        // The first call to `validate` will either get the compiled schema (if the task finished),
        // or wait on the task to complete via the `OnceCell`
        spawn_ignoring_shutdown(async move {
            // If this errors, then we'll just get the error when we call 'validate'
            let _ = this_clone.get_or_init_compiled_schema().await;
        });
        this
    }

    pub async fn validate(&self, instance: &Value) -> Result<(), Error> {
        // This will block until the schema is compiled
        self.get_or_init_compiled_schema()
            .await?
            .validate(instance)
            .map_err(|e| {
                Error::new(ErrorDetails::JsonSchemaValidation {
                    messages: vec![e.to_string()],
                    data: Box::new(instance.clone()),
                    schema: Box::new(self.value.clone()),
                })
            })
    }

    /// Ensures that the schema is valid by forcing compilation.
    /// This is useful when you want to validate the schema itself without validating any instance.
    /// Returns an error if the schema is invalid.
    pub async fn ensure_valid(&self) -> Result<(), Error> {
        self.get_or_init_compiled_schema().await?;
        Ok(())
    }

    async fn get_or_init_compiled_schema(&self) -> Result<&Validator, Error> {
        self.compiled_schema
            .get_or_try_init(|| {
                let schema = self.value.clone();
                async {
                    // Use a blocking task, since `jsonschema::validator_for` is cpu-bound
                    tokio::task::spawn_blocking(move || {
                        jsonschema::validator_for(&schema).map_err(|e| {
                            Error::new(ErrorDetails::DynamicJsonSchema {
                                message: e.to_string(),
                            })
                        })
                    })
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::JsonSchema {
                            message: format!("Task join error in DynamicJSONSchema: {e}"),
                        })
                    })?
                }
            })
            .await
    }

    #[instrument]
    pub fn parse_from_str(s: &str) -> Result<Self, Error> {
        let schema = serde_json::from_str(s).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: e.to_string(),
            })
        })?;
        Ok(Self::new(schema))
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
        write!(temp_file, "{schema}").expect("Failed to write schema to temporary file");

        let schema = StaticJSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            temp_file.path().to_owned(),
            None,
        ))
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
        write!(temp_file, "{invalid_schema}")
            .expect("Failed to write invalid schema to temporary file");

        let result = StaticJSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            temp_file.path().to_owned(),
            None,
        ));
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "Failed to compile JSON Schema `{}`: \"invalid\" is not valid under any of the schemas listed in the 'anyOf' keyword",
                temp_file.path().display()
            )
        );
    }

    #[test]
    fn test_invalid_json_content() {
        // With eager loading, file contents are loaded during config parsing.
        // This test verifies that invalid JSON content produces the right error.
        let result = StaticJSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            "invalid_file.json".into(),
            Some("not valid json".to_string()),
        ));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Failed to parse JSON Schema"));
        assert!(err_msg.contains("invalid_file.json"));
    }

    #[tokio::test]
    async fn test_dynamic_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        let dynamic_schema = DynamicJSONSchema::new(schema);
        let instance = serde_json::json!({
            "name": "John Doe",
        });
        assert!(dynamic_schema.validate(&instance).await.is_ok());
    }
}
