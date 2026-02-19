use jsonschema::Validator;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::instrument;

use crate::config::path::ResolvedTomlPathData;
use crate::error::{Error, ErrorDetails};
use crate::utils::spawn_ignoring_shutdown;

/// A JSON schema with a lazily-compiled validator.
///
/// The validator is compiled asynchronously on first access (via `validate()` or `ensure_valid()`).
/// Compilation is kicked off in the background when the schema is created via `compile_background()`,
/// so it should typically be ready by the time validation is needed.
///
/// When created via `compile()`, `from_path()`, or `from_value()`, the schema is compiled
/// synchronously and the compiled validator is stored immediately.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct JSONSchema {
    pub value: Value,
    #[serde(skip)]
    compiled: Arc<OnceCell<Validator>>,
}

impl PartialEq for JSONSchema {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Default for JSONSchema {
    fn default() -> Self {
        // Create an empty schema that accepts any object
        Self {
            value: serde_json::json!({}),
            compiled: Arc::new(OnceCell::new()),
        }
    }
}

impl JSONSchema {
    /// Creates a new JSONSchema with a pre-compiled validator.
    fn new_with_compiled(value: Value, validator: Validator) -> Result<Self, Error> {
        let compiled = Arc::new(OnceCell::new());
        compiled.set(validator).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!("Failed to set validator in OnceCell: {e}"),
            })
        })?;
        Ok(Self { value, compiled })
    }

    /// Creates a new JSONSchema with lazy compilation (no pre-compiled validator).
    fn new_lazy(schema: Value) -> Self {
        Self {
            value: schema,
            compiled: Arc::new(OnceCell::new()),
        }
    }

    /// Compiles a JSON schema synchronously.
    ///
    /// This is the preferred method for config loading where no tokio runtime is available.
    /// The schema is compiled immediately and the compiled validator is stored for reuse.
    /// Returns an error if the schema fails to compile.
    pub fn compile(schema: Value) -> Result<Self, Error> {
        let validator = jsonschema::validator_for(&schema).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!("Failed to compile JSON Schema: {e}"),
            })
        })?;
        Self::new_with_compiled(schema, validator)
    }

    /// Compiles a JSON schema asynchronously in the background.
    ///
    /// This is the preferred method for runtime-created schemas (e.g., during inference).
    /// Kicks off compilation in a background task so it should typically be ready
    /// by the time validation is needed.
    ///
    /// **Note**: This must be called from within a tokio runtime context.
    pub fn compile_background(schema: Value) -> Self {
        let this = Self::new_lazy(schema);
        let this_clone = this.clone();
        // Kick off the schema compilation in the background.
        // The first call to `validate` will either get the compiled schema (if the task finished),
        // or wait on the task to complete via the `OnceCell`
        spawn_ignoring_shutdown(async move {
            // If this errors, then we'll just get the error when we call 'validate'
            let _ = this_clone.get_or_init_compiled().await;
        });
        this
    }

    /// Creates a JSONSchema from a file path.
    ///
    /// Parses the JSON and compiles the schema synchronously.
    /// Returns an error if the JSON is invalid or the schema fails to compile.
    /// Safe to call outside of a tokio runtime.
    pub fn from_path(path: ResolvedTomlPathData) -> Result<Self, Error> {
        let content = path.data();

        let schema: Value = serde_json::from_str(content).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!(
                    "Failed to parse JSON Schema `{}`: {}",
                    path.get_template_key(),
                    e
                ),
            })
        })?;

        let validator = jsonschema::validator_for(&schema).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!(
                    "Failed to compile JSON Schema `{}`: {}",
                    path.get_template_key(),
                    e
                ),
            })
        })?;

        Self::new_with_compiled(schema, validator)
    }

    /// Creates a JSONSchema from a JSON value.
    ///
    /// Compiles the schema synchronously. This is an alias for `compile()`.
    /// Returns an error if the schema fails to compile.
    /// Safe to call outside of a tokio runtime.
    pub fn from_value(value: Value) -> Result<Self, Error> {
        Self::compile(value)
    }

    /// Validates an instance against this schema.
    ///
    /// If the schema hasn't been compiled yet, it will be compiled asynchronously first.
    pub async fn validate(&self, instance: &Value) -> Result<(), Error> {
        self.get_or_init_compiled()
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
    ///
    /// This is useful when you want to validate the schema itself without validating any instance.
    /// Returns an error if the schema is invalid.
    pub async fn ensure_valid(&self) -> Result<(), Error> {
        self.get_or_init_compiled().await?;
        Ok(())
    }

    async fn get_or_init_compiled(&self) -> Result<&Validator, Error> {
        self.compiled
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
                            message: format!("Task join error in JSONSchema: {e}"),
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
        Ok(Self::compile_background(schema))
    }
}

/// Wraps a schema with metadata indicating whether it was defined using legacy syntax
/// (e.g., `user_schema`, `assistant_schema`, `system_schema`) or new syntax (e.g., `schemas.<name>`).
/// This is used to determine whether to show a "Legacy" badge in the UI.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct SchemaWithMetadata {
    pub schema: JSONSchema,
    pub legacy_definition: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_valid_schema() {
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

        let schema = JSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            temp_file.path().to_owned(),
            None,
        ))
        .expect("Failed to load schema");

        let instance = serde_json::json!({
            "name": "John Doe",
        });
        assert!(schema.validate(&instance).await.is_ok());

        let instance = serde_json::json!({
            "name": "John Doe",
            "age": 30,
        });
        assert!(schema.validate(&instance).await.is_ok());

        let instance = serde_json::json!({
            "name": "John Doe",
            "age": 30,
            "role": "admin"
        });
        assert!(schema.validate(&instance).await.is_err());

        let instance = serde_json::json!({
            "age": "not a number"
        });
        assert!(schema.validate(&instance).await.is_err());

        let instance = serde_json::json!({});
        assert!(schema.validate(&instance).await.is_err());
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

        let result = JSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
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
        let result = JSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            "invalid_file.json".into(),
            Some("not valid json".to_string()),
        ));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Failed to parse JSON Schema"));
        assert!(err_msg.contains("invalid_file.json"));
    }

    #[tokio::test]
    async fn test_compile_background() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        let dynamic_schema = JSONSchema::compile_background(schema);
        let instance = serde_json::json!({
            "name": "John Doe",
        });
        assert!(dynamic_schema.validate(&instance).await.is_ok());
    }

    #[test]
    fn test_compile_sync() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        // compile() should work without a tokio runtime
        let compiled_schema = JSONSchema::compile(schema).expect("Failed to compile schema");
        // The schema should be pre-compiled
        assert!(compiled_schema.compiled.get().is_some());
    }

    #[test]
    fn test_compile_invalid_schema() {
        let invalid_schema = serde_json::json!({
            "type": "invalid_type"
        });

        let result = JSONSchema::compile(invalid_schema);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ensure_valid() {
        let valid_schema = JSONSchema::compile_background(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        }));
        assert!(valid_schema.ensure_valid().await.is_ok());

        let invalid_schema = JSONSchema::compile_background(serde_json::json!({
            "type": "invalid_type"
        }));
        assert!(invalid_schema.ensure_valid().await.is_err());
    }

    #[tokio::test]
    async fn test_parse_from_str() {
        let schema_str = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
        let schema = JSONSchema::parse_from_str(schema_str).expect("Failed to parse schema");

        let instance = serde_json::json!({"name": "test"});
        assert!(schema.validate(&instance).await.is_ok());
    }

    #[tokio::test]
    async fn test_default_schema() {
        let schema = JSONSchema::default();
        // Default schema should accept any object
        let instance = serde_json::json!({"anything": "goes"});
        assert!(schema.validate(&instance).await.is_ok());
        // The default schema should be pre-compiled
        assert!(schema.compiled.get().is_some());
    }

    #[test]
    fn test_from_value_reuses_compiled() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });

        let compiled_schema = JSONSchema::from_value(schema).expect("Failed to compile schema");
        // The schema should be pre-compiled (not lazy)
        assert!(
            compiled_schema.compiled.get().is_some(),
            "from_value should store the compiled validator"
        );
    }

    #[test]
    fn test_from_path_reuses_compiled() {
        let schema_json = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{schema_json}").expect("Failed to write schema to temporary file");

        let compiled_schema = JSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            temp_file.path().to_owned(),
            None,
        ))
        .expect("Failed to load schema");

        // The schema should be pre-compiled (not lazy)
        assert!(
            compiled_schema.compiled.get().is_some(),
            "from_path should store the compiled validator"
        );
    }
}
