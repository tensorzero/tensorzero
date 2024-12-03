use jsonschema::JSONSchema;
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};
use tokio::task::JoinHandle;

use crate::error::{Error, ErrorDetails};

#[derive(Debug, Serialize)]
pub enum JsonSchemaRef<'a> {
    Static(&'a JSONSchemaFromPath),
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
            JsonSchemaRef::Static(schema) => schema.value,
            JsonSchemaRef::Dynamic(schema) => &schema.value,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct JSONSchemaFromPath {
    #[serde(skip)]
    pub compiled: Arc<JSONSchema>,
    pub value: &'static serde_json::Value,
}

impl PartialEq for JSONSchemaFromPath {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Default for JSONSchemaFromPath {
    fn default() -> Self {
        // Create an empty JSON object
        let empty_schema: serde_json::Value = serde_json::json!({});

        // Leak the memory to create a 'static reference
        let static_schema: &'static serde_json::Value = Box::leak(Box::new(empty_schema));

        // Compile the schema
        #[allow(clippy::expect_used)]
        let compiled_schema =
            JSONSchema::compile(static_schema).expect("Failed to compile empty schema");

        Self {
            compiled: Arc::new(compiled_schema),
            value: static_schema,
        }
    }
}

impl JSONSchemaFromPath {
    /// Just instantiates the struct, does not load the schema
    /// You should call `load` to load the schema
    pub fn new<P: AsRef<Path>>(path: PathBuf, base_path: P) -> Result<Self, Error> {
        let path = base_path.as_ref().join(path);
        let content = fs::read_to_string(&path).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!("Failed to read JSON Schema `{}`: {}", path.display(), e),
            })
        })?;

        let schema: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!("Failed to parse JSON Schema `{}`: {}", path.display(), e),
            })
        })?;
        // We can 'leak' memory here because we want the schema to exist for the duration of the process
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(schema));
        let compiled_schema = JSONSchema::compile(schema_boxed).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!("Failed to compile JSON Schema `{}`: {}", path.display(), e),
            })
        })?;
        let compiled = Arc::new(compiled_schema);
        let value = schema_boxed;
        Ok(Self { compiled, value })
    }

    // NOTE: This function is to be used only for tests and constants
    pub fn from_value(value: &serde_json::Value) -> Result<Self, Error> {
        let schema_boxed: &'static serde_json::Value = Box::leak(Box::new(value.clone()));
        let compiled_schema = JSONSchema::compile(schema_boxed).map_err(|e| {
            Error::new(ErrorDetails::JsonSchema {
                message: format!("Failed to compile JSON Schema: {}", e),
            })
        })?;
        Ok(Self {
            compiled: Arc::new(compiled_schema),
            value: schema_boxed,
        })
    }

    pub fn validate(&self, instance: &serde_json::Value) -> Result<(), Error> {
        self.compiled.validate(instance).map_err(|e| {
            Error::new(ErrorDetails::JsonSchemaValidation {
                messages: e
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<String>>(),
                data: Box::new(instance.clone()),
                schema: Box::new(self.value.clone()),
            })
        })
    }
}

type CompilationTask = Arc<RwLock<Option<JoinHandle<Result<Arc<JSONSchema>, Error>>>>>;
/// This is a JSONSchema that is compiled on the fly.
/// This is useful for schemas that are not known at compile time, in particular, for dynamic tool definitions.
/// In order to avoid blocking the inference, we compile the schema asynchronously as the inference runs.
/// We use a tokio::sync::OnceCell to ensure that the schema is compiled only once, and an RwLock to manage
/// interior mutability of the JoinHandle on the compilation task (I don't think this is strictly necessary but Rust will complain without it).
///
/// The public API of this struct should look very normal except validation is `async`
/// There are just `new` and `validate` methods.
#[derive(Debug, Serialize, Clone)]
pub struct DynamicJSONSchema {
    pub value: Value,
    #[serde(skip)]
    compiled_schema: OnceCell<Arc<JSONSchema>>,
    #[serde(skip)]
    compilation_task: CompilationTask,
}

impl PartialEq for DynamicJSONSchema {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl DynamicJSONSchema {
    pub fn new(schema: Value) -> Self {
        let schema_clone = schema.clone();
        let compilation_task = tokio::task::spawn_blocking(move || {
            JSONSchema::compile(&schema_clone)
                .map_err(|e| {
                    Error::new(ErrorDetails::DynamicJsonSchema {
                        message: e.to_string(),
                    })
                })
                .map(Arc::new)
        });
        Self {
            value: schema,
            compiled_schema: OnceCell::new(),
            compilation_task: Arc::new(RwLock::new(Some(compilation_task))),
        }
    }

    pub async fn validate(&self, instance: &Value) -> Result<(), Error> {
        // This will block until the schema is compiled
        // We don't take the result here because we want the mutable borrow to end
        self.get_or_init_compiled_schema().await?;

        let compiled_schema = match self.compiled_schema.get() {
            Some(compiled_schema) => compiled_schema,
            None => {
                return Err(Error::new(ErrorDetails::DynamicJsonSchema {
                    message: "Schema compilation failed".to_string(),
                }))
            }
        };

        compiled_schema.validate(instance).map_err(|e| {
            let messages = e.into_iter().map(|error| error.to_string()).collect();
            Error::new(ErrorDetails::JsonSchemaValidation {
                messages,
                data: Box::new(instance.clone()),
                schema: Box::new(self.value.clone()),
            })
        })
    }

    async fn get_or_init_compiled_schema(&self) -> Result<(), Error> {
        self.compiled_schema
            .get_or_try_init(|| async {
                let mut task_guard = self.compilation_task.write().await;
                if let Some(task) = task_guard.take() {
                    task.await.map_err(|e| {
                        Error::new(ErrorDetails::JsonSchema {
                            message: format!("Task join error in DynamicJSONSchema: {}", e),
                        })
                    })?
                } else {
                    Err(Error::new(ErrorDetails::JsonSchema {
                        message: "Schema compilation already completed.".to_string(),
                    }))
                }
            })
            .await?;
        Ok(())
    }

    pub fn from_str(s: &str) -> Result<Self, Error> {
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
