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
