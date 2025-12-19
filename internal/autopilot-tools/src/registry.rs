use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use crate::client_tool::{ClientTool, ErasedClientTool};

/// Errors that can occur when working with the tool registry.
#[derive(Debug, Clone, Error)]
pub enum RegistryError {
    /// A tool with the given name is already registered.
    #[error("a tool with the name '{name}' is already registered")]
    DuplicateTool { name: String },
}

/// A registry of client-side tools.
///
/// The registry stores metadata about tools that can be executed client-side.
/// It provides methods to register tools, look them up by name, and convert
/// them to TensorZero tool definitions for use in LLM function calling.
///
/// # Example
///
/// ```ignore
/// use autopilot_tools::{ClientToolRegistry, ClientTool};
///
/// let mut registry = ClientToolRegistry::new();
/// registry.register::<ReadFileTool>();
/// registry.register::<WriteFileTool>();
///
/// // Look up a tool by name
/// if let Some(tool) = registry.get("read_file") {
///     println!("Found tool: {}", tool.name());
/// }
///
/// // List all tools
/// for tool in registry.list() {
///     println!("Tool: {} - {}", tool.name(), tool.description());
/// }
/// ```
#[derive(Default)]
pub struct ClientToolRegistry {
    tools: HashMap<String, Arc<dyn ErasedClientTool>>,
}

impl ClientToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a client tool type.
    ///
    /// The tool will be stored with its name as the key.
    ///
    /// # Errors
    ///
    /// Returns an error if a tool with the same name is already registered.
    pub fn register<T: ClientTool + Default + Send + Sync + 'static>(
        &mut self,
    ) -> Result<(), RegistryError> {
        let tool = T::default();
        let name = T::name().into_owned();
        if self.tools.contains_key(&name) {
            return Err(RegistryError::DuplicateTool { name });
        }
        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    /// Get a tool by name.
    ///
    /// Returns `None` if no tool with the given name is registered.
    pub fn get(&self, name: &str) -> Option<&dyn ErasedClientTool> {
        self.tools.get(name).map(|arc| arc.as_ref())
    }

    /// Get a tool Arc by name (for sharing ownership).
    ///
    /// Returns `None` if no tool with the given name is registered.
    pub fn get_arc(&self, name: &str) -> Option<Arc<dyn ErasedClientTool>> {
        self.tools.get(name).cloned()
    }

    /// List all registered tools.
    pub fn list(&self) -> Vec<&dyn ErasedClientTool> {
        self.tools.values().map(|arc| arc.as_ref()).collect()
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Check if a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get an iterator over tool names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::{Schema, schema_for};
    use serde::{Deserialize, Serialize};
    use std::borrow::Cow;

    #[derive(Serialize, Deserialize, schemars::JsonSchema)]
    struct TestParams {
        value: String,
    }

    #[derive(Default)]
    struct TestTool;

    impl ClientTool for TestTool {
        type LlmParams = TestParams;

        fn name() -> Cow<'static, str> {
            Cow::Borrowed("test_tool")
        }

        fn description() -> Cow<'static, str> {
            Cow::Borrowed("A test tool")
        }

        fn parameters_schema() -> Schema {
            schema_for!(TestParams)
        }
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ClientToolRegistry::new();
        registry.register::<TestTool>().unwrap();

        let tool = registry.get("test_tool");
        assert!(tool.is_some());
        let tool = tool.unwrap();
        assert_eq!(tool.name(), "test_tool");
        assert_eq!(tool.description(), "A test tool");
    }

    #[test]
    fn test_registry_list() {
        let mut registry = ClientToolRegistry::new();
        registry.register::<TestTool>().unwrap();

        let tools = registry.list();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "test_tool");
    }

    #[test]
    fn test_registry_contains() {
        let mut registry = ClientToolRegistry::new();
        registry.register::<TestTool>().unwrap();

        assert!(registry.contains("test_tool"));
        assert!(!registry.contains("nonexistent"));
    }

    #[test]
    fn test_registry_duplicate_registration_error() {
        let mut registry = ClientToolRegistry::new();
        registry.register::<TestTool>().unwrap();

        let result = registry.register::<TestTool>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateTool { ref name } if name == "test_tool"));
        assert_eq!(
            err.to_string(),
            "a tool with the name 'test_tool' is already registered"
        );
    }
}
