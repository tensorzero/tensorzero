use async_trait::async_trait;
use schemars::schema::RootSchema;
use serde_json::Value as JsonValue;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::{FunctionTool, Tool};

use crate::context::SimpleToolContext;
use crate::simple_tool::SimpleTool;
use crate::task_tool::TaskTool;

/// Type-erased tool trait for registry storage.
///
/// This provides metadata about a tool without exposing its concrete types.
pub trait ErasedTool: Send + Sync {
    /// Get the tool's unique name.
    fn name(&self) -> &'static str;

    /// Get the tool's description.
    fn description(&self) -> Cow<'static, str>;

    /// Get the JSON Schema for the tool's parameters.
    fn parameters_schema(&self) -> RootSchema;

    /// Get the tool's execution timeout.
    fn timeout(&self) -> Duration;

    /// Check if this is a durable tool (`TaskTool`) or lightweight (`SimpleTool`).
    fn is_durable(&self) -> bool;
}

/// Type-erased `SimpleTool` trait for dynamic execution.
///
/// This extends `ErasedTool` with the ability to execute with JSON params.
#[async_trait]
pub trait ErasedSimpleTool: ErasedTool {
    /// Execute the tool with JSON parameters.
    ///
    /// # Arguments
    ///
    /// * `llm_params` - The LLM-provided parameters
    /// * `side_info` - Side information (hidden from LLM)
    /// * `ctx` - The simple tool context
    /// * `idempotency_key` - A unique key for this execution
    async fn execute_erased(
        &self,
        llm_params: JsonValue,
        side_info: JsonValue,
        ctx: SimpleToolContext<'_>,
        idempotency_key: &str,
    ) -> anyhow::Result<JsonValue>;
}

/// Wrapper that implements [`ErasedTool`] for `TaskTool` types.
pub struct ErasedTaskToolWrapper<T: TaskTool>(std::marker::PhantomData<T>);

impl<T: TaskTool> ErasedTaskToolWrapper<T> {
    /// Create a new wrapper.
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: TaskTool> Default for ErasedTaskToolWrapper<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TaskTool> ErasedTool for ErasedTaskToolWrapper<T> {
    fn name(&self) -> &'static str {
        T::NAME
    }

    fn description(&self) -> Cow<'static, str> {
        T::description()
    }

    fn parameters_schema(&self) -> RootSchema {
        T::parameters_schema()
    }

    fn timeout(&self) -> Duration {
        T::timeout()
    }

    fn is_durable(&self) -> bool {
        true
    }
}

/// Blanket implementation of [`ErasedTool`] for all `SimpleTool` types.
impl<T: SimpleTool> ErasedTool for T {
    fn name(&self) -> &'static str {
        T::NAME
    }

    fn description(&self) -> Cow<'static, str> {
        T::description()
    }

    fn parameters_schema(&self) -> RootSchema {
        T::parameters_schema()
    }

    fn timeout(&self) -> Duration {
        T::timeout()
    }

    fn is_durable(&self) -> bool {
        false
    }
}

/// Blanket implementation of [`ErasedSimpleTool`] for all `SimpleTool` types.
#[async_trait]
impl<T: SimpleTool> ErasedSimpleTool for T {
    async fn execute_erased(
        &self,
        llm_params: JsonValue,
        side_info: JsonValue,
        ctx: SimpleToolContext<'_>,
        idempotency_key: &str,
    ) -> anyhow::Result<JsonValue> {
        // Deserialize params
        let typed_llm_params: T::LlmParams = serde_json::from_value(llm_params)?;
        let typed_side_info: T::SideInfo = serde_json::from_value(side_info)?;

        // Execute
        let result = T::execute(typed_llm_params, typed_side_info, ctx, idempotency_key).await?;

        // Serialize output
        Ok(serde_json::to_value(&result)?)
    }
}

/// Registry of tools, supporting both `TaskTools` and `SimpleTools`.
///
/// The registry stores type-erased tool wrappers that can be used for:
/// - Looking up tool metadata
/// - Generating LLM function definitions
/// - Dynamically invoking tools by name
pub struct ToolRegistry {
    /// All tools (for metadata queries).
    tools: HashMap<String, Arc<dyn ErasedTool>>,
    /// `SimpleTools` specifically (for step execution).
    simple_tools: HashMap<String, Arc<dyn ErasedSimpleTool>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            simple_tools: HashMap::new(),
        }
    }

    /// Register a `TaskTool`.
    ///
    /// Returns `&mut Self` for chaining.
    pub fn register_task_tool<T: TaskTool>(&mut self) -> &mut Self {
        let wrapper = Arc::new(ErasedTaskToolWrapper::<T>::new());
        self.tools.insert(T::NAME.to_string(), wrapper);
        self
    }

    /// Register a `SimpleTool`.
    ///
    /// Returns `&mut Self` for chaining.
    pub fn register_simple_tool<T: SimpleTool + Default>(&mut self) -> &mut Self {
        let tool = Arc::new(T::default());
        self.tools
            .insert(T::NAME.to_string(), tool.clone() as Arc<dyn ErasedTool>);
        self.simple_tools.insert(T::NAME.to_string(), tool);
        self
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn ErasedTool> {
        self.tools.get(name).map(AsRef::as_ref)
    }

    /// Get a `SimpleTool` by name for execution.
    pub fn get_simple_tool(&self, name: &str) -> Option<Arc<dyn ErasedSimpleTool>> {
        self.simple_tools.get(name).cloned()
    }

    /// Check if a tool is durable (`TaskTool`) or lightweight (`SimpleTool`).
    ///
    /// Returns `None` if the tool is not found.
    pub fn is_durable(&self, name: &str) -> Option<bool> {
        self.tools.get(name).map(|t| t.is_durable())
    }

    /// List all registered tool names.
    pub fn list_tools(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// List all `TaskTool` names.
    pub fn list_task_tools(&self) -> Vec<&str> {
        self.tools
            .iter()
            .filter(|(_, t)| t.is_durable())
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// List all `SimpleTool` names.
    pub fn list_simple_tools(&self) -> Vec<&str> {
        self.simple_tools.keys().map(String::as_str).collect()
    }

    /// Generate TensorZero tool definitions for all tools.
    ///
    /// This can be used directly in TensorZero inference API calls.
    ///
    /// # Errors
    ///
    /// Returns an error if a tool's parameter schema fails to serialize.
    pub fn to_tensorzero_tools(&self) -> Result<Vec<Tool>, serde_json::Error> {
        self.tools
            .values()
            .map(|tool| {
                Ok(Tool::Function(FunctionTool {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: serde_json::to_value(tool.parameters_schema())?,
                    strict: false,
                }))
            })
            .collect()
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
