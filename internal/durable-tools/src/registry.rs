use async_trait::async_trait;
use indexmap::IndexMap;
use schemars::Schema;
use serde_json::Value as JsonValue;
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::{FunctionTool, Tool};

use crate::ToolResult;
use crate::context::SimpleToolContext;
use crate::error::{NonControlToolError, ToolError};
use crate::simple_tool::SimpleTool;
use crate::task_tool::TaskTool;
use crate::tool_metadata::ToolMetadata;

/// Conversion from an erased tool reference to a TensorZero Tool definition.
impl TryFrom<&dyn ErasedTool> for Tool {
    type Error = ToolError;

    fn try_from(tool: &dyn ErasedTool) -> Result<Self, Self::Error> {
        Ok(Tool::Function(FunctionTool {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: serde_json::to_value(tool.parameters_schema()?)?,
            strict: false,
        }))
    }
}

/// Type-erased tool trait for registry storage.
///
/// This provides metadata about a tool without exposing its concrete types.
pub trait ErasedTool: Send + Sync {
    /// Get the tool's unique name.
    fn name(&self) -> Cow<'static, str>;

    /// Get the tool's description.
    fn description(&self) -> Cow<'static, str>;

    /// Get the JSON Schema for the tool's parameters.
    fn parameters_schema(&self) -> ToolResult<Schema>;

    /// Get the tool's execution timeout.
    fn timeout(&self) -> Duration;

    /// Check if this is a durable tool (`TaskTool`) or lightweight (`SimpleTool`).
    fn is_durable(&self) -> bool;

    /// Validate that the provided JSON can be deserialized into the tool's parameter types.
    ///
    /// This allows validating parameters before spawning a job, catching errors early.
    fn validate_params(&self, llm_params: &JsonValue, side_info: &JsonValue) -> ToolResult<()>;
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
pub struct ErasedTaskToolWrapper<T: TaskTool>(Arc<T>);

impl<T: TaskTool> ErasedTaskToolWrapper<T> {
    /// Create a new wrapper with the given tool instance.
    pub fn new(tool: Arc<T>) -> Self {
        Self(tool)
    }
}

impl<T: TaskTool> ErasedTool for ErasedTaskToolWrapper<T> {
    fn name(&self) -> Cow<'static, str> {
        self.0.name()
    }

    fn description(&self) -> Cow<'static, str> {
        self.0.description()
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        self.0.parameters_schema()
    }

    fn timeout(&self) -> Duration {
        self.0.timeout()
    }

    fn is_durable(&self) -> bool {
        true
    }

    fn validate_params(&self, llm_params: &JsonValue, side_info: &JsonValue) -> ToolResult<()> {
        let _: <T as ToolMetadata>::LlmParams = serde_json::from_value(llm_params.clone())
            .map_err(|e| NonControlToolError::InvalidParams {
                message: format!("llm_params: {e}"),
            })?;
        let _: T::SideInfo = serde_json::from_value(side_info.clone()).map_err(|e| {
            NonControlToolError::InvalidParams {
                message: format!("side_info: {e}"),
            }
        })?;
        Ok(())
    }
}

/// Blanket implementation of [`ErasedTool`] for all `SimpleTool` types.
impl<T: SimpleTool> ErasedTool for T {
    fn name(&self) -> Cow<'static, str> {
        ToolMetadata::name(self)
    }

    fn description(&self) -> Cow<'static, str> {
        ToolMetadata::description(self)
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        ToolMetadata::parameters_schema(self)
    }

    fn timeout(&self) -> Duration {
        ToolMetadata::timeout(self)
    }

    fn is_durable(&self) -> bool {
        false
    }

    fn validate_params(&self, llm_params: &JsonValue, side_info: &JsonValue) -> ToolResult<()> {
        let _: <T as ToolMetadata>::LlmParams = serde_json::from_value(llm_params.clone())
            .map_err(|e| NonControlToolError::InvalidParams {
                message: format!("llm_params: {e}"),
            })?;
        let _: T::SideInfo = serde_json::from_value(side_info.clone()).map_err(|e| {
            NonControlToolError::InvalidParams {
                message: format!("side_info: {e}"),
            }
        })?;
        Ok(())
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
        let typed_llm_params: <T as ToolMetadata>::LlmParams = serde_json::from_value(llm_params)?;
        let typed_side_info: T::SideInfo = serde_json::from_value(side_info)?;

        // Execute (static method)
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
/// We use an IndexMap to get consistent iteration order across runs,
/// which is important for LLM request/prompt caching
/// (the tools get passed to the LLM input)
pub struct ToolRegistry {
    /// All tools (for metadata queries).
    tools: IndexMap<String, Arc<dyn ErasedTool>>,
    /// `SimpleTools` specifically (for step execution).
    simple_tools: IndexMap<String, Arc<dyn ErasedSimpleTool>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
            simple_tools: IndexMap::new(),
        }
    }

    /// Register a `TaskTool` instance.
    ///
    /// # Errors
    ///
    /// Returns `NonControlToolError::DuplicateToolName` if a tool with the same name is already registered.
    pub fn register_task_tool_instance<T: TaskTool>(
        &mut self,
        tool: T,
    ) -> Result<Arc<T>, ToolError> {
        let tool = Arc::new(tool);
        let name = tool.name();
        if self.tools.contains_key(name.as_ref()) {
            return Err(NonControlToolError::DuplicateToolName {
                name: name.into_owned(),
            }
            .into());
        }

        let wrapper = Arc::new(ErasedTaskToolWrapper::new(tool.clone()));
        self.tools.insert(name.into_owned(), wrapper);
        Ok(tool)
    }

    /// Register a `SimpleTool` instance.
    ///
    /// # Errors
    ///
    /// Returns `NonControlToolError::DuplicateToolName` if a tool with the same name is already registered.
    pub fn register_simple_tool_instance<T: SimpleTool>(
        &mut self,
        tool: T,
    ) -> Result<Arc<T>, ToolError> {
        let tool = Arc::new(tool);
        let name = tool.name();
        if self.tools.contains_key(name.as_ref()) {
            return Err(NonControlToolError::DuplicateToolName {
                name: name.into_owned(),
            }
            .into());
        }

        let name = name.into_owned();
        self.tools
            .insert(name.clone(), tool.clone() as Arc<dyn ErasedTool>);
        self.simple_tools
            .insert(name, tool.clone() as Arc<dyn ErasedSimpleTool>);
        Ok(tool)
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

    /// Validate parameters for a tool by name.
    ///
    /// Returns `NonControlToolError::ToolNotFound` if the tool doesn't exist.
    /// Returns `NonControlToolError::InvalidParams` if validation fails.
    pub fn validate_params(
        &self,
        tool_name: &str,
        llm_params: &JsonValue,
        side_info: &JsonValue,
    ) -> ToolResult<()> {
        let tool = self
            .get(tool_name)
            .ok_or_else(|| NonControlToolError::ToolNotFound {
                name: tool_name.to_string(),
            })?;
        tool.validate_params(llm_params, side_info)
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

    /// Iterate over all registered tools.
    ///
    /// Use with `Tool::try_from` to convert to TensorZero tool definitions:
    /// ```ignore
    /// let tools: Result<Vec<Tool>, _> = registry
    ///     .iter()
    ///     .map(Tool::try_from)
    ///     .collect();
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &dyn ErasedTool> {
        self.tools.values().map(|arc| arc.as_ref())
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
