//! TypeScript tool types and implementations.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use schemars::Schema;
use serde_json::Value as JsonValue;

use super::error::TypeScriptToolError;
use super::transpile::transpile_typescript;
use crate::ToolContext;
use crate::error::{NonControlToolError, ToolResult};
use crate::task_tool::TaskTool;
use crate::tool_metadata::ToolMetadata;

/// A tool implemented in TypeScript.
///
/// The TypeScript code must export a default object with:
/// - `name`: string - The tool name
/// - `description`: string - Tool description
/// - `parameters_schema`: object - JSON Schema for LLM parameters
/// - `run(params, sideInfo)`: async function - The tool implementation
///
/// The `run` function has access to the global `ctx` object which provides:
/// - `ctx.taskId()` - Get the current task ID
/// - `ctx.episodeId()` - Get the current episode ID
/// - `ctx.callTool(name, llmParams, sideInfo)` - Call another tool
/// - `ctx.spawnTool(name, llmParams, sideInfo)` - Spawn a background tool
/// - `ctx.joinTool(handle)` - Wait for a spawned tool
/// - `ctx.inference(params)` - Make an inference call
/// - `ctx.rand()` - Get a durable random number
/// - `ctx.now()` - Get the durable current time
/// - `ctx.uuid7()` - Generate a durable UUID
/// - `ctx.sleepFor(name, durationMs)` - Durable sleep
/// - `ctx.awaitEvent(name, timeoutMs?)` - Wait for an event
/// - `ctx.emitEvent(name, payload)` - Emit an event
#[derive(Debug, Clone)]
pub struct TypeScriptTool {
    name: Cow<'static, str>,
    description: Cow<'static, str>,
    typescript_code: String,
    /// Pre-transpiled JavaScript code (transpiled at build time)
    js_code: String,
    parameters_schema: JsonValue,
    timeout: Duration,
}

impl TypeScriptTool {
    /// Create a new builder for a TypeScript tool.
    pub fn builder(name: impl Into<Cow<'static, str>>) -> TypeScriptToolBuilder {
        TypeScriptToolBuilder::new(name)
    }

    /// Get the pre-transpiled JavaScript code.
    pub fn js_code(&self) -> &str {
        &self.js_code
    }

    /// Get the original TypeScript code.
    pub fn typescript_code(&self) -> &str {
        &self.typescript_code
    }
}

/// Builder for constructing a `TypeScriptTool`.
pub struct TypeScriptToolBuilder {
    name: Cow<'static, str>,
    description: Option<Cow<'static, str>>,
    typescript_code: Option<String>,
    parameters_schema: Option<JsonValue>,
    timeout: Duration,
}

impl TypeScriptToolBuilder {
    /// Create a new builder with the given tool name.
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            description: None,
            typescript_code: None,
            parameters_schema: None,
            timeout: Duration::from_secs(60),
        }
    }

    /// Set the tool description.
    #[must_use]
    pub fn description(mut self, description: impl Into<Cow<'static, str>>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the TypeScript source code.
    #[must_use]
    pub fn typescript_code(mut self, code: impl Into<String>) -> Self {
        self.typescript_code = Some(code.into());
        self
    }

    /// Set the parameters JSON schema.
    #[must_use]
    pub fn parameters_schema(mut self, schema: JsonValue) -> Self {
        self.parameters_schema = Some(schema);
        self
    }

    /// Set the tool timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Build the TypeScript tool.
    ///
    /// This transpiles the TypeScript code to JavaScript at build time.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `typescript_code` was not provided
    /// - TypeScript transpilation fails
    pub fn build(self) -> Result<TypeScriptTool, TypeScriptToolError> {
        let typescript_code = self.typescript_code.ok_or_else(|| {
            TypeScriptToolError::InvalidTool("typescript_code is required".into())
        })?;

        // Transpile once at build time
        let js_code = transpile_typescript(&typescript_code)?;

        let description = self.description.unwrap_or(Cow::Borrowed(""));

        let parameters_schema = self.parameters_schema.unwrap_or_else(|| {
            serde_json::json!({
                "type": "object",
                "properties": {},
            })
        });

        Ok(TypeScriptTool {
            name: self.name,
            description,
            typescript_code,
            js_code,
            parameters_schema,
            timeout: self.timeout,
        })
    }
}

/// Instance-aware TypeScript tool that can be registered with the executor.
///
/// This is a wrapper that holds the actual TypeScript code and metadata,
/// allowing the tool to be executed with instance-specific configuration.
///
/// Unlike the base `TypeScriptTool`, this type implements `TaskTool`
/// and uses the `JsRuntimePool` from `ToolAppState` for execution.
#[derive(Clone)]
pub struct TypeScriptToolInstance {
    tool: Arc<TypeScriptTool>,
}

impl TypeScriptToolInstance {
    /// Create a new instance from a TypeScriptTool.
    pub fn new(tool: TypeScriptTool) -> Self {
        Self {
            tool: Arc::new(tool),
        }
    }

    /// Get the tool name.
    pub fn tool_name(&self) -> &str {
        &self.tool.name
    }

    /// Get the tool description.
    pub fn tool_description(&self) -> &str {
        &self.tool.description
    }

    /// Get the parameters schema as JSON.
    pub fn parameters_schema_json(&self) -> &JsonValue {
        &self.tool.parameters_schema
    }
}

impl ToolMetadata for TypeScriptToolInstance {
    type LlmParams = JsonValue;
    type SideInfo = JsonValue;
    type Output = JsonValue;

    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.tool.name.to_string())
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(self.tool.description.to_string())
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        serde_json::from_value(self.tool.parameters_schema.clone()).map_err(|e| {
            NonControlToolError::Internal {
                message: format!("Failed to deserialize schema: {e}"),
            }
            .into()
        })
    }

    fn timeout(&self) -> Duration {
        self.tool.timeout
    }
}

#[async_trait]
impl TaskTool for TypeScriptToolInstance {
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        side_info: <Self as ToolMetadata>::SideInfo,
        ctx: &ToolContext,
    ) -> ToolResult<<Self as ToolMetadata>::Output> {
        let pool = ctx.js_runtime_pool().await?;
        let task_id = ctx.task_id().await;
        let episode_id = ctx.episode_id().await;

        pool.execute(
            self.tool.js_code.clone(),
            llm_params,
            side_info,
            ctx.clone(),
            task_id,
            episode_id,
        )
        .await
        .map_err(|e| {
            NonControlToolError::Internal {
                message: e.to_string(),
            }
            .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creates_tool() {
        let tool = TypeScriptTool::builder("test_tool")
            .description("A test tool")
            .typescript_code(
                r#"
                export default {
                    name: "test_tool",
                    async run(params, sideInfo) {
                        return { result: "ok" };
                    }
                };
            "#,
            )
            .parameters_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                }
            }))
            .build()
            .expect("Failed to build tool");

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "A test tool");
        assert!(!tool.js_code.is_empty(), "JS code should be transpiled");
    }

    #[test]
    fn test_builder_requires_typescript_code() {
        let result = TypeScriptTool::builder("test_tool").build();

        assert!(result.is_err());
        match result {
            Err(TypeScriptToolError::InvalidTool(msg)) => {
                assert!(msg.contains("typescript_code"));
            }
            _ => panic!("Expected InvalidTool error"),
        }
    }

    #[test]
    fn test_builder_default_schema() {
        let tool = TypeScriptTool::builder("test_tool")
            .typescript_code("export default { async run() { return {}; } };")
            .build()
            .expect("Failed to build tool");

        // Default schema should be an empty object type
        assert_eq!(
            tool.parameters_schema,
            serde_json::json!({
                "type": "object",
                "properties": {},
            })
        );
    }

    #[test]
    fn test_builder_custom_timeout() {
        let tool = TypeScriptTool::builder("test_tool")
            .typescript_code("export default { async run() { return {}; } };")
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build tool");

        assert_eq!(tool.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_instance_getters() {
        let tool = TypeScriptTool::builder("my_tool")
            .description("My description")
            .typescript_code("export default { async run() { return {}; } };")
            .parameters_schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "foo": { "type": "string" }
                }
            }))
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build tool");

        let instance = TypeScriptToolInstance::new(tool);

        assert_eq!(instance.tool_name(), "my_tool");
        assert_eq!(instance.tool_description(), "My description");
        assert_eq!(instance.timeout(), Duration::from_secs(30));
    }
}
