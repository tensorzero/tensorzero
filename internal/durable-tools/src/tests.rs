//! Unit tests for durable-tools (no Postgres required).
//!
//! Integration tests that require Postgres are in `tests/integration.rs`.

use std::borrow::Cow;
use std::time::Duration;

use async_trait::async_trait;
use schemars::{JsonSchema, schema::RootSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::ToolContext;
use crate::context::SimpleToolContext;
use crate::error::{ToolError, ToolResult};
use crate::executor::ToolExecutorBuilder;
use crate::registry::{ErasedTaskToolWrapper, ErasedTool, ToolRegistry};
use crate::simple_tool::SimpleTool;
use crate::task_tool::TaskTool;

// ============================================================================
// Test Fixtures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct EchoParams {
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct EchoOutput {
    echoed: String,
}

/// A simple `SimpleTool` for testing.
#[derive(Default)]
struct EchoSimpleTool;

#[async_trait]
impl SimpleTool for EchoSimpleTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("echo_simple")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Echoes the input message")
    }

    fn parameters_schema() -> RootSchema {
        schema_for!(EchoParams)
    }

    type LlmParams = EchoParams;
    type SideInfo = ();
    type Output = EchoOutput;

    fn timeout() -> Duration {
        Duration::from_secs(10)
    }

    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        Ok(EchoOutput {
            echoed: llm_params.message,
        })
    }
}

/// A simple `TaskTool` for testing.
struct EchoTaskTool;

#[async_trait]
impl TaskTool for EchoTaskTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("echo_task")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Echoes the input message (durable)")
    }

    fn parameters_schema() -> RootSchema {
        schema_for!(EchoParams)
    }

    type LlmParams = EchoParams;
    type SideInfo = ();
    type Output = EchoOutput;

    fn timeout() -> Duration {
        Duration::from_secs(60)
    }

    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        Ok(EchoOutput {
            echoed: llm_params.message,
        })
    }
}

/// Another `TaskTool` with different timeout for testing defaults.
struct DefaultTimeoutTaskTool;

#[async_trait]
impl TaskTool for DefaultTimeoutTaskTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("default_timeout_task")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Uses default timeout")
    }

    fn parameters_schema() -> RootSchema {
        schema_for!(EchoParams)
    }

    type LlmParams = EchoParams;
    type SideInfo = ();
    type Output = EchoOutput;

    // Uses default timeout (120 seconds)

    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        Ok(EchoOutput {
            echoed: llm_params.message,
        })
    }
}

/// Another `SimpleTool` with default timeout.
#[derive(Default)]
struct DefaultTimeoutSimpleTool;

#[async_trait]
impl SimpleTool for DefaultTimeoutSimpleTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("default_timeout_simple")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Uses default timeout")
    }

    fn parameters_schema() -> RootSchema {
        schema_for!(EchoParams)
    }

    type LlmParams = EchoParams;
    type SideInfo = ();
    type Output = EchoOutput;

    // Uses default timeout (30 seconds)

    async fn execute(
        llm_params: Self::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        Ok(EchoOutput {
            echoed: llm_params.message,
        })
    }
}

// ============================================================================
// ToolRegistry Tests
// ============================================================================

mod registry_tests {
    use super::*;

    #[test]
    fn register_task_tool_adds_to_tools_map() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool::<EchoTaskTool>();

        assert!(registry.get("echo_task").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn register_simple_tool_adds_to_both_maps() {
        let mut registry = ToolRegistry::new();
        registry.register_simple_tool::<EchoSimpleTool>();

        // Should be in both tools and simple_tools
        assert!(registry.get("echo_simple").is_some());
        assert!(registry.get_simple_tool("echo_simple").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn get_returns_registered_tool() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool::<EchoTaskTool>();

        let tool = registry.get("echo_task").unwrap();
        assert_eq!(tool.name(), "echo_task");
        assert_eq!(
            tool.description().as_ref(),
            "Echoes the input message (durable)"
        );
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn get_simple_tool_returns_simple_tool() {
        let mut registry = ToolRegistry::new();
        registry.register_simple_tool::<EchoSimpleTool>();

        let tool = registry.get_simple_tool("echo_simple");
        assert!(tool.is_some());
    }

    #[test]
    fn get_simple_tool_returns_none_for_task_tool() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool::<EchoTaskTool>();

        // TaskTools are not in simple_tools map
        assert!(registry.get_simple_tool("echo_task").is_none());
    }

    #[test]
    fn is_durable_returns_true_for_task_tool() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool::<EchoTaskTool>();

        assert_eq!(registry.is_durable("echo_task"), Some(true));
    }

    #[test]
    fn is_durable_returns_false_for_simple_tool() {
        let mut registry = ToolRegistry::new();
        registry.register_simple_tool::<EchoSimpleTool>();

        assert_eq!(registry.is_durable("echo_simple"), Some(false));
    }

    #[test]
    fn is_durable_returns_none_for_unknown() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.is_durable("nonexistent"), None);
    }

    #[test]
    fn list_tools_returns_all_names() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool::<EchoTaskTool>();
        registry.register_simple_tool::<EchoSimpleTool>();

        let tools = registry.list_tools();
        assert_eq!(tools.len(), 2);
        assert!(tools.contains(&"echo_task"));
        assert!(tools.contains(&"echo_simple"));
    }

    #[test]
    fn list_task_tools_filters_to_durable_only() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool::<EchoTaskTool>();
        registry.register_simple_tool::<EchoSimpleTool>();

        let task_tools = registry.list_task_tools();
        assert_eq!(task_tools.len(), 1);
        assert!(task_tools.contains(&"echo_task"));
    }

    #[test]
    fn list_simple_tools_filters_to_non_durable_only() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool::<EchoTaskTool>();
        registry.register_simple_tool::<EchoSimpleTool>();

        let simple_tools = registry.list_simple_tools();
        assert_eq!(simple_tools.len(), 1);
        assert!(simple_tools.contains(&"echo_simple"));
    }

    #[test]
    fn to_tensorzero_tools_generates_correct_structure() {
        use tensorzero::Tool;

        let mut registry = ToolRegistry::new();
        registry.register_simple_tool::<EchoSimpleTool>();

        let tools = registry.to_tensorzero_tools().unwrap();
        assert_eq!(tools.len(), 1);

        let tool = &tools[0];
        match tool {
            Tool::Function(func) => {
                assert_eq!(func.name, "echo_simple");
                assert_eq!(func.description, "Echoes the input message");
                assert!(func.parameters.is_object());
                assert!(!func.strict);
            }
            Tool::OpenAICustom(_) => panic!("Expected Function tool"),
        }
    }

    #[test]
    fn len_and_is_empty_reflect_state() {
        let mut registry = ToolRegistry::new();

        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register_task_tool::<EchoTaskTool>();

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        registry.register_simple_tool::<EchoSimpleTool>();

        assert_eq!(registry.len(), 2);
    }
}

// ============================================================================
// Type Erasure Tests
// ============================================================================

mod erasure_tests {
    use super::*;

    #[test]
    fn erased_task_tool_wrapper_exposes_name() {
        let wrapper = ErasedTaskToolWrapper::<EchoTaskTool>::new();
        assert_eq!(wrapper.name(), "echo_task");
    }

    #[test]
    fn erased_task_tool_wrapper_exposes_description() {
        let wrapper = ErasedTaskToolWrapper::<EchoTaskTool>::new();
        assert_eq!(
            wrapper.description().as_ref(),
            "Echoes the input message (durable)"
        );
    }

    #[test]
    fn erased_task_tool_wrapper_exposes_timeout() {
        let wrapper = ErasedTaskToolWrapper::<EchoTaskTool>::new();
        assert_eq!(wrapper.timeout(), Duration::from_secs(60));
    }

    #[test]
    fn erased_task_tool_wrapper_default_timeout() {
        let wrapper = ErasedTaskToolWrapper::<DefaultTimeoutTaskTool>::new();
        assert_eq!(wrapper.timeout(), Duration::from_secs(120));
    }

    #[test]
    fn erased_task_tool_wrapper_is_durable_true() {
        let wrapper = ErasedTaskToolWrapper::<EchoTaskTool>::new();
        assert!(wrapper.is_durable());
    }

    #[test]
    fn erased_simple_tool_exposes_metadata() {
        let tool = EchoSimpleTool;
        assert_eq!(tool.name(), "echo_simple");
        assert_eq!(tool.description().as_ref(), "Echoes the input message");
        assert_eq!(tool.timeout(), Duration::from_secs(10));
    }

    #[test]
    fn erased_simple_tool_default_timeout() {
        let tool = DefaultTimeoutSimpleTool;
        assert_eq!(tool.timeout(), Duration::from_secs(30));
    }

    #[test]
    fn erased_simple_tool_is_durable_false() {
        let tool = EchoSimpleTool;
        assert!(!tool.is_durable());
    }

    #[test]
    fn erased_task_tool_wrapper_parameters_schema_has_message_field() {
        let wrapper = ErasedTaskToolWrapper::<EchoTaskTool>::new();
        let schema = wrapper.parameters_schema();

        // The schema should be an object with a "message" property
        let schema_json = serde_json::to_value(&schema).unwrap();
        assert_eq!(schema_json["type"], "object");
        assert!(schema_json["properties"]["message"].is_object());
    }

    // Note: execute_erased tests require a real PgPool and are in tests/integration.rs
}

// ============================================================================
// ToolExecutorBuilder Tests
// ============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn builder_default_queue_name_is_tools() {
        let builder = ToolExecutorBuilder::new();
        // We can't directly inspect the builder, but we can verify behavior
        // by checking the default is used when not overridden.
        // For now, just verify the builder can be created.
        let _ = builder;
    }

    #[test]
    fn builder_methods_return_self_for_chaining() {
        let builder = ToolExecutorBuilder::new()
            .queue_name("custom_queue")
            .default_max_attempts(10);

        // Verify chaining works - if it compiles, it works
        let _ = builder;
    }

    #[test]
    fn builder_accepts_database_url() {
        let builder = ToolExecutorBuilder::new().database_url("postgres://localhost/test".into());

        let _ = builder;
    }
}

// ============================================================================
// Error Conversion Tests
// ============================================================================

mod error_tests {
    use super::*;
    use durable::{ControlFlow, TaskError};

    #[test]
    fn tool_error_from_task_error_task_internal() {
        let task_err = TaskError::TaskInternal(anyhow::anyhow!("test error"));
        let tool_err: ToolError = task_err.into();

        match tool_err {
            ToolError::ExecutionFailed(e) => {
                assert_eq!(e.to_string(), "test error");
            }
            _ => panic!("Expected ExecutionFailed"),
        }
    }

    #[test]
    fn tool_error_from_task_error_control_flow_suspend() {
        let task_err = TaskError::Control(ControlFlow::Suspend);
        let tool_err: ToolError = task_err.into();

        match tool_err {
            ToolError::Control(ControlFlow::Suspend) => {}
            _ => panic!("Expected Control(Suspend)"),
        }
    }

    #[test]
    fn tool_error_from_task_error_control_flow_cancelled() {
        let task_err = TaskError::Control(ControlFlow::Cancelled);
        let tool_err: ToolError = task_err.into();

        match tool_err {
            ToolError::Control(ControlFlow::Cancelled) => {}
            _ => panic!("Expected Control(Cancelled)"),
        }
    }

    #[test]
    fn task_error_from_tool_error_execution_failed() {
        let tool_err = ToolError::ExecutionFailed(anyhow::anyhow!("test error"));
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::TaskInternal(e) => {
                assert_eq!(e.to_string(), "test error");
            }
            _ => panic!("Expected TaskInternal"),
        }
    }

    #[test]
    fn task_error_from_tool_error_control_flow() {
        let tool_err = ToolError::Control(ControlFlow::Suspend);
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::Control(ControlFlow::Suspend) => {}
            _ => panic!("Expected Control(Suspend)"),
        }
    }

    #[test]
    fn task_error_from_tool_error_tool_not_found() {
        let tool_err = ToolError::ToolNotFound("missing_tool".to_string());
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::TaskInternal(e) => {
                assert!(e.to_string().contains("missing_tool"));
            }
            _ => panic!("Expected TaskInternal"),
        }
    }

    #[test]
    fn task_error_from_tool_error_invalid_params() {
        let tool_err = ToolError::InvalidParams("bad params".to_string());
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::TaskInternal(e) => {
                assert!(e.to_string().contains("bad params"));
            }
            _ => panic!("Expected TaskInternal"),
        }
    }

    #[test]
    fn task_error_from_tool_error_serialization() {
        let json_err = serde_json::from_str::<String>("not valid json").unwrap_err();
        let tool_err = ToolError::Serialization(json_err);
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::Serialization(_) => {}
            _ => panic!("Expected Serialization"),
        }
    }
}
