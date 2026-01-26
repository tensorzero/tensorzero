//! Unit tests for durable-tools (no Postgres required).
//!
//! Integration tests that require Postgres are in `tests/integration.rs`.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ToolContext;
use crate::context::SimpleToolContext;
use crate::error::{NonControlToolError, ToolError, ToolResult};
use crate::executor::ToolExecutorBuilder;
use crate::registry::{ErasedTaskToolWrapper, ErasedTool, ToolRegistry};
use crate::simple_tool::SimpleTool;
use crate::task_tool::TaskTool;
use crate::tool_metadata::ToolMetadata;

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

impl ToolMetadata for EchoSimpleTool {
    type SideInfo = ();
    type Output = EchoOutput;
    type LlmParams = EchoParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("echo_simple")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Echoes the input message")
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(10)
    }
}

#[async_trait]
impl SimpleTool for EchoSimpleTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
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
#[derive(Default)]
struct EchoTaskTool;

impl ToolMetadata for EchoTaskTool {
    type SideInfo = ();
    type Output = EchoOutput;
    type LlmParams = EchoParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("echo_task")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Echoes the input message (durable)")
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(60)
    }
}

#[async_trait]
impl TaskTool for EchoTaskTool {
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        Ok(EchoOutput {
            echoed: llm_params.message,
        })
    }
}

/// Another `TaskTool` with different timeout for testing defaults.
#[derive(Default)]
struct DefaultTimeoutTaskTool;

impl ToolMetadata for DefaultTimeoutTaskTool {
    type SideInfo = ();
    type Output = EchoOutput;
    type LlmParams = EchoParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("default_timeout_task")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Uses default timeout")
    }
    // Uses default timeout (60 seconds from ToolMetadata)
}

#[async_trait]
impl TaskTool for DefaultTimeoutTaskTool {
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
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

impl ToolMetadata for DefaultTimeoutSimpleTool {
    type SideInfo = ();
    type Output = EchoOutput;
    type LlmParams = EchoParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("default_timeout_simple")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Uses default timeout")
    }
    // Uses default timeout (60 seconds from ToolMetadata)
}

#[async_trait]
impl SimpleTool for DefaultTimeoutSimpleTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
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
        registry.register_task_tool_instance(EchoTaskTool).unwrap();

        assert!(registry.get("echo_task").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn register_simple_tool_adds_to_both_maps() {
        let mut registry = ToolRegistry::new();
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        // Should be in both tools and simple_tools
        assert!(registry.get("echo_simple").is_some());
        assert!(registry.get_simple_tool("echo_simple").is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn get_returns_registered_tool() {
        let mut registry = ToolRegistry::new();
        registry
            .register_task_tool_instance::<EchoTaskTool>(EchoTaskTool)
            .unwrap();

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
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        let tool = registry.get_simple_tool("echo_simple");
        assert!(tool.is_some());
    }

    #[test]
    fn get_simple_tool_returns_none_for_task_tool() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool_instance(EchoTaskTool).unwrap();

        // TaskTools are not in simple_tools map
        assert!(registry.get_simple_tool("echo_task").is_none());
    }

    #[test]
    fn is_durable_returns_true_for_task_tool() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool_instance(EchoTaskTool).unwrap();

        assert_eq!(registry.is_durable("echo_task"), Some(true));
    }

    #[test]
    fn is_durable_returns_false_for_simple_tool() {
        let mut registry = ToolRegistry::new();
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

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
        registry.register_task_tool_instance(EchoTaskTool).unwrap();
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        let tools = registry.list_tools();
        assert_eq!(tools.len(), 2);
        assert!(tools.contains(&"echo_task"));
        assert!(tools.contains(&"echo_simple"));
    }

    #[test]
    fn list_task_tools_filters_to_durable_only() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool_instance(EchoTaskTool).unwrap();
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        let task_tools = registry.list_task_tools();
        assert_eq!(task_tools.len(), 1);
        assert!(task_tools.contains(&"echo_task"));
    }

    #[test]
    fn list_simple_tools_filters_to_non_durable_only() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool_instance(EchoTaskTool).unwrap();
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        let simple_tools = registry.list_simple_tools();
        assert_eq!(simple_tools.len(), 1);
        assert!(simple_tools.contains(&"echo_simple"));
    }

    #[test]
    fn iter_and_try_from_generates_correct_structure() {
        use tensorzero::Tool;

        let mut registry = ToolRegistry::new();
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        let tools: Vec<Tool> = registry
            .iter()
            .map(Tool::try_from)
            .collect::<Result<_, _>>()
            .unwrap();
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

        registry.register_task_tool_instance(EchoTaskTool).unwrap();

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn register_task_tool_errors_on_duplicate() {
        let mut registry = ToolRegistry::new();
        registry.register_task_tool_instance(EchoTaskTool).unwrap();

        // Second registration should fail
        let result = registry.register_task_tool_instance(EchoTaskTool);
        match result {
            Err(ToolError::NonControl(NonControlToolError::DuplicateToolName { name })) => {
                assert_eq!(name, "echo_task");
            }
            _ => panic!("Expected DuplicateToolName error"),
        }

        // Registry should still have only one tool
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn register_simple_tool_errors_on_duplicate() {
        let mut registry = ToolRegistry::new();
        registry
            .register_simple_tool_instance(EchoSimpleTool)
            .unwrap();

        // Second registration should fail
        let result = registry.register_simple_tool_instance(EchoSimpleTool);
        match result {
            Err(ToolError::NonControl(NonControlToolError::DuplicateToolName { name })) => {
                assert_eq!(name, "echo_simple");
            }
            _ => panic!("Expected DuplicateToolName error"),
        }

        // Registry should still have only one tool
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn register_tools_with_same_name_errors() {
        use std::borrow::Cow;

        // Create a SimpleTool with the same name as EchoTaskTool
        #[derive(Default)]
        struct ConflictingSimpleTool;

        impl ToolMetadata for ConflictingSimpleTool {
            type SideInfo = ();
            type Output = EchoOutput;
            type LlmParams = EchoParams;

            fn name(&self) -> Cow<'static, str> {
                Cow::Borrowed("echo_task") // Same name as EchoTaskTool
            }

            fn description(&self) -> Cow<'static, str> {
                Cow::Borrowed("Conflicting tool")
            }
        }

        #[async_trait::async_trait]
        impl SimpleTool for ConflictingSimpleTool {
            async fn execute(
                llm_params: <Self as ToolMetadata>::LlmParams,
                _side_info: Self::SideInfo,
                _ctx: SimpleToolContext<'_>,
                _idempotency_key: &str,
            ) -> ToolResult<Self::Output> {
                Ok(EchoOutput {
                    echoed: llm_params.message,
                })
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register_task_tool_instance(EchoTaskTool).unwrap();

        // Registering a SimpleTool with the same name should fail
        let result = registry.register_simple_tool_instance(ConflictingSimpleTool);
        match result {
            Err(ToolError::NonControl(NonControlToolError::DuplicateToolName { name })) => {
                assert_eq!(name, "echo_task");
            }
            _ => panic!("Expected DuplicateToolName error"),
        }

        // Registry should still have only one tool
        assert_eq!(registry.len(), 1);
    }
}

// ============================================================================
// Type Erasure Tests
// ============================================================================

mod erasure_tests {
    use super::*;

    #[test]
    fn erased_task_tool_wrapper_exposes_name() {
        let wrapper = ErasedTaskToolWrapper::new(Arc::new(EchoTaskTool));
        assert_eq!(wrapper.name(), "echo_task");
    }

    #[test]
    fn erased_task_tool_wrapper_exposes_description() {
        let wrapper = ErasedTaskToolWrapper::new(Arc::new(EchoTaskTool));
        assert_eq!(
            wrapper.description().as_ref(),
            "Echoes the input message (durable)"
        );
    }

    #[test]
    fn erased_task_tool_wrapper_exposes_timeout() {
        let wrapper = ErasedTaskToolWrapper::new(Arc::new(EchoTaskTool));
        assert_eq!(wrapper.timeout(), Duration::from_secs(60));
    }

    #[test]
    fn erased_task_tool_wrapper_default_timeout() {
        let wrapper = ErasedTaskToolWrapper::new(Arc::new(DefaultTimeoutTaskTool));
        assert_eq!(wrapper.timeout(), Duration::from_secs(60));
    }

    #[test]
    fn erased_task_tool_wrapper_is_durable_true() {
        let wrapper = ErasedTaskToolWrapper::new(Arc::new(EchoTaskTool));
        assert!(wrapper.is_durable());
    }

    #[test]
    fn erased_simple_tool_exposes_metadata() {
        let tool = EchoSimpleTool;
        assert_eq!(ToolMetadata::name(&tool), "echo_simple");
        assert_eq!(
            ToolMetadata::description(&tool).as_ref(),
            "Echoes the input message"
        );
        assert_eq!(ToolMetadata::timeout(&tool), Duration::from_secs(10));
    }

    #[test]
    fn erased_simple_tool_default_timeout() {
        let tool = DefaultTimeoutSimpleTool;
        assert_eq!(ToolMetadata::timeout(&tool), Duration::from_secs(60));
    }

    #[test]
    fn erased_simple_tool_is_durable_false() {
        let tool = EchoSimpleTool;
        assert!(!ErasedTool::is_durable(&tool));
    }

    #[test]
    fn erased_task_tool_wrapper_parameters_schema_has_message_field() {
        let wrapper = ErasedTaskToolWrapper::new(Arc::new(EchoTaskTool));
        let schema = wrapper.parameters_schema().unwrap();

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
            ToolError::NonControl(NonControlToolError::Internal { message }) => {
                assert_eq!(message, "test error");
            }
            _ => panic!("Expected NonControl(Internal)"),
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
    fn task_error_from_tool_error_user() {
        let tool_err = ToolError::NonControl(NonControlToolError::User {
            message: "test error".to_string(),
            error_data: serde_json::json!({"kind": "TestError"}),
        });
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::User { message, .. } => {
                assert_eq!(message, "test error");
            }
            _ => panic!("Expected User"),
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
        let tool_err = ToolError::NonControl(NonControlToolError::ToolNotFound {
            name: "missing_tool".to_string(),
        });
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::User { message, .. } => {
                assert!(message.contains("missing_tool"));
            }
            _ => panic!("Expected User"),
        }
    }

    #[test]
    fn task_error_from_tool_error_invalid_params() {
        let tool_err = ToolError::NonControl(NonControlToolError::InvalidParams {
            message: "bad params".to_string(),
        });
        let task_err: TaskError = tool_err.into();

        match task_err {
            TaskError::User { message, .. } => {
                assert!(message.contains("bad params"));
            }
            _ => panic!("Expected User"),
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
