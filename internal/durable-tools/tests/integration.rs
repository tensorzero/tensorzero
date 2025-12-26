//! Integration tests for durable-tools (requires Postgres).
//!
//! These tests use `#[sqlx::test]` with the durable migrator to automatically
//! set up the database schema before each test.
//!
//! Run with: cargo test --test integration

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use durable::MIGRATOR;
use durable::WorkerOptions;
use durable_tools::{
    ErasedSimpleTool, InferenceClient, InferenceError, SimpleTool, SimpleToolContext, TaskTool,
    ToolContext, ToolExecutor, ToolMetadata, ToolResult,
};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tensorzero::{
    ClientInferenceParams, InferenceResponse, Input, InputMessage, InputMessageContent, Role, Tool,
    Usage,
};
use tensorzero_core::endpoints::inference::ChatInferenceResponse;
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};
use tokio::sync::Mutex;
use uuid::Uuid;

// ============================================================================
// Mock Inference Client
// ============================================================================

/// A mock inference client that returns configurable responses.
struct MockInferenceClient {
    response: Option<InferenceResponse>,
}

impl MockInferenceClient {
    /// Create a mock that returns an error (for tests that don't use inference).
    fn error_on_call() -> Self {
        Self { response: None }
    }

    /// Create a mock that returns the given response.
    fn with_response(response: InferenceResponse) -> Self {
        Self {
            response: Some(response),
        }
    }
}

#[async_trait]
impl InferenceClient for MockInferenceClient {
    async fn inference(
        &self,
        _params: ClientInferenceParams,
    ) -> Result<InferenceResponse, InferenceError> {
        self.response
            .clone()
            .ok_or(InferenceError::StreamingNotSupported)
    }

    async fn create_autopilot_event(
        &self,
        _session_id: Uuid,
        _request: durable_tools::CreateEventRequest,
    ) -> Result<durable_tools::CreateEventResponse, InferenceError> {
        Err(InferenceError::AutopilotUnavailable)
    }

    async fn list_autopilot_events(
        &self,
        _session_id: Uuid,
        _params: durable_tools::ListEventsParams,
    ) -> Result<durable_tools::ListEventsResponse, InferenceError> {
        Err(InferenceError::AutopilotUnavailable)
    }

    async fn list_autopilot_sessions(
        &self,
        _params: durable_tools::ListSessionsParams,
    ) -> Result<durable_tools::ListSessionsResponse, InferenceError> {
        Err(InferenceError::AutopilotUnavailable)
    }
}

/// Create a mock chat inference response with the given text content.
fn create_mock_chat_response(text: &str) -> InferenceResponse {
    InferenceResponse::Chat(ChatInferenceResponse {
        inference_id: Uuid::now_v7(),
        episode_id: Uuid::now_v7(),
        variant_name: "test_variant".to_string(),
        content: vec![ContentBlockChatOutput::Text(Text {
            text: text.to_string(),
        })],
        usage: Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
        },
        original_response: None,
        finish_reason: None,
    })
}

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
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("echo_simple")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Echoes the input message")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(EchoParams))
    }

    type LlmParams = EchoParams;

    fn timeout() -> Duration {
        Duration::from_secs(10)
    }
    type SideInfo = ();
    type Output = EchoOutput;
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
struct EchoTaskTool;

impl ToolMetadata for EchoTaskTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("echo_task")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Echoes the input message (durable)")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(EchoParams))
    }

    type LlmParams = EchoParams;

    fn timeout() -> Duration {
        Duration::from_secs(60)
    }
    type SideInfo = ();
    type Output = EchoOutput;
}

#[async_trait]
impl TaskTool for EchoTaskTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        Ok(EchoOutput {
            echoed: llm_params.message,
        })
    }
}

// ============================================================================
// Inference Test Fixtures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct InferencePromptParams {
    prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct InferenceToolOutput {
    response: String,
}

/// Helper to extract text content from an `InferenceResponse`.
fn extract_text_from_response(response: &InferenceResponse) -> String {
    match response {
        InferenceResponse::Chat(chat_response) => {
            for content_block in &chat_response.content {
                if let ContentBlockChatOutput::Text(text) = content_block {
                    return text.text.clone();
                }
            }
            String::new()
        }
        InferenceResponse::Json(_) => String::new(),
    }
}

/// A `SimpleTool` that calls inference and returns the response text.
struct InferenceSimpleTool;

impl ToolMetadata for InferenceSimpleTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("inference_simple")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Calls inference and returns the response")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(InferencePromptParams))
    }

    type LlmParams = InferencePromptParams;

    fn timeout() -> Duration {
        Duration::from_secs(30)
    }
    type SideInfo = ();
    type Output = InferenceToolOutput;
}

#[async_trait]
impl SimpleTool for InferenceSimpleTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        let input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: llm_params.prompt,
                })],
            }],
        };

        let inference_params = ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            input,
            ..Default::default()
        };

        let response = ctx
            .inference(inference_params)
            .await
            .map_err(|e| anyhow::anyhow!("Inference failed: {e}"))?;
        let text = extract_text_from_response(&response);

        Ok(InferenceToolOutput { response: text })
    }
}

/// A `TaskTool` that calls inference and returns the response text.
struct InferenceTaskTool;

impl ToolMetadata for InferenceTaskTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("inference_task")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Calls inference (durable) and returns the response")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(InferencePromptParams))
    }

    type LlmParams = InferencePromptParams;

    fn timeout() -> Duration {
        Duration::from_secs(60)
    }
    type SideInfo = ();
    type Output = InferenceToolOutput;
}

#[async_trait]
impl TaskTool for InferenceTaskTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        let input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: llm_params.prompt,
                })],
            }],
        };

        let inference_params = ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            input,
            ..Default::default()
        };

        let response = ctx.inference(inference_params).await?;
        let text = extract_text_from_response(&response);

        Ok(InferenceToolOutput { response: text })
    }
}

// ============================================================================
// Execute Erased Tests
// ============================================================================

#[sqlx::test(migrator = "MIGRATOR")]
async fn execute_erased_deserializes_and_serializes_correctly(pool: PgPool) -> sqlx::Result<()> {
    let tool = EchoSimpleTool;

    let inference_client: Arc<dyn InferenceClient> = Arc::new(MockInferenceClient::error_on_call());
    let ctx = SimpleToolContext::new(&pool, &inference_client);
    let llm_params = serde_json::json!({"message": "hello"});
    // Unit type () deserializes from null, not {}
    let side_info = serde_json::json!(null);

    let result = tool
        .execute_erased(llm_params, side_info, ctx, "test-key")
        .await
        .unwrap();

    assert_eq!(result["echoed"], "hello");
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn execute_erased_returns_error_on_invalid_params(pool: PgPool) -> sqlx::Result<()> {
    let tool = EchoSimpleTool;

    let inference_client: Arc<dyn InferenceClient> = Arc::new(MockInferenceClient::error_on_call());
    let ctx = SimpleToolContext::new(&pool, &inference_client);
    // Missing required "message" field
    let llm_params = serde_json::json!({"wrong_field": "hello"});
    let side_info = serde_json::json!(null);

    let result = tool
        .execute_erased(llm_params, side_info, ctx, "test-key")
        .await;

    assert!(result.is_err());
    Ok(())
}

// ============================================================================
// ToolExecutor Integration Tests
// ============================================================================

#[sqlx::test(migrator = "MIGRATOR")]
async fn tool_executor_registers_and_lists_tools(pool: PgPool) -> sqlx::Result<()> {
    let inference_client: Arc<dyn InferenceClient> = Arc::new(MockInferenceClient::error_on_call());
    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(format!("test_queue_{}", Uuid::now_v7()))
        .inference_client(inference_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .register_simple_tool::<EchoSimpleTool>()
        .await
        .unwrap();
    executor.register_task_tool::<EchoTaskTool>().await.unwrap();

    let definitions = executor.tool_definitions().await.unwrap();
    assert_eq!(definitions.len(), 2);

    let names: Vec<&str> = definitions
        .iter()
        .map(|d| match d {
            Tool::Function(f) => f.name.as_str(),
            Tool::OpenAICustom(_) => panic!("Expected function tool"),
        })
        .collect();

    assert!(names.contains(&"echo_simple"));
    assert!(names.contains(&"echo_task"));
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn tool_executor_spawns_task_tool(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());
    let inference_client: Arc<dyn InferenceClient> = Arc::new(MockInferenceClient::error_on_call());

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .inference_client(inference_client)
        .build()
        .await
        .expect("Failed to build executor");

    // Create the queue before spawning
    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor.register_task_tool::<EchoTaskTool>().await.unwrap();

    let episode_id = Uuid::now_v7();
    let result = executor
        .spawn_tool::<EchoTaskTool>(
            EchoParams {
                message: "test message".to_string(),
            },
            (), // No side info
            episode_id,
        )
        .await;

    assert!(result.is_ok(), "spawn_tool failed: {:?}", result.err());
    let spawn_result = result.unwrap();
    assert!(!spawn_result.task_id.is_nil());
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn spawn_tool_by_name_works(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());
    let inference_client: Arc<dyn InferenceClient> = Arc::new(MockInferenceClient::error_on_call());

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .inference_client(inference_client)
        .build()
        .await
        .expect("Failed to build executor");

    // Create the queue before spawning
    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor.register_task_tool::<EchoTaskTool>().await.unwrap();

    let episode_id = Uuid::now_v7();
    let result = executor
        .spawn_tool_by_name(
            "echo_task",
            serde_json::json!({"message": "dynamic call"}),
            episode_id,
        )
        .await;

    assert!(
        result.is_ok(),
        "spawn_tool_by_name failed: {:?}",
        result.err()
    );
    let spawn_result = result.unwrap();
    assert!(!spawn_result.task_id.is_nil());
    Ok(())
}

// ============================================================================
// Idempotency Key Uniqueness Tests
// ============================================================================

/// Static storage for captured idempotency keys during testing.
static CAPTURED_KEYS: std::sync::LazyLock<Arc<Mutex<Vec<String>>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));

/// A `SimpleTool` that captures its idempotency key for testing.
#[derive(Default)]
struct KeyCapturingSimpleTool;

impl ToolMetadata for KeyCapturingSimpleTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("key_capturing_tool")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Captures idempotency keys for testing")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(EchoParams))
    }

    type LlmParams = EchoParams;
    type SideInfo = ();
    type Output = EchoOutput;
}

#[async_trait]
impl SimpleTool for KeyCapturingSimpleTool {
    async fn execute(
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        // Capture the idempotency key
        CAPTURED_KEYS.lock().await.push(idempotency_key.to_string());

        Ok(EchoOutput {
            echoed: llm_params.message,
        })
    }
}

/// A `TaskTool` that calls a `SimpleTool` multiple times.
struct MultiCallTaskTool;

impl ToolMetadata for MultiCallTaskTool {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("multi_call_task")
    }

    fn description() -> Cow<'static, str> {
        Cow::Borrowed("Calls a SimpleTool multiple times")
    }

    fn parameters_schema() -> ToolResult<Schema> {
        Ok(schema_for!(EchoParams))
    }

    type LlmParams = EchoParams;
    type SideInfo = ();
    type Output = EchoOutput;
}

#[async_trait]
impl TaskTool for MultiCallTaskTool {
    async fn execute(
        _llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        // Call the same SimpleTool three times with different params
        ctx.call_tool(
            "key_capturing_tool",
            serde_json::json!({"message": "first"}),
        )
        .await?;

        ctx.call_tool(
            "key_capturing_tool",
            serde_json::json!({"message": "second"}),
        )
        .await?;

        ctx.call_tool(
            "key_capturing_tool",
            serde_json::json!({"message": "third"}),
        )
        .await?;

        Ok(EchoOutput {
            echoed: "done".to_string(),
        })
    }
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn calling_same_tool_multiple_times_generates_unique_idempotency_keys(
    pool: PgPool,
) -> sqlx::Result<()> {
    // Clear any previously captured keys
    CAPTURED_KEYS.lock().await.clear();

    let queue_name = format!("test_queue_{}", Uuid::now_v7());
    let inference_client: Arc<dyn InferenceClient> = Arc::new(MockInferenceClient::error_on_call());

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .inference_client(inference_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    // Register both tools
    executor
        .register_simple_tool::<KeyCapturingSimpleTool>()
        .await
        .unwrap();
    executor
        .register_task_tool::<MultiCallTaskTool>()
        .await
        .unwrap();

    // Spawn the task
    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool::<MultiCallTaskTool>(
            EchoParams {
                message: "test".to_string(),
            },
            (),
            episode_id,
        )
        .await
        .expect("Failed to spawn task");

    // Start a worker to execute the task
    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for task to complete
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Shutdown worker
    worker.shutdown().await;

    // Verify we captured 3 idempotency keys
    let keys = CAPTURED_KEYS.lock().await;
    assert_eq!(
        keys.len(),
        3,
        "Expected 3 captured keys, got {}",
        keys.len()
    );

    // Verify all keys are unique
    let unique_keys: std::collections::HashSet<_> = keys.iter().collect();
    assert_eq!(
        unique_keys.len(),
        3,
        "Expected 3 unique keys, but some were duplicates: {keys:?}"
    );

    Ok(())
}

// ============================================================================
// Inference Tests
// ============================================================================

#[sqlx::test(migrator = "MIGRATOR")]
async fn simple_tool_calls_inference_successfully(pool: PgPool) -> sqlx::Result<()> {
    let mock_response = create_mock_chat_response("Hello from TensorZero!");
    let inference_client: Arc<dyn InferenceClient> =
        Arc::new(MockInferenceClient::with_response(mock_response));

    let tool = InferenceSimpleTool;
    let ctx = SimpleToolContext::new(&pool, &inference_client);
    let llm_params = serde_json::json!({"prompt": "Say hello"});
    let side_info = serde_json::json!(null);

    let result = tool
        .execute_erased(llm_params, side_info, ctx, "test-inference-key")
        .await
        .expect("SimpleTool inference call should succeed");

    assert_eq!(result["response"], "Hello from TensorZero!");
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn simple_tool_propagates_inference_error(pool: PgPool) -> sqlx::Result<()> {
    // Mock returns error when response is None
    let inference_client: Arc<dyn InferenceClient> = Arc::new(MockInferenceClient::error_on_call());

    let tool = InferenceSimpleTool;
    let ctx = SimpleToolContext::new(&pool, &inference_client);
    let llm_params = serde_json::json!({"prompt": "This will fail"});
    let side_info = serde_json::json!(null);

    let result = tool
        .execute_erased(llm_params, side_info, ctx, "test-error-key")
        .await;

    assert!(result.is_err(), "Expected inference error to propagate");
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn task_tool_with_inference_can_be_registered(pool: PgPool) -> sqlx::Result<()> {
    let mock_response = create_mock_chat_response("Response from TaskTool!");
    let inference_client: Arc<dyn InferenceClient> =
        Arc::new(MockInferenceClient::with_response(mock_response));

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(format!("test_queue_{}", Uuid::now_v7()))
        .inference_client(inference_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .register_task_tool::<InferenceTaskTool>()
        .await
        .unwrap();

    let definitions = executor.tool_definitions().await.unwrap();
    let names: Vec<&str> = definitions
        .iter()
        .filter_map(|d| match d {
            Tool::Function(f) => Some(f.name.as_str()),
            Tool::OpenAICustom(_) => None,
        })
        .collect();

    assert!(
        names.contains(&"inference_task"),
        "InferenceTaskTool should be registered"
    );
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn task_tool_with_inference_can_be_spawned(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());
    let mock_response = create_mock_chat_response("Spawned response!");
    let inference_client: Arc<dyn InferenceClient> =
        Arc::new(MockInferenceClient::with_response(mock_response));

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .inference_client(inference_client)
        .build()
        .await
        .expect("Failed to build executor");

    // Create the queue before spawning
    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor
        .register_task_tool::<InferenceTaskTool>()
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let result = executor
        .spawn_tool::<InferenceTaskTool>(
            InferencePromptParams {
                prompt: "Generate something".to_string(),
            },
            (), // No side info
            episode_id,
        )
        .await;

    assert!(
        result.is_ok(),
        "spawn_tool for InferenceTaskTool failed: {:?}",
        result.err()
    );
    let spawn_result = result.unwrap();
    assert!(!spawn_result.task_id.is_nil());
    Ok(())
}
