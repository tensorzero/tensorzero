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
use durable::SpawnOptions;
use durable::WorkerOptions;
use durable_tools::{
    ErasedSimpleTool, MockTensorZeroClient, NonControlToolError, SimpleTool, SimpleToolContext,
    TaskTool, TensorZeroClient, TensorZeroClientError, ToolContext, ToolExecutor, ToolMetadata,
    ToolResult,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::{AssertSqlSafe, PgPool};
use tensorzero::{
    ClientInferenceParams, InferenceResponse, Input, InputMessage, InputMessageContent, Role, Tool,
    Usage,
};
use tensorzero_core::endpoints::inference::ChatInferenceResponse;
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text};
use tokio::sync::Mutex;
use uuid::Uuid;

// ============================================================================
// Mock TensorZero Client Helpers
// ============================================================================

/// Create a mock that returns an error for inference calls.
fn mock_client_error_on_call() -> MockTensorZeroClient {
    let mut mock = MockTensorZeroClient::new();
    mock.expect_inference()
        .returning(|_| Box::pin(async { Err(TensorZeroClientError::StreamingNotSupported) }));
    mock
}

/// Create a mock that returns the given response for inference/action calls.
fn mock_client_with_response(response: InferenceResponse) -> MockTensorZeroClient {
    let mut mock = MockTensorZeroClient::new();
    let response_clone = response.clone();
    mock.expect_inference().returning(move |_| {
        let r = response.clone();
        Box::pin(async move { Ok(r) })
    });
    mock.expect_action().returning(move |_, _| {
        let r = response_clone.clone();
        Box::pin(async move { Ok(durable_tools::ActionResponse::Inference(r)) })
    });
    mock
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
        raw_usage: None,
        original_response: None,
        raw_response: None,
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
#[derive(Default)]
struct InferenceSimpleTool;

impl ToolMetadata for InferenceSimpleTool {
    type SideInfo = ();
    type Output = InferenceToolOutput;
    type LlmParams = InferencePromptParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("inference_simple")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Calls inference and returns the response")
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(30)
    }
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

        let response = ctx.inference(inference_params).await.map_err(|e| {
            NonControlToolError::User {
                message: format!("Inference failed: {e}"),
                error_data: serde_json::json!({"kind": "InferenceError", "message": e.to_string()}),
            }
        })?;
        let text = extract_text_from_response(&response);

        Ok(InferenceToolOutput { response: text })
    }
}

/// A `TaskTool` that calls inference and returns the response text.
#[derive(Default)]
struct InferenceTaskTool;

impl ToolMetadata for InferenceTaskTool {
    type SideInfo = ();
    type Output = InferenceToolOutput;
    type LlmParams = InferencePromptParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("inference_task")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Calls inference (durable) and returns the response")
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(60)
    }
}

#[async_trait]
impl TaskTool for InferenceTaskTool {
    async fn execute(
        &self,
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

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_error_on_call());
    let ctx = SimpleToolContext::new(&pool, &t0_client);
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

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_error_on_call());
    let ctx = SimpleToolContext::new(&pool, &t0_client);
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
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_error_on_call());
    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(format!("test_queue_{}", Uuid::now_v7()))
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .register_simple_tool_instance(EchoSimpleTool)
        .await
        .unwrap();
    executor
        .register_task_tool_instance(EchoTaskTool)
        .await
        .unwrap();

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
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_error_on_call());

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
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
        .register_task_tool_instance(EchoTaskTool)
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let result = executor
        .spawn_tool_by_name(
            "echo_task",
            serde_json::json!({"message": "test message"}),
            serde_json::json!(null),
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

#[sqlx::test(migrator = "MIGRATOR")]
async fn spawn_tool_by_name_works(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_error_on_call());

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
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
        .register_task_tool_instance(EchoTaskTool)
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let result = executor
        .spawn_tool_by_name(
            "echo_task",
            serde_json::json!({"message": "dynamic call"}),
            serde_json::json!(null),
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
    type SideInfo = ();
    type Output = EchoOutput;
    type LlmParams = EchoParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("key_capturing_tool")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Captures idempotency keys for testing")
    }
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
#[derive(Default)]
struct MultiCallTaskTool;

impl ToolMetadata for MultiCallTaskTool {
    type SideInfo = ();
    type Output = EchoOutput;
    type LlmParams = EchoParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("multi_call_task")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Calls a SimpleTool multiple times")
    }
}

#[async_trait]
impl TaskTool for MultiCallTaskTool {
    async fn execute(
        &self,
        _llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> ToolResult<Self::Output> {
        // Call the same SimpleTool three times with different params
        ctx.call_tool(
            "key_capturing_tool",
            serde_json::json!({"message": "first"}),
            serde_json::json!(null),
            SpawnOptions::default(),
        )
        .await?;

        ctx.call_tool(
            "key_capturing_tool",
            serde_json::json!({"message": "second"}),
            serde_json::json!(null),
            SpawnOptions::default(),
        )
        .await?;

        ctx.call_tool(
            "key_capturing_tool",
            serde_json::json!({"message": "third"}),
            serde_json::json!(null),
            SpawnOptions::default(),
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
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_error_on_call());

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
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
        .register_simple_tool_instance(KeyCapturingSimpleTool)
        .await
        .unwrap();
    executor
        .register_task_tool_instance(MultiCallTaskTool)
        .await
        .unwrap();

    // Spawn the task
    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool_by_name(
            "multi_call_task",
            serde_json::json!({"message": "test"}),
            serde_json::json!(null),
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
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_with_response(mock_response));

    let tool = InferenceSimpleTool;
    let ctx = SimpleToolContext::new(&pool, &t0_client);
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
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_error_on_call());

    let tool = InferenceSimpleTool;
    let ctx = SimpleToolContext::new(&pool, &t0_client);
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
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_with_response(mock_response));

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(format!("test_queue_{}", Uuid::now_v7()))
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .register_task_tool_instance(InferenceTaskTool)
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
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_with_response(mock_response));

    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
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
        .register_task_tool_instance(InferenceTaskTool)
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let result = executor
        .spawn_tool_by_name(
            "inference_task",
            serde_json::json!({"prompt": "Generate something"}),
            serde_json::json!(null),
            episode_id,
        )
        .await;

    assert!(
        result.is_ok(),
        "spawn_tool_by_name for InferenceTaskTool failed: {:?}",
        result.err()
    );
    let spawn_result = result.unwrap();
    assert!(!spawn_result.task_id.is_nil());
    Ok(())
}

// ============================================================================
// Empty Inference Output Tests
// ============================================================================

/// Create a mock that returns an empty chat response (no content blocks).
fn mock_client_with_empty_chat_response() -> MockTensorZeroClient {
    let mut mock = MockTensorZeroClient::new();
    mock.expect_inference().returning(|_| {
        Box::pin(async {
            Ok(InferenceResponse::Chat(ChatInferenceResponse {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
                variant_name: "test_variant".to_string(),
                content: vec![], // Empty content
                usage: Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(0),
                },
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: None,
            }))
        })
    });
    mock
}

/// Create a mock that returns a JSON response with None raw output.
fn mock_client_with_empty_json_response() -> MockTensorZeroClient {
    use tensorzero_core::endpoints::inference::JsonInferenceResponse;
    use tensorzero_core::inference::types::JsonInferenceOutput;

    let mut mock = MockTensorZeroClient::new();
    mock.expect_inference().returning(|_| {
        Box::pin(async {
            Ok(InferenceResponse::Json(JsonInferenceResponse {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
                variant_name: "test_variant".to_string(),
                output: JsonInferenceOutput {
                    raw: None, // Empty raw output
                    parsed: None,
                },
                usage: Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(0),
                },
                raw_usage: None,
                original_response: None,
                raw_response: None,
                finish_reason: None,
            }))
        })
    });
    mock
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn task_tool_inference_fails_on_empty_chat_response(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_with_empty_chat_response());

    let executor = ToolExecutor::builder()
        .pool(pool.clone())
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor
        .register_task_tool_instance(InferenceTaskTool)
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let spawn_result = executor
        .spawn_tool_by_name(
            "inference_task",
            serde_json::json!({"prompt": "This should fail"}),
            serde_json::json!(null),
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

    // Wait for task to be processed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Shutdown worker
    worker.shutdown().await;

    // Check task status by querying the runs table directly
    let task_record: (Option<chrono::DateTime<chrono::Utc>>, Option<String>) = sqlx::query_as(
        AssertSqlSafe(format!(
            "SELECT failed_at, failure_reason::text FROM durable.\"r_{queue_name}\" WHERE task_id = $1 AND failed_at IS NOT NULL",
        )),
    )
    .bind(spawn_result.task_id)
    .fetch_one(&pool)
    .await
    .expect("Failed to query task status");

    assert!(
        task_record.0.is_some(),
        "Task should have failed due to empty inference output"
    );
    assert!(
        task_record
            .1
            .as_ref()
            .map(|e| e.contains("empty output"))
            .unwrap_or(false),
        "Error message should mention empty output, got: {:?}",
        task_record.1
    );

    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn task_tool_inference_fails_on_empty_json_response(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client_with_empty_json_response());

    let executor = ToolExecutor::builder()
        .pool(pool.clone())
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor
        .register_task_tool_instance(InferenceTaskTool)
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let spawn_result = executor
        .spawn_tool_by_name(
            "inference_task",
            serde_json::json!({"prompt": "This should fail with JSON"}),
            serde_json::json!(null),
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

    // Wait for task to be processed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Shutdown worker
    worker.shutdown().await;

    // Check task status by querying the runs table directly
    let task_record: (Option<chrono::DateTime<chrono::Utc>>, Option<String>) = sqlx::query_as(
        AssertSqlSafe(format!(
            "SELECT failed_at, failure_reason::text FROM durable.\"r_{queue_name}\" WHERE task_id = $1 AND failed_at IS NOT NULL",
        )),
    )
    .bind(spawn_result.task_id)
    .fetch_one(&pool)
    .await
    .expect("Failed to query task status");

    assert!(
        task_record.0.is_some(),
        "Task should have failed due to empty JSON inference output"
    );
    assert!(
        task_record
            .1
            .as_ref()
            .map(|e| e.contains("empty output"))
            .unwrap_or(false),
        "Error message should mention empty output, got: {:?}",
        task_record.1
    );

    Ok(())
}
