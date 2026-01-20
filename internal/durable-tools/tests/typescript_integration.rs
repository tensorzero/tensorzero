//! TypeScript tool integration tests (requires Postgres and typescript feature).
//!
//! These tests verify that TypeScript tools can be registered and executed,
//! and can call Rust tools and vice versa.
//!
//! Run with: cargo test --test typescript_integration --features typescript

#![cfg(feature = "typescript")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use common::MockTensorZeroClient;
use durable::MIGRATOR;
use durable::SpawnOptions;
use durable::WorkerOptions;
use durable_tools::typescript::{TypeScriptTool, TypeScriptToolInstance};
use durable_tools::{
    SimpleTool, SimpleToolContext, TaskTool, TensorZeroClient, ToolContext, ToolExecutor,
    ToolMetadata, ToolResult,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{AssertSqlSafe, PgPool};
use uuid::Uuid;

// ============================================================================
// Test Helpers
// ============================================================================

#[derive(sqlx::FromRow, Debug)]
struct TaskRow {
    state: String,
    completed_payload: Option<serde_json::Value>,
    last_attempt_run: Option<uuid::Uuid>,
}

#[derive(sqlx::FromRow, Debug)]
struct RunRow {
    failure_reason: Option<serde_json::Value>,
}

/// Helper to query task state and result from the database.
async fn get_task_row(pool: &PgPool, queue_name: &str, task_id: uuid::Uuid) -> TaskRow {
    let query = AssertSqlSafe(format!(
        r#"SELECT state, completed_payload, last_attempt_run FROM durable."t_{queue_name}" WHERE task_id = $1"#
    ));
    sqlx::query_as(query)
        .bind(task_id)
        .fetch_one(pool)
        .await
        .expect("Failed to query task")
}

/// Helper to query run failure reason from the database.
async fn get_run_failure(
    pool: &PgPool,
    queue_name: &str,
    run_id: uuid::Uuid,
) -> Option<serde_json::Value> {
    let query = AssertSqlSafe(format!(
        r#"SELECT failure_reason FROM durable."r_{queue_name}" WHERE run_id = $1"#
    ));
    let row: RunRow = sqlx::query_as(query)
        .bind(run_id)
        .fetch_one(pool)
        .await
        .expect("Failed to query run");
    row.failure_reason
}

/// Wait for a task to complete (or fail) with timeout.
async fn wait_for_task_completion(
    pool: &PgPool,
    queue_name: &str,
    task_id: uuid::Uuid,
    timeout: Duration,
) -> TaskRow {
    let start = std::time::Instant::now();
    loop {
        let row = get_task_row(pool, queue_name, task_id).await;
        if row.state == "completed" || row.state == "failed" {
            return row;
        }
        if start.elapsed() > timeout {
            // Get failure details if available
            let failure_reason = if let Some(run_id) = row.last_attempt_run {
                get_run_failure(pool, queue_name, run_id).await
            } else {
                None
            };
            panic!(
                "Task {} did not complete within {:?}. Current state: {:?}, failure_reason: {:?}",
                task_id, timeout, row.state, failure_reason
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ============================================================================
// Test Fixtures - Rust SimpleTool
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct AddParams {
    a: i32,
    b: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AddOutput {
    sum: i32,
}

/// A simple `SimpleTool` that adds two numbers, for testing TypeScript -> Rust calls.
#[derive(Default, Clone)]
struct AddSimpleTool;

impl ToolMetadata for AddSimpleTool {
    type SideInfo = ();
    type Output = AddOutput;
    type LlmParams = AddParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("add")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Adds two numbers")
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(10)
    }
}

#[async_trait]
impl SimpleTool for AddSimpleTool {
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        _ctx: SimpleToolContext<'_>,
        _idempotency_key: &str,
    ) -> ToolResult<Self::Output> {
        Ok(AddOutput {
            sum: llm_params.a + llm_params.b,
        })
    }
}

// ============================================================================
// Test Fixtures - Rust TaskTool that calls TypeScript
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct GreetParams {
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct GreetOutput {
    greeting: String,
}

/// A `TaskTool` that calls a TypeScript tool.
#[derive(Default, Clone)]
struct RustCallsTypeScriptTool;

impl ToolMetadata for RustCallsTypeScriptTool {
    type SideInfo = ();
    type Output = GreetOutput;
    type LlmParams = GreetParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("rust_calls_typescript")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Calls a TypeScript tool from Rust")
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(60)
    }
}

#[async_trait]
impl TaskTool for RustCallsTypeScriptTool {
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        ctx: &ToolContext,
    ) -> ToolResult<Self::Output> {
        // Call the TypeScript greeter tool
        let result = ctx
            .call_tool(
                "ts_greeter",
                json!({"name": llm_params.name}),
                json!(null),
                SpawnOptions::default(),
            )
            .await?;

        let greeting = result["greeting"]
            .as_str()
            .unwrap_or("No greeting")
            .to_string();

        Ok(GreetOutput { greeting })
    }
}

// ============================================================================
// Test Fixtures - Rust TaskTool that calls TypeScript which calls Rust
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
struct ChainedAddParams {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ChainedAddOutput {
    original_x: i32,
    original_y: i32,
    sum: i32,
}

/// A `TaskTool` that calls a TypeScript tool, which in turn calls a Rust SimpleTool.
/// This tests the full chain: Rust TaskTool → TypeScript → Rust SimpleTool.
#[derive(Default, Clone)]
struct RustCallsTsCallsRustTool;

impl ToolMetadata for RustCallsTsCallsRustTool {
    type SideInfo = ();
    type Output = ChainedAddOutput;
    type LlmParams = ChainedAddParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("rust_calls_ts_calls_rust")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed("Calls a TypeScript tool that calls a Rust tool")
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(60)
    }
}

#[async_trait]
impl TaskTool for RustCallsTsCallsRustTool {
    async fn execute(
        &self,
        llm_params: <Self as ToolMetadata>::LlmParams,
        _side_info: Self::SideInfo,
        ctx: &ToolContext,
    ) -> ToolResult<Self::Output> {
        // Call the TypeScript tool that calls the Rust add tool
        let result = ctx
            .call_tool(
                "ts_calls_rust",
                json!({"x": llm_params.x, "y": llm_params.y}),
                json!(null),
                SpawnOptions::default(),
            )
            .await?;

        Ok(ChainedAddOutput {
            original_x: result["original_x"].as_i64().unwrap_or(0) as i32,
            original_y: result["original_y"].as_i64().unwrap_or(0) as i32,
            sum: result["sum_from_rust"].as_i64().unwrap_or(0) as i32,
        })
    }
}

// ============================================================================
// TypeScript Tool Execution Tests
// ============================================================================

/// Create a simple TypeScript tool that echoes its input.
fn create_echo_ts_tool() -> TypeScriptToolInstance {
    let tool = TypeScriptTool::builder("ts_echo")
        .description("Echoes the input message")
        .typescript_code(
            r#"
            export default {
                name: "ts_echo",
                description: "Echoes the input",
                parameters_schema: {
                    type: "object",
                    properties: {
                        message: { type: "string" }
                    },
                    required: ["message"]
                },
                async run(params: { message: string }, sideInfo: any) {
                    return { echoed: params.message };
                }
            };
        "#,
        )
        .parameters_schema(json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        }))
        .build()
        .expect("Failed to build TypeScript tool");

    TypeScriptToolInstance::new(tool)
}

/// Create a TypeScript tool that calls the Rust `add` SimpleTool.
fn create_ts_calls_rust_tool() -> TypeScriptToolInstance {
    let tool = TypeScriptTool::builder("ts_calls_rust")
        .description("Calls the Rust add tool")
        .typescript_code(
            r#"
            export default {
                name: "ts_calls_rust",
                description: "Calls Rust add tool",
                parameters_schema: {
                    type: "object",
                    properties: {
                        x: { type: "number" },
                        y: { type: "number" }
                    },
                    required: ["x", "y"]
                },
                async run(params: { x: number, y: number }, sideInfo: any) {
                    // Call the Rust SimpleTool
                    const result = await ctx.callTool("add", { a: params.x, b: params.y }, null);
                    return {
                        original_x: params.x,
                        original_y: params.y,
                        sum_from_rust: result.sum
                    };
                }
            };
        "#,
        )
        .parameters_schema(json!({
            "type": "object",
            "properties": {
                "x": { "type": "number" },
                "y": { "type": "number" }
            },
            "required": ["x", "y"]
        }))
        .build()
        .expect("Failed to build TypeScript tool");

    TypeScriptToolInstance::new(tool)
}

/// Create a TypeScript greeter tool (called by Rust TaskTool).
fn create_ts_greeter_tool() -> TypeScriptToolInstance {
    let tool = TypeScriptTool::builder("ts_greeter")
        .description("Greets a person")
        .typescript_code(
            r#"
            export default {
                name: "ts_greeter",
                description: "Greets a person",
                parameters_schema: {
                    type: "object",
                    properties: {
                        name: { type: "string" }
                    },
                    required: ["name"]
                },
                async run(params: { name: string }, sideInfo: any) {
                    return { greeting: `Hello, ${params.name}!` };
                }
            };
        "#,
        )
        .parameters_schema(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"]
        }))
        .build()
        .expect("Failed to build TypeScript tool");

    TypeScriptToolInstance::new(tool)
}

/// Create a TypeScript tool that throws an error.
fn create_ts_error_tool() -> TypeScriptToolInstance {
    let tool = TypeScriptTool::builder("ts_error")
        .description("Always throws an error")
        .typescript_code(
            r#"
            export default {
                name: "ts_error",
                description: "Always throws an error",
                parameters_schema: {
                    type: "object",
                    properties: {
                        message: { type: "string" }
                    },
                    required: ["message"]
                },
                async run(params: { message: string }, sideInfo: any) {
                    throw new Error(`Intentional error: ${params.message}`);
                }
            };
        "#,
        )
        .parameters_schema(json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        }))
        .build()
        .expect("Failed to build TypeScript tool");

    TypeScriptToolInstance::new(tool)
}

// ============================================================================
// Tests
// ============================================================================

#[sqlx::test(migrator = "MIGRATOR")]
async fn typescript_tool_can_be_registered_via_register_task_tool_instance(
    pool: PgPool,
) -> sqlx::Result<()> {
    let queue_name = format!("ts_reg_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());
    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    // Register TypeScript tool via register_task_tool_instance
    let ts_tool = create_echo_ts_tool();
    executor
        .register_task_tool_instance(ts_tool)
        .await
        .expect("Failed to register TypeScript tool");

    // Verify tool is registered
    let definitions = executor.tool_definitions().await.unwrap();
    let names: Vec<&str> = definitions
        .iter()
        .filter_map(|d| match d {
            tensorzero::Tool::Function(f) => Some(f.name.as_str()),
            tensorzero::Tool::OpenAICustom(_) => None,
        })
        .collect();

    assert!(
        names.contains(&"ts_echo"),
        "TypeScript tool should be registered"
    );
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn typescript_tool_executes_and_returns_result(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("ts_exec_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());
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

    // Register TypeScript tool
    let ts_tool = create_echo_ts_tool();
    executor
        .register_task_tool_instance(ts_tool)
        .await
        .expect("Failed to register TypeScript tool");

    // Spawn the tool
    let episode_id = Uuid::now_v7();
    let spawn_result = executor
        .spawn_tool_by_name(
            "ts_echo",
            json!({"message": "Hello from TypeScript!"}),
            json!(null),
            episode_id,
        )
        .await
        .expect("Failed to spawn TypeScript tool");

    // Start worker
    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for task completion with timeout
    let task_row = wait_for_task_completion(
        &pool,
        &queue_name,
        spawn_result.task_id,
        Duration::from_secs(10),
    )
    .await;
    worker.shutdown().await;

    // If task failed, print the failure reason for debugging
    if task_row.state == "failed"
        && let Some(run_id) = task_row.last_attempt_run
    {
        let failure = get_run_failure(&pool, &queue_name, run_id).await;
        panic!("Task failed with failure_reason: {failure:?}");
    }

    assert_eq!(
        task_row.state, "completed",
        "Task should be completed, got: {task_row:?}"
    );

    let result = task_row
        .completed_payload
        .expect("Task should have a result");
    assert_eq!(
        result["echoed"], "Hello from TypeScript!",
        "TypeScript tool should echo the message"
    );
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn rust_task_tool_can_call_typescript_tool(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("ts_r2ts_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());
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

    // Register TypeScript greeter tool
    let ts_greeter = create_ts_greeter_tool();
    executor
        .register_task_tool_instance(ts_greeter)
        .await
        .expect("Failed to register TypeScript tool");

    // Register Rust TaskTool that calls TypeScript
    executor
        .register_task_tool::<RustCallsTypeScriptTool>()
        .await
        .expect("Failed to register Rust tool");

    // Spawn the Rust tool (which will call the TypeScript tool)
    let episode_id = Uuid::now_v7();
    let spawn_result = executor
        .spawn_tool::<RustCallsTypeScriptTool>(
            GreetParams {
                name: "World".to_string(),
            },
            (),
            episode_id,
            SpawnOptions::default(),
        )
        .await
        .expect("Failed to spawn Rust tool");

    // Start worker
    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for task completion with timeout
    let task_row = wait_for_task_completion(
        &pool,
        &queue_name,
        spawn_result.task_id,
        Duration::from_secs(10),
    )
    .await;
    worker.shutdown().await;

    assert_eq!(
        task_row.state, "completed",
        "Task should be completed, got: {task_row:?}"
    );

    let result = task_row
        .completed_payload
        .expect("Task should have a result");
    assert_eq!(
        result["greeting"], "Hello, World!",
        "Rust tool should receive greeting from TypeScript tool"
    );
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn typescript_tool_can_call_rust_simple_tool(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("ts_ts2r_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());
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

    // Register Rust SimpleTool
    executor
        .register_simple_tool::<AddSimpleTool>()
        .await
        .expect("Failed to register Rust SimpleTool");

    // Register TypeScript tool that calls the Rust tool
    let ts_tool = create_ts_calls_rust_tool();
    executor
        .register_task_tool_instance(ts_tool)
        .await
        .expect("Failed to register TypeScript tool");

    // Spawn the TypeScript tool (which will call the Rust SimpleTool)
    let episode_id = Uuid::now_v7();
    let spawn_result = executor
        .spawn_tool_by_name(
            "ts_calls_rust",
            json!({"x": 10, "y": 32}),
            json!(null),
            episode_id,
        )
        .await
        .expect("Failed to spawn TypeScript tool");

    // Start worker
    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for task completion with timeout
    let task_row = wait_for_task_completion(
        &pool,
        &queue_name,
        spawn_result.task_id,
        Duration::from_secs(10),
    )
    .await;
    worker.shutdown().await;

    assert_eq!(
        task_row.state, "completed",
        "Task should be completed, got: {task_row:?}"
    );

    let result = task_row
        .completed_payload
        .expect("Task should have a result");
    assert_eq!(result["original_x"], 10);
    assert_eq!(result["original_y"], 32);
    assert_eq!(
        result["sum_from_rust"], 42,
        "TypeScript tool should receive sum from Rust SimpleTool"
    );
    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn typescript_tool_error_propagates_correctly(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("ts_err_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());

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

    // Register TypeScript tool that throws an error
    let ts_tool = create_ts_error_tool();
    executor
        .register_task_tool_instance(ts_tool)
        .await
        .expect("Failed to register TypeScript tool");

    // Spawn the tool
    let episode_id = Uuid::now_v7();
    let spawn_result = executor
        .spawn_tool_by_name(
            "ts_error",
            json!({"message": "test failure"}),
            json!(null),
            episode_id,
        )
        .await
        .expect("Failed to spawn TypeScript tool");

    // Start worker with max_attempts=1 so it fails immediately
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
    worker.shutdown().await;

    // Check task state - it should be failed (or retrying)
    let query = AssertSqlSafe(format!(
        r#"SELECT state FROM durable."t_{queue_name}" WHERE task_id = $1"#
    ));
    let task_state: String = sqlx::query_scalar(query)
        .bind(spawn_result.task_id)
        .fetch_one(&pool)
        .await
        .expect("Failed to query task state");

    // The task should have failed or be in failed state
    // Note: depending on retry configuration, it might be "failed" or "pending" with attempts > 0
    assert!(
        task_state == "failed" || task_state == "pending",
        "Task should be failed or pending retry, got: {task_state}"
    );

    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn typescript_tool_instance_implements_clone(pool: PgPool) -> sqlx::Result<()> {
    // This test verifies that TypeScriptToolInstance implements Clone,
    // which is required for register_task_tool_instance.
    let queue_name = format!("ts_cln_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());
    let executor = ToolExecutor::builder()
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    // Create and clone the tool instance
    let ts_tool = create_echo_ts_tool();
    let _cloned = ts_tool.clone(); // This should compile

    // Register should work (it needs Clone internally)
    executor
        .register_task_tool_instance(ts_tool)
        .await
        .expect("Failed to register TypeScript tool");

    Ok(())
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn rust_task_tool_calls_typescript_which_calls_rust(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("ts_chain_{}", Uuid::now_v7());
    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(MockTensorZeroClient::new());
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

    // Register Rust SimpleTool (the innermost tool in the chain)
    executor
        .register_simple_tool::<AddSimpleTool>()
        .await
        .expect("Failed to register Rust SimpleTool");

    // Register TypeScript tool that calls the Rust SimpleTool
    let ts_tool = create_ts_calls_rust_tool();
    executor
        .register_task_tool_instance(ts_tool)
        .await
        .expect("Failed to register TypeScript tool");

    // Register Rust TaskTool that calls the TypeScript tool (outermost in the chain)
    executor
        .register_task_tool::<RustCallsTsCallsRustTool>()
        .await
        .expect("Failed to register Rust TaskTool");

    // Spawn the outermost Rust tool (which calls TypeScript, which calls Rust)
    let episode_id = Uuid::now_v7();
    let spawn_result = executor
        .spawn_tool::<RustCallsTsCallsRustTool>(
            ChainedAddParams { x: 100, y: 23 },
            (),
            episode_id,
            SpawnOptions::default(),
        )
        .await
        .expect("Failed to spawn Rust tool");

    // Start worker
    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    // Wait for task completion with timeout
    let task_row = wait_for_task_completion(
        &pool,
        &queue_name,
        spawn_result.task_id,
        Duration::from_secs(15),
    )
    .await;
    worker.shutdown().await;

    // If task failed, print the failure reason for debugging
    if task_row.state == "failed"
        && let Some(run_id) = task_row.last_attempt_run
    {
        let failure = get_run_failure(&pool, &queue_name, run_id).await;
        panic!("Task failed with failure_reason: {failure:?}");
    }

    assert_eq!(
        task_row.state, "completed",
        "Task should be completed, got: {task_row:?}"
    );

    let result = task_row
        .completed_payload
        .expect("Task should have a result");

    // Verify the full chain worked: Rust → TypeScript → Rust → result
    assert_eq!(
        result["original_x"], 100,
        "Should preserve original x through the chain"
    );
    assert_eq!(
        result["original_y"], 23,
        "Should preserve original y through the chain"
    );
    assert_eq!(
        result["sum"], 123,
        "Rust TaskTool should receive sum computed by Rust SimpleTool via TypeScript"
    );

    Ok(())
}
