use chrono::{DateTime, Utc};
use durable::{Durable, TaskContext, TaskHandle};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::{ClientInferenceParams, InferenceResponse};
use uuid::Uuid;

use crate::error::{ToolError, ToolResult};
use crate::inference::{InferenceClient, InferenceError};
use crate::registry::ToolRegistry;
use crate::task_tool::TaskToolParams;

/// Type alias for the Durable client with `ToolAppState`.
pub type DurableClient = Durable<ToolAppState>;

/// Application state passed to all tools via durable's `State` type parameter.
///
/// This is cloned and passed to each task execution by the durable worker.
#[derive(Clone)]
pub struct ToolAppState {
    /// Database connection pool for database operations.
    pool: PgPool,
    /// Tool registry for looking up and calling other tools.
    tool_registry: Arc<tokio::sync::RwLock<ToolRegistry>>,
    /// Inference client for calling TensorZero inference.
    inference_client: Arc<dyn InferenceClient>,
}

impl ToolAppState {
    /// Create a new application state.
    pub fn new(
        pool: PgPool,
        tool_registry: Arc<tokio::sync::RwLock<ToolRegistry>>,
        inference_client: Arc<dyn InferenceClient>,
    ) -> Self {
        Self {
            pool,
            tool_registry,
            inference_client,
        }
    }

    /// Get a reference to the database pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

/// Context provided to `TaskTool` execution.
///
/// Wraps durable's `TaskContext` and provides access to the application context
/// along with helper methods for calling other tools and checkpointing operations.
pub struct ToolContext<'a> {
    task_ctx: &'a mut TaskContext<ToolAppState>,
    app_state: &'a ToolAppState,
    episode_id: Uuid,
    /// Counter for generating unique tool call identifiers.
    tool_call_counter: u32,
}

impl<'a> ToolContext<'a> {
    /// Create a new tool context.
    pub fn new(
        task_ctx: &'a mut TaskContext<ToolAppState>,
        app_ctx: &'a ToolAppState,
        episode_id: Uuid,
    ) -> Self {
        Self {
            task_ctx,
            app_state: app_ctx,
            episode_id,
            tool_call_counter: 0,
        }
    }

    /// Get the next tool call ID (increments counter).
    fn next_tool_call_id(&mut self) -> u32 {
        self.tool_call_counter += 1;
        self.tool_call_counter
    }

    /// Get the task ID (useful as an idempotency key base).
    pub fn task_id(&self) -> Uuid {
        self.task_ctx.task_id
    }

    /// Get the episode ID for this tool execution.
    pub fn episode_id(&self) -> Uuid {
        self.episode_id
    }

    /// Get a reference to the database pool.
    pub fn pool(&self) -> &PgPool {
        &self.app_state.pool
    }

    /// Get mutable access to the underlying durable `TaskContext`.
    ///
    /// Use this for advanced operations not exposed by `ToolContext`.
    pub fn task_ctx(&mut self) -> &mut TaskContext<ToolAppState> {
        self.task_ctx
    }

    /// Execute a checkpointed step.
    ///
    /// If the step was already completed in a previous run, returns the cached
    /// result without re-executing the closure. This provides "exactly-once"
    /// semantics for side effects within the step.
    ///
    /// # Errors
    ///
    /// Returns an error if the step execution fails.
    pub async fn step<T, P, Fut>(
        &mut self,
        base_name: &str,
        params: P,
        f: fn(P, ToolAppState) -> Fut,
    ) -> ToolResult<T>
    where
        P: Serialize,
        T: Serialize + DeserializeOwned + Send,
        Fut: std::future::Future<Output = anyhow::Result<T>> + Send,
    {
        self.task_ctx
            .step(base_name, params, f)
            .await
            .map_err(Into::into)
    }

    /// Call another tool by name with JSON params.
    ///
    /// Side info defaults to `null` (compatible with `SideInfo = ()`).
    ///
    /// - For `TaskTool`: spawns as a subtask and joins to wait for completion
    /// - For `SimpleTool`: executes within a checkpointed step
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub async fn call_tool(
        &mut self,
        tool_name: &str,
        llm_params: JsonValue,
    ) -> ToolResult<JsonValue> {
        self.call_tool_with_side_info(tool_name, llm_params, serde_json::json!(null))
            .await
    }

    /// Call another tool by name with JSON params and explicit side info.
    ///
    /// - For `TaskTool`: spawns as a subtask and joins to wait for completion
    /// - For `SimpleTool`: executes within a checkpointed step
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub async fn call_tool_with_side_info(
        &mut self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
    ) -> ToolResult<JsonValue> {
        let is_durable = {
            let registry = self.app_state.tool_registry.read().await;
            registry
                .is_durable(tool_name)
                .ok_or_else(|| ToolError::ToolNotFound(tool_name.to_string()))?
        };

        if is_durable {
            self.call_task_tool(tool_name, llm_params, side_info).await
        } else {
            self.call_simple_tool(tool_name, llm_params, side_info)
                .await
        }
    }

    /// Call a `TaskTool` (spawns as subtask and waits for completion).
    async fn call_task_tool(
        &mut self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
    ) -> ToolResult<JsonValue> {
        // Get a unique ID for this tool call to ensure each invocation
        // spawns a new subtask, even when calling the same tool multiple times.
        let call_id = self.next_tool_call_id();

        // Wrap params with side_info and episode_id for the TaskToolAdapter
        let wrapped_params = TaskToolParams {
            llm_params,
            side_info,
            episode_id: self.episode_id,
        };
        let wrapped_params = serde_json::to_value(wrapped_params)?;

        // Spawn the tool as a subtask using spawn_subtask_by_name
        let spawn_name = format!("spawn:{tool_name}:{call_id}");
        let handle: TaskHandle<JsonValue> = self
            .spawn_subtask_by_name(&spawn_name, tool_name, wrapped_params)
            .await?;

        // Join and wait for result
        let result: JsonValue = self.task_ctx.join(handle).await?;

        Ok(result)
    }

    /// Spawn a subtask by task name (for dynamic tool invocation).
    ///
    /// This delegates to durable's `TaskContext::spawn_by_name` which handles
    /// checkpointing and uses the queue name configured on the durable client.
    async fn spawn_subtask_by_name<T: DeserializeOwned>(
        &mut self,
        name: &str,
        task_name: &str,
        params: JsonValue,
    ) -> ToolResult<TaskHandle<T>> {
        let handle: TaskHandle<T> = self
            .task_ctx
            .spawn_by_name(name, task_name, params, durable::SpawnOptions::default())
            .await?;
        Ok(handle)
    }

    /// Call a `SimpleTool` (executes within a checkpointed step).
    async fn call_simple_tool(
        &mut self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
    ) -> ToolResult<JsonValue> {
        // Get a unique ID for this tool call to ensure each invocation
        // gets a unique idempotency key, even when calling the same tool multiple times.
        let call_id = self.next_tool_call_id();
        let task_id = self.task_id();

        let step_name = format!("simple:{tool_name}:{call_id}");

        // Clone what we need for the step closure
        let tool_name = tool_name.to_string();

        self.step(
            &step_name,
            (tool_name, task_id, call_id, llm_params, side_info),
            |(tool_name, task_id, call_id, llm_params, side_info), state| async move {
                // Create SimpleToolContext
                let simple_ctx = SimpleToolContext::new(&state.pool, &state.inference_client);

                // Generate idempotency key using task_id and call_id
                let idempotency_key = format!("{task_id}:{tool_name}:{call_id}");

                // Get the simple tool from registry
                let simple_tool = {
                    state
                        .tool_registry
                        .read()
                        .await
                        .get_simple_tool(&tool_name)
                        .ok_or_else(|| ToolError::ToolNotFound(tool_name.clone()))?
                };

                simple_tool
                    .execute_erased(llm_params, side_info, simple_ctx, &idempotency_key)
                    .await
            },
        )
        .await
    }

    /// Sleep for a duration (durable - survives restarts).
    ///
    /// # Errors
    ///
    /// Returns an error if the sleep operation fails.
    pub async fn sleep_for(&mut self, name: &str, duration: Duration) -> ToolResult<()> {
        self.task_ctx
            .sleep_for(name, duration)
            .await
            .map_err(Into::into)
    }

    /// Wait for an event by name.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting for the event fails or times out.
    pub async fn await_event<T: DeserializeOwned>(
        &mut self,
        event_name: &str,
        timeout: Option<Duration>,
    ) -> ToolResult<T> {
        self.task_ctx
            .await_event(event_name, timeout)
            .await
            .map_err(Into::into)
    }

    /// Emit an event.
    ///
    /// # Errors
    ///
    /// Returns an error if emitting the event fails.
    pub async fn emit_event<T: Serialize>(&self, event_name: &str, payload: &T) -> ToolResult<()> {
        self.task_ctx
            .emit_event(event_name, payload)
            .await
            .map_err(Into::into)
    }

    /// Generate a durable random value in [0, 1).
    ///
    /// # Errors
    ///
    /// Returns an error if generating the random value fails.
    pub async fn rand(&mut self) -> ToolResult<f64> {
        self.task_ctx.rand().await.map_err(Into::into)
    }

    /// Get the current time as a durable checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if getting the current time fails.
    pub async fn now(&mut self) -> ToolResult<DateTime<Utc>> {
        self.task_ctx.now().await.map_err(Into::into)
    }

    /// Generate a durable UUID v7.
    ///
    /// # Errors
    ///
    /// Returns an error if generating the UUID fails.
    pub async fn uuid7(&mut self) -> ToolResult<Uuid> {
        self.task_ctx.uuid7().await.map_err(Into::into)
    }

    /// Call TensorZero inference with full parameter control.
    ///
    /// This is a checkpointed operation - results are cached on restart.
    /// Streaming inference is not supported and will return an error.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let response = ctx.inference(ClientInferenceParams {
    ///     function_name: Some("my_function".to_string()),
    ///     input: Input { messages: Some(vec![...]), ..Default::default() },
    ///     ..Default::default()
    /// }).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the inference call fails.
    pub async fn inference(
        &mut self,
        params: ClientInferenceParams,
    ) -> ToolResult<InferenceResponse> {
        let step_name = format!(
            "inference:{}",
            params
                .function_name
                .as_deref()
                .or(params.model_name.as_deref())
                .unwrap_or("unknown")
        );

        self.step(&step_name, params, |params, state| async move {
            state
                .inference_client
                .inference(params)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))
        })
        .await
    }
}

/// Simplified context for `SimpleTool` execution.
///
/// `SimpleTools` run inside a `TaskTool`'s `step()` checkpoint, so they don't
/// have access to checkpointing operations themselves. They can access the
/// database pool for queries and the inference client for TensorZero calls.
pub struct SimpleToolContext<'a> {
    pool: &'a PgPool,
    inference_client: &'a Arc<dyn InferenceClient>,
}

impl<'a> SimpleToolContext<'a> {
    /// Create a new simple tool context.
    pub fn new(pool: &'a PgPool, inference_client: &'a Arc<dyn InferenceClient>) -> Self {
        Self {
            pool,
            inference_client,
        }
    }

    /// Get a reference to the database pool.
    pub fn pool(&self) -> &PgPool {
        self.pool
    }

    /// Call TensorZero inference.
    ///
    /// Note: `SimpleTools` run inside a `TaskTool`'s `step()`, so this call
    /// is already checkpointed by the outer step.
    ///
    /// # Errors
    ///
    /// Returns an error if the inference call fails.
    pub async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, InferenceError> {
        self.inference_client.inference(params).await
    }
}
