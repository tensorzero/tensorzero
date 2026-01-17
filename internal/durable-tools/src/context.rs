use chrono::{DateTime, Utc};
use durable::{Durable, SpawnOptions, TaskContext, TaskHandle};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::{ClientInferenceParams, InferenceResponse};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::error::{NonControlToolError, ToolResult};
use crate::registry::ToolRegistry;
use crate::task_tool::TaskToolParams;
use crate::tensorzero_client::{TensorZeroClient, TensorZeroClientError};

/// Handle returned by `spawn_tool`, can be joined later with `join_tool`.
///
/// This enum allows a uniform API for both `TaskTool` and `SimpleTool`:
/// - `TaskTool`: spawns as a background subtask, `join_tool` waits for completion
/// - `SimpleTool`: executes immediately (still checkpointed), `join_tool` returns stored result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolHandle {
    /// TaskTool - runs in background, join waits for completion
    Async(TaskHandle<JsonValue>),
    /// SimpleTool - already executed, result stored inline
    Sync(JsonValue),
}

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
    /// TensorZero client for calling inference and autopilot operations.
    t0_client: Arc<dyn TensorZeroClient>,
}

impl ToolAppState {
    /// Create a new application state.
    pub fn new(
        pool: PgPool,
        tool_registry: Arc<tokio::sync::RwLock<ToolRegistry>>,
        t0_client: Arc<dyn TensorZeroClient>,
    ) -> Self {
        Self {
            pool,
            tool_registry,
            t0_client,
        }
    }

    /// Get a reference to the database pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Get the TensorZero client.
    ///
    /// This provides access to inference, autopilot events, and other client operations.
    pub fn t0_client(&self) -> &Arc<dyn TensorZeroClient> {
        &self.t0_client
    }
}

/// Internal state for `ToolContext`, protected by a mutex.
struct ToolContextInner {
    task_ctx: TaskContext<ToolAppState>,
    app_state: ToolAppState,
    episode_id: Uuid,
    /// Counter for generating unique tool call identifiers.
    tool_call_counter: u32,
}

/// Context provided to `TaskTool` execution.
///
/// Wraps durable's `TaskContext` and provides access to the application context
/// along with helper methods for calling other tools and checkpointing operations.
///
/// This type is `Clone` and uses interior mutability via `Arc<Mutex<...>>`,
/// allowing it to be shared across async boundaries (e.g., with embedded runtimes).
#[derive(Clone)]
pub struct ToolContext {
    inner: Arc<Mutex<ToolContextInner>>,
}

impl ToolContext {
    /// Create a new tool context.
    ///
    /// Takes ownership of the `TaskContext` and `ToolAppState`.
    pub fn new(
        task_ctx: TaskContext<ToolAppState>,
        app_state: ToolAppState,
        episode_id: Uuid,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ToolContextInner {
                task_ctx,
                app_state,
                episode_id,
                tool_call_counter: 0,
            })),
        }
    }

    /// Get the task ID (useful as an idempotency key base).
    pub async fn task_id(&self) -> Uuid {
        let inner = self.inner.lock().await;
        inner.task_ctx.task_id
    }

    /// Get the episode ID for this tool execution.
    pub async fn episode_id(&self) -> Uuid {
        let inner = self.inner.lock().await;
        inner.episode_id
    }

    /// Get a clone of the database pool.
    pub async fn pool(&self) -> PgPool {
        let inner = self.inner.lock().await;
        inner.app_state.pool.clone()
    }

    /// Get the inference client for direct access to all client operations.
    ///
    /// This provides access to inference, autopilot events, and other client operations.
    /// For durability, wrap client calls in `ctx.step()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Send a tool result to autopilot (checkpointed)
    /// ctx.step("send_result", params, |params, state| async move {
    ///     state.t0_client()
    ///         .create_autopilot_event(session_id, request)
    ///         .await
    ///         .map_err(|e| anyhow::anyhow!("{e}"))
    /// }).await?;
    /// ```
    pub async fn client(&self) -> Arc<dyn TensorZeroClient> {
        let inner = self.inner.lock().await;
        inner.app_state.t0_client.clone()
    }

    /// Get a reference to the tool registry.
    ///
    /// Use this to iterate over tools and convert them to TensorZero tool definitions.
    /// The caller is responsible for acquiring the read lock.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = ctx.tool_registry().await;
    /// let guard = registry.read().await;
    /// let tools: Result<Vec<Tool>, _> = guard.iter()
    ///     .filter(|t| !t.is_durable())
    ///     .map(Tool::try_from)
    ///     .collect();
    /// ```
    pub async fn tool_registry(&self) -> Arc<tokio::sync::RwLock<ToolRegistry>> {
        let inner = self.inner.lock().await;
        inner.app_state.tool_registry.clone()
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
        &self,
        base_name: &str,
        params: P,
        f: fn(P, ToolAppState) -> Fut,
    ) -> ToolResult<T>
    where
        P: Serialize,
        T: Serialize + DeserializeOwned + Send,
        Fut: std::future::Future<Output = anyhow::Result<T>> + Send,
    {
        let mut inner = self.inner.lock().await;
        inner
            .task_ctx
            .step(base_name, params, f)
            .await
            .map_err(Into::into)
    }

    /// Call another tool by name and wait for its result.
    ///
    /// This is a convenience method that spawns and immediately joins.
    /// For more control, use `spawn_tool` and `join_tool` separately.
    ///
    /// - For `TaskTool`: spawns as a subtask and joins to wait for completion
    /// - For `SimpleTool`: executes within a checkpointed step
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The registered name of the tool to call
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub async fn call_tool(
        &self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        options: SpawnOptions,
    ) -> ToolResult<JsonValue> {
        let handle = self
            .spawn_tool(tool_name, llm_params, side_info, options)
            .await?;
        self.join_tool(handle).await
    }

    /// Spawn a tool without waiting for completion.
    ///
    /// Returns a `ToolHandle` that can be joined later with `join_tool`.
    ///
    /// - For `TaskTool`: spawns as a background subtask
    /// - For `SimpleTool`: executes immediately (still checkpointed), returns completed handle
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The registered name of the tool to spawn
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Spawn multiple tools
    /// let h1 = ctx.spawn_tool("research", json!({"topic": "rust"}), json!(null)).await?;
    /// let h2 = ctx.spawn_tool("search", json!({"query": "async"}), json!(null)).await?;
    ///
    /// // Do other work while TaskTools run in background...
    ///
    /// // Join to get results
    /// let r1 = ctx.join_tool(h1).await?;
    /// let r2 = ctx.join_tool(h2).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or spawning fails.
    pub async fn spawn_tool(
        &self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        options: SpawnOptions,
    ) -> ToolResult<ToolHandle> {
        let mut inner = self.inner.lock().await;

        let is_durable = {
            let registry = inner.app_state.tool_registry.read().await;
            // Validate params before spawning
            registry.validate_params(tool_name, &llm_params, &side_info)?;
            registry
                .is_durable(tool_name)
                .ok_or_else(|| NonControlToolError::ToolNotFound {
                    name: tool_name.to_string(),
                })?
        };

        if is_durable {
            // TaskTool: spawn as subtask
            inner.tool_call_counter += 1;
            let call_id = inner.tool_call_counter;
            let episode_id = inner.episode_id;

            let wrapped_params = TaskToolParams {
                llm_params,
                side_info,
                episode_id,
            };
            let wrapped_params = serde_json::to_value(wrapped_params)?;
            let spawn_name = format!("spawn:{tool_name}:{call_id}");
            let handle: TaskHandle<JsonValue> = inner
                .task_ctx
                .spawn_by_name(&spawn_name, tool_name, wrapped_params, options)
                .await?;
            Ok(ToolHandle::Async(handle))
        } else {
            // SimpleTool: execute immediately (still checkpointed via step)
            inner.tool_call_counter += 1;
            let call_id = inner.tool_call_counter;
            let task_id = inner.task_ctx.task_id;

            let step_name = format!("simple:{tool_name}:{call_id}");

            // Clone what we need for the step closure
            let tool_name = tool_name.to_string();

            let result: JsonValue = inner
                .task_ctx
                .step(
                    &step_name,
                    (tool_name.clone(), task_id, call_id, llm_params, side_info),
                    |(tool_name, task_id, call_id, llm_params, side_info), state| async move {
                        // Create SimpleToolContext
                        let simple_ctx = SimpleToolContext::new(&state.pool, &state.t0_client);

                        // Generate idempotency key using task_id and call_id
                        let idempotency_key = format!("{task_id}:{tool_name}:{call_id}");

                        // Get the simple tool from registry
                        let simple_tool = {
                            state
                                .tool_registry
                                .read()
                                .await
                                .get_simple_tool(&tool_name)
                                .ok_or_else(|| NonControlToolError::ToolNotFound {
                                    name: tool_name.clone(),
                                })?
                        };

                        simple_tool
                            .execute_erased(llm_params, side_info, simple_ctx, &idempotency_key)
                            .await
                    },
                )
                .await?;

            Ok(ToolHandle::Sync(result))
        }
    }

    /// Wait for a spawned tool to complete and return its result.
    ///
    /// - For `Async` handles (TaskTool): waits for the subtask to complete
    /// - For `Sync` handles (SimpleTool): returns the stored result immediately
    ///
    /// # Errors
    ///
    /// Returns an error if joining fails or the tool execution failed.
    pub async fn join_tool(&self, handle: ToolHandle) -> ToolResult<JsonValue> {
        match handle {
            ToolHandle::Async(h) => {
                let mut inner = self.inner.lock().await;
                inner.task_ctx.join(h).await.map_err(Into::into)
            }
            ToolHandle::Sync(result) => Ok(result),
        }
    }

    /// Sleep for a duration (durable - survives restarts).
    ///
    /// # Errors
    ///
    /// Returns an error if the sleep operation fails.
    pub async fn sleep_for(&self, name: &str, duration: Duration) -> ToolResult<()> {
        let mut inner = self.inner.lock().await;
        inner
            .task_ctx
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
        &self,
        event_name: &str,
        timeout: Option<Duration>,
    ) -> ToolResult<T> {
        let mut inner = self.inner.lock().await;
        inner
            .task_ctx
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
        let inner = self.inner.lock().await;
        inner
            .task_ctx
            .emit_event(event_name, payload)
            .await
            .map_err(Into::into)
    }

    /// Generate a durable random value in [0, 1).
    ///
    /// # Errors
    ///
    /// Returns an error if generating the random value fails.
    pub async fn rand(&self) -> ToolResult<f64> {
        let mut inner = self.inner.lock().await;
        inner.task_ctx.rand().await.map_err(Into::into)
    }

    /// Get the current time as a durable checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if getting the current time fails.
    pub async fn now(&self) -> ToolResult<DateTime<Utc>> {
        let mut inner = self.inner.lock().await;
        inner.task_ctx.now().await.map_err(Into::into)
    }

    /// Generate a durable UUID v7.
    ///
    /// # Errors
    ///
    /// Returns an error if generating the UUID fails.
    pub async fn uuid7(&self) -> ToolResult<Uuid> {
        let mut inner = self.inner.lock().await;
        inner.task_ctx.uuid7().await.map_err(Into::into)
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
    pub async fn inference(&self, params: ClientInferenceParams) -> ToolResult<InferenceResponse> {
        let step_name = format!(
            "inference:{}",
            params
                .function_name
                .as_deref()
                .or(params.model_name.as_deref())
                .unwrap_or("unknown")
        );

        let mut inner = self.inner.lock().await;
        inner
            .task_ctx
            .step(&step_name, params, |params, state| async move {
                state
                    .t0_client
                    .inference(params)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .await
            .map_err(Into::into)
    }
}

/// Simplified context for `SimpleTool` execution.
///
/// `SimpleTools` run inside a `TaskTool`'s `step()` checkpoint, so they don't
/// have access to checkpointing operations themselves. They can access the
/// database pool for queries and the TensorZero client for inference calls.
pub struct SimpleToolContext<'a> {
    pool: &'a PgPool,
    t0_client: &'a Arc<dyn TensorZeroClient>,
}

impl<'a> SimpleToolContext<'a> {
    /// Create a new simple tool context.
    pub fn new(pool: &'a PgPool, t0_client: &'a Arc<dyn TensorZeroClient>) -> Self {
        Self { pool, t0_client }
    }

    /// Get a reference to the database pool.
    pub fn pool(&self) -> &PgPool {
        self.pool
    }

    /// Get the TensorZero client for direct access to all client operations.
    ///
    /// This provides access to inference, autopilot events, and other client operations.
    /// Note: `SimpleTools` run inside a `TaskTool`'s `step()`, so client calls
    /// are already checkpointed by the outer step.
    pub fn client(&self) -> &Arc<dyn TensorZeroClient> {
        self.t0_client
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
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        self.t0_client.inference(params).await
    }
}
