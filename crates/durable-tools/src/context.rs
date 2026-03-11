use chrono::{DateTime, Utc};
use durable::{
    Durable, HeartbeatHandle, Heartbeater, SpawnOptions, StepState, TaskContext, TaskHandle,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::{ClientInferenceParams, InferenceResponse};
use tensorzero_core::cache::CacheParamsOptions;
use tensorzero_core::embeddings::{EmbeddingEncodingFormat, EmbeddingInput};
use tensorzero_core::endpoints::embeddings::{EmbeddingResponse, EmbeddingsParams};
use tensorzero_core::endpoints::inference::InferenceCredentials;
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
pub type DurableClient<S> = Durable<ToolAppState<S>>;

/// Application state passed to all tools via durable's `State` type parameter.
///
/// This is cloned and passed to each task execution by the durable worker.
#[derive(Clone)]
pub struct ToolAppState<S = ()> {
    /// Database connection pool for database operations.
    pool: PgPool,
    /// Tool registry for looking up and calling other tools.
    tool_registry: Arc<ToolRegistry>,
    /// TensorZero client for calling inference and autopilot operations.
    t0_client: Arc<dyn TensorZeroClient>,
    extra_state: S,
}

impl<S> ToolAppState<S> {
    /// Create a new application state.
    pub fn new(
        pool: PgPool,
        tool_registry: Arc<ToolRegistry>,
        t0_client: Arc<dyn TensorZeroClient>,
        extra_state: S,
    ) -> Self {
        Self {
            pool,
            tool_registry,
            t0_client,
            extra_state,
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

    /// Get a reference to the extra state.
    pub fn extra_state(&self) -> &S {
        &self.extra_state
    }
}

/// Context provided to `TaskTool` execution.
///
/// Wraps durable's `TaskContext` and provides access to the application context
/// along with helper methods for calling other tools and checkpointing operations.
pub struct ToolContext<S: Clone + Send + Sync + 'static = ()> {
    task_ctx: TaskContext<ToolAppState<S>>,
    app_state: ToolAppState<S>,
    episode_id: Uuid,
    /// Counter for generating unique tool call identifiers.
    tool_call_counter: u32,
}

impl<S: Clone + Send + Sync + 'static> ToolContext<S> {
    /// Create a new tool context.
    pub fn new(
        task_ctx: TaskContext<ToolAppState<S>>,
        app_ctx: ToolAppState<S>,
        episode_id: Uuid,
    ) -> Self {
        Self {
            task_ctx,
            app_state: app_ctx,
            episode_id,
            tool_call_counter: 0,
        }
    }

    /// Get the heartbeat handle for extending the task lease.
    ///
    /// This returns the real `HeartbeatHandle` from the durable task context.
    /// Use this in `TaskTool::execute()` when you need to pass a heartbeater
    /// to non-durable code (e.g., `TensorZeroClient::action()`).
    pub fn heartbeater(&self) -> HeartbeatHandle {
        self.task_ctx.heartbeat_handle()
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

    /// Get the inference client for direct access to all client operations.
    ///
    /// This provides access to inference, autopilot events, and other client operations.
    /// For durability, wrap client calls in `ctx.step()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Send a tool result to autopilot (checkpointed)
    /// ctx.step("send_result", params, |params, step_state| async move {
    ///     step_state.state.t0_client()
    ///         .create_autopilot_event(session_id, request)
    ///         .await
    ///         .map_err(|e| anyhow::anyhow!("{e}"))
    /// }).await?;
    /// ```
    pub fn client(&self) -> Arc<dyn TensorZeroClient> {
        self.app_state.t0_client.clone()
    }

    /// Get a reference to the tool registry.
    ///
    /// Use this to iterate over tools and convert them to TensorZero tool definitions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = ctx.registry();
    /// let tools: Result<Vec<Tool>, _> = registry.iter()
    ///     .filter(|t| !t.is_durable())
    ///     .map(Tool::try_from)
    ///     .collect();
    /// ```
    pub fn registry(&self) -> &ToolRegistry {
        &self.app_state.tool_registry
    }

    /// Get access to the 'extra_side' provided when building the 'DurableClient'
    pub fn extra_state(&self) -> &S {
        &self.app_state.extra_state
    }

    /// Get mutable access to the underlying durable `TaskContext`.
    ///
    /// Use this for advanced operations not exposed by `ToolContext`.
    pub fn task_ctx(&mut self) -> &mut TaskContext<ToolAppState<S>> {
        &mut self.task_ctx
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
        f: fn(P, StepState<ToolAppState<S>>) -> Fut,
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
        &mut self,
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
        &mut self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        options: SpawnOptions,
    ) -> ToolResult<ToolHandle> {
        let registry = &self.app_state.tool_registry;
        // Validate params before spawning
        registry.validate_params(tool_name, &llm_params, &side_info)?;
        let is_durable =
            registry
                .is_durable(tool_name)
                .ok_or_else(|| NonControlToolError::ToolNotFound {
                    name: tool_name.to_string(),
                })?;

        if is_durable {
            // TaskTool: spawn as subtask
            let call_id = self.next_tool_call_id();
            let wrapped_params = TaskToolParams {
                llm_params,
                side_info,
                episode_id: self.episode_id,
            };
            let wrapped_params = serde_json::to_value(wrapped_params)?;
            let spawn_name = format!("spawn:{tool_name}:{call_id}");
            let handle: TaskHandle<JsonValue> = self
                .spawn_subtask_by_name(&spawn_name, tool_name, wrapped_params, options)
                .await?;
            Ok(ToolHandle::Async(handle))
        } else {
            // SimpleTool: execute immediately (still checkpointed via step)
            let result = self
                .call_simple_tool(tool_name, llm_params, side_info)
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
    pub async fn join_tool(&mut self, handle: ToolHandle) -> ToolResult<JsonValue> {
        match handle {
            ToolHandle::Async(h) => self.task_ctx.join(h).await.map_err(Into::into),
            ToolHandle::Sync(result) => Ok(result),
        }
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
        options: SpawnOptions,
    ) -> ToolResult<TaskHandle<T>> {
        let handle: TaskHandle<T> = self
            .task_ctx
            .spawn_by_name(name, task_name, params, options)
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
            |(tool_name, task_id, call_id, llm_params, side_info), step_state| async move {
                let heartbeater = step_state.heartbeater.clone();
                let state = &step_state.state;

                // Create SimpleToolContext
                let simple_ctx =
                    SimpleToolContext::new(&state.pool, &state.t0_client, &heartbeater);

                // Generate idempotency key using task_id and call_id
                let idempotency_key = format!("{task_id}:{tool_name}:{call_id}");

                // Get the simple tool from registry
                let simple_tool =
                    state
                        .tool_registry
                        .get_simple_tool(&tool_name)
                        .ok_or_else(|| NonControlToolError::ToolNotFound {
                            name: tool_name.clone(),
                        })?;

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

    /// Extend the task's lease to prevent timeout.
    ///
    /// Use this for long-running operations that don't naturally checkpoint.
    /// Each `step()` call also extends the lease automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if the heartbeat fails.
    pub async fn heartbeat(&self, duration: Option<Duration>) -> ToolResult<()> {
        self.task_ctx.heartbeat(duration).await.map_err(Into::into)
    }

    /// Call TensorZero inference with full parameter control.
    ///
    /// This is a checkpointed operation - results are cached on restart.
    /// Streaming inference is not supported and will return an error.
    ///
    /// The inference step will automatically fail if the model returns an empty output:
    /// - For chat functions: no content blocks in the response
    /// - For JSON functions: `raw` field is `None`
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
    /// Returns an error if the inference call fails or if the output is empty.
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

        self.step(
            &step_name,
            (params, step_name.clone()),
            |(params, step_name), step_state| async move {
                let response = step_state
                    .state
                    .t0_client
                    .inference(params)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

                // Check for empty output inside the step so it can be retried
                let is_empty = match &response {
                    InferenceResponse::Chat(chat) => chat.content.is_empty(),
                    InferenceResponse::Json(json) => json.output.raw.is_none(),
                };

                if is_empty {
                    return Err(anyhow::anyhow!(
                        "Inference `{step_name}` returned empty output"
                    ));
                }

                Ok(response)
            },
        )
        .await
    }

    /// Compute embeddings with the configured embedding model.
    ///
    /// This is a checkpointed operation - results are cached on restart.
    ///
    /// # Errors
    ///
    /// Returns an error if the embeddings call fails.
    pub async fn embeddings(&mut self, params: EmbeddingsParams) -> ToolResult<EmbeddingResponse> {
        let step_name = format!("embeddings:{}", params.model_name);

        let serializable = SerializableEmbeddingsParams {
            input: params.input,
            model_name: params.model_name,
            dimensions: params.dimensions,
            encoding_format: params.encoding_format,
            dryrun: params.dryrun,
            credentials: params.credentials,
            cache_options: params.cache_options,
            include_raw_response: params.include_raw_response,
        };

        self.step(&step_name, serializable, embeddings_step).await
    }
}

/// Simplified context for `SimpleTool` execution.
///
/// `SimpleTools` run inside a `TaskTool`'s `step()` checkpoint, so they don't
/// have access to checkpointing operations themselves. They can access the
/// database pool for queries and the TensorZero client for inference calls.
///
/// A heartbeater is available for long-running operations (e.g., evaluations)
/// that need to extend the task lease to prevent timeout.
pub struct SimpleToolContext<'a> {
    pool: &'a PgPool,
    t0_client: &'a Arc<dyn TensorZeroClient>,
    heartbeater: &'a Arc<dyn Heartbeater>,
}

impl<'a> SimpleToolContext<'a> {
    /// Create a new simple tool context.
    pub fn new(
        pool: &'a PgPool,
        t0_client: &'a Arc<dyn TensorZeroClient>,
        heartbeater: &'a Arc<dyn Heartbeater>,
    ) -> Self {
        Self {
            pool,
            t0_client,
            heartbeater,
        }
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

    /// Get the heartbeater for extending the task lease during long-running operations.
    pub fn heartbeater(&self) -> &Arc<dyn Heartbeater> {
        self.heartbeater
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

    /// Compute embeddings with the configured embedding model.
    ///
    /// Note: `SimpleTools` run inside a `TaskTool`'s `step()`, so this call
    /// is already checkpointed by the outer step.
    ///
    /// # Errors
    ///
    /// Returns an error if the embeddings call fails.
    pub async fn embeddings(
        &self,
        params: EmbeddingsParams,
    ) -> Result<EmbeddingResponse, TensorZeroClientError> {
        self.t0_client.embeddings(params).await
    }
}

/// Serializable version of `EmbeddingsParams` for durable checkpointing.
///
/// Credentials are excluded from serialization (`#[serde(skip)]`) so they
/// are never persisted to the checkpoint database. On replay, the step result
/// is returned from the checkpoint and the function is not re-executed, so
/// credentials are not needed.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableEmbeddingsParams {
    input: EmbeddingInput,
    model_name: String,
    dimensions: Option<u32>,
    encoding_format: EmbeddingEncodingFormat,
    dryrun: Option<bool>,
    #[serde(skip)]
    credentials: InferenceCredentials,
    cache_options: CacheParamsOptions,
    include_raw_response: bool,
}

async fn embeddings_step<S: Send + Sync + 'static>(
    serializable: SerializableEmbeddingsParams,
    step_state: StepState<ToolAppState<S>>,
) -> anyhow::Result<EmbeddingResponse> {
    let params = EmbeddingsParams {
        input: serializable.input,
        model_name: serializable.model_name,
        dimensions: serializable.dimensions,
        encoding_format: serializable.encoding_format,
        dryrun: serializable.dryrun,
        credentials: serializable.credentials,
        cache_options: serializable.cache_options,
        include_raw_response: serializable.include_raw_response,
    };
    step_state
        .state
        .t0_client
        .embeddings(params)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))
}
