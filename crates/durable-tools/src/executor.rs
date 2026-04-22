use durable::{DurableBuilder, DurableError, SpawnOptions, SpawnResult, Worker, WorkerOptions};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value as JsonValue;
use sqlx::{Executor, PgPool, Postgres};
use std::sync::Arc;
use tensorzero::Tool;
use uuid::Uuid;

use crate::context::{DurableClient, ToolAppState};
use crate::error::ToolError;
use crate::registry::ToolRegistry;
use crate::simple_tool::SimpleTool;
use crate::task_tool::{TaskTool, TaskToolAdapter};
use crate::tensorzero_client::TensorZeroClient;
use durable_tools_spawn::TaskToolParams;

/// High-level orchestrator for tool execution.
///
/// `ToolExecutor` manages both the durable client and tool registry,
/// providing a unified interface for:
/// - Registering tools (both `TaskTools` and `SimpleTools`)
/// - Spawning tool executions
/// - Starting workers
/// - Getting tool definitions for LLM function calling
///
/// # Example
///
/// ```ignore
/// use durable_tools::{ToolExecutor, TaskTool, SimpleTool, WorkerOptions, http_gateway_client};
/// use secrecy::SecretString;
/// use url::Url;
///
/// // Create TensorZero client
/// let t0_client = http_gateway_client(Url::parse("http://localhost:3000")?)?;
///
/// // Create executor with tools registered at build time
/// let database_url: SecretString = std::env::var("DATABASE_URL")?.into();
/// let executor = ToolExecutor::builder()
///     .database_url(database_url)
///     .queue_name("tools")
///     .t0_client(t0_client)
///     .register_task_tool_instance(ResearchTool)?
///     .register_simple_tool_instance(SearchTool)?
///     .build()
///     .await?;
///
/// // Spawn a tool execution by name
/// let episode_id = Uuid::now_v7();
/// executor.spawn_tool_by_name(
///     "research",
///     serde_json::json!({"topic": "rust"}),
///     serde_json::json!(null),  // No side info
///     episode_id,
/// ).await?;
///
/// // Spawn with side info
/// executor.spawn_tool_by_name(
///     "github",
///     serde_json::to_value(params)?,
///     serde_json::to_value(credentials)?,
///     episode_id,
/// ).await?;
///
/// // Start a worker
/// let worker = executor.start_worker(WorkerOptions::default()).await?;
/// ```
pub struct ToolExecutor<S: Clone + Send + Sync + 'static = ()> {
    /// The durable client for task management.
    durable: DurableClient<S>,
    /// The tool registry (immutable after construction).
    registry: Arc<ToolRegistry>,
}

impl<S: Clone + Send + Sync + 'static> ToolExecutor<S> {
    /// Create a new executor with default settings.
    ///
    /// # Arguments
    ///
    /// * `database_url` - Database connection URL (as a `SecretString` for security)
    /// * `queue_name` - Name of the durable queue for tool tasks
    /// * `t0_client` - The TensorZero client for inference and autopilot calls
    /// * `extra_state` - Extra state to pass to the executor, which will be made available to all tools.
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails.
    pub async fn new(
        database_url: SecretString,
        queue_name: &str,
        t0_client: Arc<dyn TensorZeroClient>,
        extra_state: S,
    ) -> anyhow::Result<Self> {
        Self::builder(extra_state)
            .database_url(database_url)
            .queue_name(queue_name)
            .t0_client(t0_client)
            .build()
            .await
    }

    /// Create a builder for custom configuration.
    pub fn builder(extra_state: S) -> ToolExecutorBuilder<S> {
        ToolExecutorBuilder::new(extra_state)
    }

    /// Spawn a tool by name with JSON parameters.
    ///
    /// This allows dynamic tool invocation without knowing the concrete type.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The registered name of the tool to spawn
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    /// * `episode_id` - The episode ID for this execution
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found, parameters are invalid, or spawning fails.
    pub async fn spawn_tool_by_name(
        &self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        episode_id: Uuid,
    ) -> anyhow::Result<SpawnResult> {
        // Validate params before spawning
        self.registry
            .validate_params(tool_name, &llm_params, &side_info)?;

        let wrapped_params = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };

        self.durable
            .spawn_by_name(
                tool_name,
                serde_json::to_value(wrapped_params)?,
                SpawnOptions::default(),
            )
            .await
            .map_err(Into::into)
    }

    /// Spawn a tool by name using a custom executor (e.g., a transaction).
    ///
    /// This allows dynamic tool invocation without knowing the concrete type,
    /// while atomically enqueuing as part of a larger transaction.
    ///
    /// # Arguments
    ///
    /// * `executor` - The executor to use (e.g., `&mut *tx` for a transaction)
    /// * `tool_name` - The registered name of the tool to spawn
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    /// * `episode_id` - The episode ID for this execution
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found, parameters are invalid, or spawning fails.
    pub async fn spawn_tool_by_name_with<'e, E>(
        &self,
        executor: E,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        episode_id: Uuid,
    ) -> anyhow::Result<SpawnResult>
    where
        E: Executor<'e, Database = Postgres>,
    {
        // Validate params before spawning
        self.registry
            .validate_params(tool_name, &llm_params, &side_info)?;

        let wrapped_params = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };

        self.durable
            .spawn_by_name_with(
                executor,
                tool_name,
                serde_json::to_value(wrapped_params)?,
                SpawnOptions::default(),
            )
            .await
            .map_err(Into::into)
    }

    /// Start a worker that processes tool tasks.
    pub async fn start_worker(&self, options: WorkerOptions) -> Result<Worker, DurableError> {
        self.durable.start_worker(options).await
    }

    /// Get tool definitions in TensorZero format.
    ///
    /// # Errors
    ///
    /// Returns an error if a tool's parameter schema generation or serialization fails.
    pub fn tool_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        self.registry.iter().map(Tool::try_from).collect()
    }

    /// Get a reference to the tool registry.
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Get a reference to the underlying durable client.
    pub fn durable(&self) -> &DurableClient<S> {
        &self.durable
    }

    /// Get the database pool.
    pub fn pool(&self) -> &PgPool {
        self.durable.pool()
    }

    /// Get the queue name.
    pub fn queue_name(&self) -> &str {
        self.durable.queue_name()
    }
}

/// Builder for creating a [`ToolExecutor`] with custom configuration.
///
/// Tools must be registered on the builder before calling [`build`](Self::build):
/// - Use [`register_task_tool_instance`](Self::register_task_tool_instance) for durable tools
/// - Use [`register_simple_tool_instance`](Self::register_simple_tool_instance) for lightweight tools
pub struct ToolExecutorBuilder<S: Clone + Send + Sync + 'static = ()> {
    database_url: Option<SecretString>,
    pool: Option<PgPool>,
    queue_name: String,
    default_max_attempts: u32,
    t0_client: Option<Arc<dyn TensorZeroClient>>,
    extra_state: S,
    /// Tool registry — tools are registered immediately.
    registry: ToolRegistry,
    /// Durable builder — task tools are registered immediately.
    durable_builder: DurableBuilder<ToolAppState<S>>,
}

impl<S: Clone + Send + Sync + 'static> ToolExecutorBuilder<S> {
    /// Create a new builder with default settings.
    pub fn new(extra_state: S) -> Self {
        Self {
            database_url: None,
            pool: None,
            queue_name: "tools".to_string(),
            default_max_attempts: 5,
            t0_client: None,
            extra_state,
            registry: ToolRegistry::new(),
            durable_builder: DurableBuilder::new(),
        }
    }

    /// Set the database URL (will create a new connection pool).
    #[must_use]
    pub fn database_url(mut self, url: SecretString) -> Self {
        self.database_url = Some(url);
        self
    }

    /// Use an existing connection pool.
    #[must_use]
    pub fn pool(mut self, pool: PgPool) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Set the queue name (default: "tools").
    #[must_use]
    pub fn queue_name(mut self, name: impl Into<String>) -> Self {
        self.queue_name = name.into();
        self
    }

    /// Set the default max attempts for tool executions (default: 5).
    #[must_use]
    pub fn default_max_attempts(mut self, attempts: u32) -> Self {
        self.default_max_attempts = attempts;
        self
    }

    /// Set the TensorZero client (required).
    #[must_use]
    pub fn t0_client(mut self, client: Arc<dyn TensorZeroClient>) -> Self {
        self.t0_client = Some(client);
        self
    }

    /// Register a `TaskTool` instance.
    ///
    /// This registers the tool with both the tool registry and the durable
    /// client immediately.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::DuplicateToolName` if a tool with the same name is already registered.
    pub fn register_task_tool_instance<T: TaskTool<ExtraState = S>>(
        mut self,
        tool: T,
    ) -> Result<Self, ToolError> {
        let tool = self.registry.register_task_tool_instance(tool)?;
        self.durable_builder = self
            .durable_builder
            .register_instance(TaskToolAdapter::new(tool))?;
        Ok(self)
    }

    /// Register a `SimpleTool` instance.
    ///
    /// `SimpleTools` don't need to be registered with the durable client
    /// since they run inside `TaskTool` steps.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::DuplicateToolName` if a tool with the same name is already registered.
    pub fn register_simple_tool_instance<T: SimpleTool>(
        mut self,
        tool: T,
    ) -> Result<Self, ToolError> {
        self.registry.register_simple_tool_instance(tool)?;
        Ok(self)
    }

    /// Register a `TaskTool` instance (mutable reference variant).
    ///
    /// Like [`register_task_tool_instance`](Self::register_task_tool_instance) but takes
    /// `&mut self` instead of `self`, useful when registering tools from a visitor or loop.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::DuplicateToolName` if a tool with the same name is already registered.
    pub fn push_task_tool_instance<T: TaskTool<ExtraState = S>>(
        &mut self,
        tool: T,
    ) -> Result<(), ToolError> {
        let tool = self.registry.register_task_tool_instance(tool)?;
        self.durable_builder
            .register_instance_mut(TaskToolAdapter::new(tool))?;
        Ok(())
    }

    /// Register a `SimpleTool` instance (mutable reference variant).
    ///
    /// Like [`register_simple_tool_instance`](Self::register_simple_tool_instance) but takes
    /// `&mut self` instead of `self`, useful when registering tools from a visitor or loop.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::DuplicateToolName` if a tool with the same name is already registered.
    pub fn push_simple_tool_instance<T: SimpleTool>(&mut self, tool: T) -> Result<(), ToolError> {
        self.registry.register_simple_tool_instance(tool)?;
        Ok(())
    }

    /// Build the [`ToolExecutor`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails or if the TensorZero
    /// client was not provided.
    pub async fn build(self) -> anyhow::Result<ToolExecutor<S>> {
        // TensorZero client is required
        let t0_client = self
            .t0_client
            .ok_or_else(|| anyhow::anyhow!("t0_client is required"))?;

        // Get or create the pool
        let pool = if let Some(pool) = self.pool {
            pool
        } else {
            let url = self.database_url.ok_or_else(|| {
                anyhow::anyhow!("No database URL configured. Set database_url() or pool()")
            })?;
            PgPool::connect(url.expose_secret()).await?
        };

        let registry = Arc::new(self.registry);

        // Create the app context with the pool and TensorZero client
        let app_ctx =
            ToolAppState::new(pool.clone(), registry.clone(), t0_client, self.extra_state);

        // Finalize the durable client with pool, queue config, and app state
        let durable = self
            .durable_builder
            .pool(pool)
            .queue_name(&self.queue_name)
            .default_max_attempts(self.default_max_attempts)
            .build_with_state(app_ctx)
            .await?;

        Ok(ToolExecutor { durable, registry })
    }
}

impl<S: Default + Clone + Send + Sync + 'static> Default for ToolExecutorBuilder<S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}
