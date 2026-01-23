use durable::{DurableBuilder, DurableError, SpawnOptions, SpawnResult, Worker, WorkerOptions};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value as JsonValue;
use sqlx::{Executor, PgPool, Postgres};
use std::sync::Arc;
use tensorzero::Tool;
use tokio::sync::RwLock;
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
/// // Create executor
/// let database_url: SecretString = std::env::var("DATABASE_URL")?.into();
/// let executor = ToolExecutor::builder()
///     .database_url(database_url)
///     .queue_name("tools")
///     .t0_client(t0_client)
///     .build()
///     .await?;
///
/// // Register tools (pass instances)
/// executor.register_task_tool_instance(ResearchTool).await?;
/// executor.register_simple_tool_instance(SearchTool).await?;
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
pub struct ToolExecutor {
    /// The durable client for task management.
    durable: DurableClient,
    /// The tool registry (wrapped in `RwLock` for thread-safe registration).
    registry: Arc<RwLock<ToolRegistry>>,
}

impl ToolExecutor {
    /// Create a new executor with default settings.
    ///
    /// # Arguments
    ///
    /// * `database_url` - Database connection URL (as a `SecretString` for security)
    /// * `queue_name` - Name of the durable queue for tool tasks
    /// * `t0_client` - The TensorZero client for inference and autopilot calls
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails.
    pub async fn new(
        database_url: SecretString,
        queue_name: &str,
        t0_client: Arc<dyn TensorZeroClient>,
    ) -> anyhow::Result<Self> {
        Self::builder()
            .database_url(database_url)
            .queue_name(queue_name)
            .t0_client(t0_client)
            .build()
            .await
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> ToolExecutorBuilder {
        ToolExecutorBuilder::new()
    }

    /// Register a `TaskTool` instance.
    ///
    /// This registers the tool with both the tool registry and the durable
    /// client (so it can be executed by workers).
    ///
    /// # Errors
    ///
    /// Returns `ToolError::DuplicateToolName` if a tool with the same name is already registered.
    /// Returns `ToolError::SchemaGeneration` if the tool's parameter schema generation fails.
    pub async fn register_task_tool_instance<T: TaskTool>(
        &self,
        tool: T,
    ) -> Result<&Self, ToolError> {
        let tool = {
            let mut registry = self.registry.write().await;
            registry.register_task_tool_instance(tool)?
        };

        self.durable
            .register_instance(TaskToolAdapter::new(tool))
            .await?;

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
    /// Returns `ToolError::SchemaGeneration` if the tool's parameter schema generation fails.
    pub async fn register_simple_tool_instance<T: SimpleTool>(
        &self,
        tool: T,
    ) -> Result<&Self, ToolError> {
        let mut registry = self.registry.write().await;
        registry.register_simple_tool_instance(tool)?;
        Ok(self)
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
        {
            let registry = self.registry.read().await;
            registry.validate_params(tool_name, &llm_params, &side_info)?;
        }

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
        {
            let registry = self.registry.read().await;
            registry.validate_params(tool_name, &llm_params, &side_info)?;
        }

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
    pub async fn tool_definitions(&self) -> Result<Vec<Tool>, ToolError> {
        let registry = self.registry.read().await;
        registry.iter().map(Tool::try_from).collect()
    }

    /// Get a reference to the tool registry.
    ///
    /// Note: This returns an `Arc<RwLock<ToolRegistry>>` which requires
    /// async locking to access. For most use cases, prefer the helper
    /// methods on `ToolExecutor`.
    pub fn registry(&self) -> Arc<RwLock<ToolRegistry>> {
        self.registry.clone()
    }

    /// Get a reference to the underlying durable client.
    pub fn durable(&self) -> &DurableClient {
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
pub struct ToolExecutorBuilder {
    database_url: Option<SecretString>,
    pool: Option<PgPool>,
    queue_name: String,
    default_max_attempts: u32,
    t0_client: Option<Arc<dyn TensorZeroClient>>,
}

impl ToolExecutorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            database_url: None,
            pool: None,
            queue_name: "tools".to_string(),
            default_max_attempts: 5,
            t0_client: None,
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

    /// Build the [`ToolExecutor`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails or if the TensorZero
    /// client was not provided.
    pub async fn build(self) -> anyhow::Result<ToolExecutor> {
        // TensorZero client is required
        let t0_client = self
            .t0_client
            .ok_or_else(|| anyhow::anyhow!("t0_client is required"))?;

        // Create the tool registry
        let registry = Arc::new(RwLock::new(ToolRegistry::new()));

        // Get or create the pool
        let pool = if let Some(pool) = self.pool {
            pool
        } else {
            let url = self.database_url.ok_or_else(|| {
                anyhow::anyhow!("No database URL configured. Set database_url() or pool()")
            })?;
            PgPool::connect(url.expose_secret()).await?
        };

        // Create the app context with the pool and TensorZero client
        let app_ctx = ToolAppState::new(pool.clone(), registry.clone(), t0_client);

        // Build the durable client with the app context
        let durable = DurableBuilder::new()
            .pool(pool)
            .queue_name(&self.queue_name)
            .default_max_attempts(self.default_max_attempts)
            .build_with_state(app_ctx)
            .await?;

        Ok(ToolExecutor { durable, registry })
    }
}

impl Default for ToolExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
