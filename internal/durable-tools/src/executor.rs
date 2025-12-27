use durable::{DurableBuilder, DurableError, SpawnOptions, SpawnResult, Worker, WorkerOptions};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;
use tensorzero::Tool;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::context::{DurableClient, ToolAppState};
use crate::error::ToolError;
use crate::inference::InferenceClient;
use crate::registry::ToolRegistry;
use crate::simple_tool::SimpleTool;
use crate::task_tool::{TaskTool, TaskToolAdapter};
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
/// // Create inference client
/// let inference_client = http_gateway_client(Url::parse("http://localhost:3000")?)?;
///
/// // Create executor
/// let database_url: SecretString = std::env::var("DATABASE_URL")?.into();
/// let executor = ToolExecutor::builder()
///     .database_url(database_url)
///     .queue_name("tools")
///     .inference_client(inference_client)
///     .build()
///     .await?;
///
/// // Register tools
/// executor.register_task_tool::<ResearchTool>().await;
/// executor.register_simple_tool::<SearchTool>().await;
///
/// // Spawn a tool execution (without side info)
/// let episode_id = Uuid::now_v7();
/// executor.spawn_tool::<ResearchTool>(params, (), episode_id).await?;
///
/// // Spawn with side info
/// executor.spawn_tool::<GitHubTool>(params, credentials, episode_id).await?;
///
/// // Start a worker
/// let worker = executor.start_worker(WorkerOptions::default()).await;
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
    /// * `inference_client` - The inference client for TensorZero calls
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails.
    pub async fn new(
        database_url: SecretString,
        queue_name: &str,
        inference_client: Arc<dyn InferenceClient>,
    ) -> anyhow::Result<Self> {
        Self::builder()
            .database_url(database_url)
            .queue_name(queue_name)
            .inference_client(inference_client)
            .build()
            .await
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> ToolExecutorBuilder {
        ToolExecutorBuilder::new()
    }

    /// Register a `TaskTool`.
    ///
    /// This registers the tool with both the tool registry and the durable
    /// client (so it can be executed by workers).
    ///
    /// # Errors
    ///
    /// Returns `ToolError::DuplicateToolName` if a tool with the same name is already registered.
    /// Returns `ToolError::SchemaGeneration` if the tool's parameter schema generation fails.
    pub async fn register_task_tool<T: TaskTool>(&self) -> Result<&Self, ToolError> {
        // Register with tool registry
        {
            let mut registry = self.registry.write().await;
            registry.register_task_tool::<T>()?;
        }

        // Register the adapter with durable
        self.durable.register::<TaskToolAdapter<T>>().await?;

        Ok(self)
    }

    /// Register a `SimpleTool`.
    ///
    /// `SimpleTools` don't need to be registered with the durable client
    /// since they run inside `TaskTool` steps.
    ///
    /// # Errors
    ///
    /// Returns `ToolError::DuplicateToolName` if a tool with the same name is already registered.
    /// Returns `ToolError::SchemaGeneration` if the tool's parameter schema generation fails.
    pub async fn register_simple_tool<T: SimpleTool + Default>(&self) -> Result<&Self, ToolError> {
        let mut registry = self.registry.write().await;
        registry.register_simple_tool::<T>()?;
        Ok(self)
    }

    /// Spawn a `TaskTool` execution.
    ///
    /// # Arguments
    ///
    /// * `llm_params` - The LLM-provided parameters
    /// * `side_info` - Side information (hidden from LLM), use `()` if not needed
    /// * `episode_id` - The episode ID for this execution
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the tool fails.
    pub async fn spawn_tool<T: TaskTool>(
        &self,
        llm_params: T::LlmParams,
        side_info: T::SideInfo,
        episode_id: Uuid,
    ) -> anyhow::Result<SpawnResult> {
        let wrapped = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };
        self.durable
            .spawn_with_options::<TaskToolAdapter<T>>(wrapped, SpawnOptions::default())
            .await
            .map_err(Into::into)
    }

    /// Spawn a `TaskTool` execution with custom spawn options.
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the tool fails.
    pub async fn spawn_tool_with_options<T: TaskTool>(
        &self,
        llm_params: T::LlmParams,
        side_info: T::SideInfo,
        episode_id: Uuid,
        options: SpawnOptions,
    ) -> anyhow::Result<SpawnResult> {
        let wrapped = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };
        self.durable
            .spawn_with_options::<TaskToolAdapter<T>>(wrapped, options)
            .await
            .map_err(Into::into)
    }

    /// Spawn a tool by name with JSON parameters.
    ///
    /// This allows dynamic tool invocation without knowing the concrete type.
    /// Side info defaults to `null` (compatible with `SideInfo = ()`).
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the tool fails.
    pub async fn spawn_tool_by_name(
        &self,
        tool_name: &str,
        llm_params: JsonValue,
        episode_id: Uuid,
    ) -> anyhow::Result<SpawnResult> {
        self.spawn_tool_by_name_with_side_info(
            tool_name,
            llm_params,
            serde_json::json!(null),
            episode_id,
        )
        .await
    }

    /// Spawn a tool by name with JSON parameters and explicit side info.
    ///
    /// This allows dynamic tool invocation without knowing the concrete type.
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the tool fails.
    pub async fn spawn_tool_by_name_with_side_info(
        &self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        episode_id: Uuid,
    ) -> anyhow::Result<SpawnResult> {
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
        registry.to_tensorzero_tools()
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
    inference_client: Option<Arc<dyn InferenceClient>>,
}

impl ToolExecutorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            database_url: None,
            pool: None,
            queue_name: "tools".to_string(),
            default_max_attempts: 5,
            inference_client: None,
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

    /// Set the inference client for TensorZero calls (required).
    #[must_use]
    pub fn inference_client(mut self, client: Arc<dyn InferenceClient>) -> Self {
        self.inference_client = Some(client);
        self
    }

    /// Build the [`ToolExecutor`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails or if the inference
    /// client was not provided.
    pub async fn build(self) -> anyhow::Result<ToolExecutor> {
        // Inference client is required
        let inference_client = self
            .inference_client
            .ok_or_else(|| anyhow::anyhow!("inference_client is required"))?;

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

        // Create the app context with the pool and inference client
        let app_ctx = ToolAppState::new(pool.clone(), registry.clone(), inference_client);

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
