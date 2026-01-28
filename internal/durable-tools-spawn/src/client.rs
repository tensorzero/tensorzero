//! Lightweight spawn-only client.

use durable::{Durable, DurableBuilder, SpawnOptions, SpawnResult};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value as JsonValue;
use sqlx::{Executor, PgPool, Postgres};
use uuid::Uuid;

use crate::error::SpawnError;
use crate::params::TaskToolParams;

/// A lightweight client for spawning durable tasks.
///
/// Unlike the full `ToolExecutor` in `durable-tools`, this client only
/// supports spawning tasks by name. It does not include tool registration,
/// workers, or inference capabilities.
///
/// # Example
///
/// ```ignore
/// use durable_tools_spawn::{SpawnClient, SpawnClientBuilder};
/// use secrecy::SecretString;
/// use uuid::Uuid;
///
/// let client = SpawnClient::builder()
///     .database_url(database_url)
///     .queue_name("tools")
///     .build()
///     .await?;
///
/// let episode_id = Uuid::now_v7();
/// client.spawn_tool_by_name(
///     "research",
///     serde_json::json!({"topic": "rust"}),
///     serde_json::json!(null),  // side_info
///     episode_id,
/// ).await?;
/// ```
pub struct SpawnClient {
    durable: Durable<()>,
}

impl SpawnClient {
    /// Create a builder for custom configuration.
    pub fn builder() -> SpawnClientBuilder {
        SpawnClientBuilder::new()
    }

    /// Spawn a task by name with JSON parameters.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The registered name of the tool to spawn
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    /// * `episode_id` - The episode ID for this execution
    /// * `options` - Options for spawning the task
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the task fails.
    pub async fn spawn_tool_by_name(
        &self,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        episode_id: Uuid,
        options: SpawnOptions,
    ) -> Result<SpawnResult, SpawnError> {
        let wrapped_params = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };

        self.durable
            .spawn_by_name_unchecked(tool_name, serde_json::to_value(wrapped_params)?, options)
            .await
            .map_err(Into::into)
    }

    /// Spawn a task by name using a custom executor (e.g., a transaction).
    ///
    /// This allows you to atomically enqueue a task as part of a larger transaction.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tx = client.pool().begin().await?;
    ///
    /// sqlx::query("INSERT INTO orders (id) VALUES ($1)")
    ///     .bind(order_id)
    ///     .execute(&mut *tx)
    ///     .await?;
    ///
    /// client.spawn_tool_by_name_with(
    ///     &mut *tx,
    ///     "process_order",
    ///     serde_json::json!({"order_id": order_id}),
    ///     serde_json::json!(null),
    ///     episode_id,
    /// ).await?;
    ///
    /// tx.commit().await?;
    /// ```
    ///
    /// # Arguments
    ///
    /// * `executor` - The executor to use (e.g., `&mut *tx` for a transaction)
    /// * `tool_name` - The registered name of the tool to spawn
    /// * `llm_params` - Parameters visible to the LLM
    /// * `side_info` - Hidden parameters (use `json!(null)` if not needed)
    /// * `episode_id` - The episode ID for this execution
    /// * `options` - Options for spawning the task
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the task fails.
    pub async fn spawn_tool_by_name_with<'e, E>(
        &self,
        executor: E,
        tool_name: &str,
        llm_params: JsonValue,
        side_info: JsonValue,
        episode_id: Uuid,
        options: SpawnOptions,
    ) -> Result<SpawnResult, SpawnError>
    where
        E: Executor<'e, Database = Postgres>,
    {
        let wrapped_params = TaskToolParams {
            llm_params,
            side_info,
            episode_id,
        };

        self.durable
            .spawn_by_name_unchecked_with(
                executor,
                tool_name,
                serde_json::to_value(wrapped_params)?,
                options,
            )
            .await
            .map_err(Into::into)
    }

    /// Get the queue name.
    pub fn queue_name(&self) -> &str {
        self.durable.queue_name()
    }

    /// Get the database pool.
    pub fn pool(&self) -> &PgPool {
        self.durable.pool()
    }

    /// Interrupt all durable tasks associated with a session ID.
    ///
    /// This queries for non-terminal tasks where `params->'side_info'->>'session_id'`
    /// matches the provided session ID, then interrupts each one. The durable framework's
    /// `cancel_task` function automatically cascades interruption to child tasks.
    ///
    /// # Returns
    ///
    /// Returns the number of tasks that were interrupted.
    ///
    /// # Errors
    ///
    /// Returns an error if any database operation fails.
    pub async fn interrupt_tasks_by_session_id(&self, session_id: Uuid) -> Result<u64, SpawnError> {
        let queue_name = self.queue_name();

        // Find all non-terminal tasks with this session_id
        // Using QueryBuilder for dynamic table name (queue_name is a trusted internal value)
        let mut query_builder: sqlx::QueryBuilder<Postgres> =
            sqlx::QueryBuilder::new("SELECT task_id FROM durable.t_");
        query_builder.push(queue_name);
        query_builder.push(" WHERE params->'side_info'->>'session_id' = ");
        query_builder.push_bind(session_id.to_string());
        query_builder.push(" AND state NOT IN ('completed', 'failed', 'cancelled')");

        let task_ids: Vec<(Uuid,)> = query_builder
            .build_query_as()
            .fetch_all(self.pool())
            .await?;

        // Interrupt each task (cascade to children handled by durable.cancel_task)
        for (task_id,) in &task_ids {
            sqlx::query("SELECT durable.cancel_task($1, $2)")
                .bind(queue_name)
                .bind(task_id)
                .execute(self.pool())
                .await?;
        }

        Ok(task_ids.len() as u64)
    }
}

/// Builder for creating a [`SpawnClient`].
pub struct SpawnClientBuilder {
    database_url: Option<SecretString>,
    pool: Option<PgPool>,
    queue_name: String,
}

impl SpawnClientBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            database_url: None,
            pool: None,
            queue_name: "tools".to_string(),
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

    /// Build the [`SpawnClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails.
    pub async fn build(self) -> Result<SpawnClient, SpawnError> {
        let pool = if let Some(pool) = self.pool {
            pool
        } else {
            let url = self.database_url.ok_or(SpawnError::MissingConfig(
                "No database URL configured. Set database_url() or pool()",
            ))?;
            PgPool::connect(url.expose_secret())
                .await
                .map_err(SpawnError::Database)?
        };

        let durable = DurableBuilder::new()
            .pool(pool)
            .queue_name(&self.queue_name)
            .build()
            .await?;

        Ok(SpawnClient { durable })
    }
}

impl Default for SpawnClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}
