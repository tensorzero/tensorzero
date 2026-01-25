//! Autopilot worker implementation.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use autopilot_tools::ToolVisitor;
use durable_tools::{
    SimpleTool, TaskTool, TensorZeroClient, ToolError, ToolExecutor, Worker, WorkerOptions,
};
use serde::Serialize;
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::wrapper::ClientSimpleToolWrapper;
use crate::wrapper::ClientTaskToolWrapper;

/// Configuration for the autopilot worker.
pub struct AutopilotWorkerConfig {
    /// Database pool for the durable task queue (shared with gateway).
    pub pool: PgPool,
    /// Queue name for durable tasks (default: "autopilot").
    pub queue_name: String,
    /// TensorZero client for calling inference and autopilot operations.
    pub t0_client: Arc<dyn TensorZeroClient>,
    /// Default max attempts for a task in the worker
    pub default_max_attempts: u32,
}

impl AutopilotWorkerConfig {
    /// Create config with required components.
    ///
    /// # Arguments
    ///
    /// * `pool` - Database pool for the durable task queue
    /// * `t0_client` - TensorZero client for inference and autopilot operations
    ///
    /// Environment variables:
    /// - `TENSORZERO_AUTOPILOT_QUEUE_NAME`: Queue name (default: "autopilot")
    pub fn new(
        pool: PgPool,
        t0_client: Arc<dyn TensorZeroClient>,
        default_max_attempts: u32,
    ) -> Self {
        let mut queue_name = autopilot_client::DEFAULT_SPAWN_QUEUE_NAME.to_string();
        if cfg!(feature = "e2e_tests")
            && let Some(name) = std::env::var("TENSORZERO_AUTOPILOT_QUEUE_NAME").ok()
        {
            queue_name = name;
        }

        Self {
            pool,
            queue_name,
            t0_client,
            default_max_attempts,
        }
    }
}

/// The autopilot worker that executes client tools.
pub struct AutopilotWorker {
    executor: Arc<ToolExecutor>,
}

impl AutopilotWorker {
    /// Create a new autopilot worker.
    ///
    /// # Errors
    ///
    /// Returns an error if the executor cannot be created.
    pub async fn new(config: AutopilotWorkerConfig) -> Result<Self> {
        let executor = ToolExecutor::builder()
            .pool(config.pool)
            .queue_name(&config.queue_name)
            .t0_client(config.t0_client)
            .default_max_attempts(config.default_max_attempts)
            .build()
            .await?;

        Ok(Self {
            executor: Arc::new(executor),
        })
    }

    /// Register all autopilot tools with the executor.
    #[allow(clippy::unused_async, clippy::allow_attributes)]
    pub async fn register_tools(&self) -> Result<()> {
        let visitor = LocalToolVisitor {
            executor: &self.executor,
        };
        autopilot_tools::for_each_tool(&visitor).await?;

        // Register internal tools directly without the ClientSimpleToolWrapper.
        // AutoRejectToolCallTool only writes a NotAvailable authorization -
        // it doesn't need the wrapper to publish a tool_result.
        self.executor
            .register_simple_tool::<autopilot_tools::tools::AutoRejectToolCallTool>()
            .await?;

        Ok(())
    }

    /// Get a clone of the Arc-wrapped executor for tool spawning.
    pub fn executor(&self) -> Arc<ToolExecutor> {
        self.executor.clone()
    }

    /// Start the durable worker.
    ///
    /// This is the fallible startup path that validates the durable configuration
    /// (queue exists, migrations applied, etc.). Call this before spawning the
    /// background task to ensure configuration errors are surfaced at startup.
    ///
    /// # Errors
    ///
    /// Returns an error if the worker cannot be started (e.g., queue missing,
    /// migrations not applied, database connection issues).
    pub async fn start(&self) -> Result<Worker> {
        self.executor
            .start_worker(WorkerOptions::default())
            .await
            .map_err(Into::into)
    }

    /// Run the worker until cancellation.
    ///
    /// This should be called in a spawned task after `start()` succeeds.
    pub async fn run_until_cancelled(worker: Worker, cancel_token: CancellationToken) {
        tokio::select! {
            () = cancel_token.cancelled() => {
                tracing::info!("Autopilot worker received shutdown signal");
                worker.shutdown().await;
            }
        }
    }
}

/// Visitor that registers tools for local execution on the autopilot worker.
///
/// All tools are wrapped to:
/// 1. Inject [`AutopilotSideInfo`] around the tool's native `SideInfo`
/// 2. Publish results to the autopilot API after execution
///
/// - TaskTools are wrapped in [`ClientToolWrapper`]
/// - SimpleTools are wrapped in [`ClientSimpleToolWrapper`] which promotes them to TaskTools
struct LocalToolVisitor<'a> {
    executor: &'a ToolExecutor,
}

#[async_trait]
impl ToolVisitor for LocalToolVisitor<'_> {
    type Error = ToolError;

    async fn visit_task_tool<T>(&self) -> Result<(), ToolError>
    where
        T: TaskTool + Default,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display,
    {
        self.executor
            .register_task_tool::<ClientTaskToolWrapper<T>>()
            .await?;
        Ok(())
    }

    async fn visit_simple_tool<T>(&self) -> Result<(), ToolError>
    where
        T: SimpleTool + Default,
        T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
        <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display,
    {
        // Register as a TaskTool (ClientSimpleToolWrapper promotes SimpleTool to TaskTool)
        self.executor
            .register_task_tool::<ClientSimpleToolWrapper<T>>()
            .await?;
        Ok(())
    }
}

/// Handle to the running autopilot worker.
///
/// This handle exposes the `ToolExecutor` for spawning tool executions
/// from the gateway.
#[derive(Clone)]
pub struct AutopilotWorkerHandle {
    executor: Arc<ToolExecutor>,
}

impl AutopilotWorkerHandle {
    /// Get a reference to the executor for spawning tools.
    pub fn executor(&self) -> &ToolExecutor {
        &self.executor
    }
}

/// Spawn the autopilot worker as a background task.
///
/// The worker will run alongside the gateway and shut down gracefully
/// when the gateway shuts down.
///
/// # Arguments
///
/// * `deferred_tasks` - Task tracker from the gateway for spawning background tasks
/// * `cancel_token` - Cancellation token for graceful shutdown
/// * `config` - Worker configuration
///
/// # Returns
///
/// Returns `Ok(AutopilotWorkerHandle)` if the worker was successfully spawned.
///
/// # Errors
///
/// Returns an error if the worker cannot be started. This catches configuration
/// errors (e.g., missing queue, migrations not applied) at startup rather than
/// failing silently in the background.
pub async fn spawn_autopilot_worker(
    deferred_tasks: &TaskTracker,
    cancel_token: CancellationToken,
    config: AutopilotWorkerConfig,
) -> Result<AutopilotWorkerHandle> {
    let worker = AutopilotWorker::new(config).await?;
    worker.register_tools().await?;

    // Start the durable worker before spawning to catch configuration errors early.
    // If durable is misconfigured (queue missing, migrations not applied), this will
    // return an error instead of silently failing in the background.
    let durable_worker = worker.start().await?;

    // Create the handle with a shared reference to the executor
    let handle = AutopilotWorkerHandle {
        executor: worker.executor(),
    };
    deferred_tasks.spawn(async move {
        AutopilotWorker::run_until_cancelled(durable_worker, cancel_token).await;
    });
    Ok(handle)
}
