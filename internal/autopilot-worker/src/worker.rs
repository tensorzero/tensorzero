//! Autopilot worker implementation.

use std::sync::Arc;

use anyhow::Result;
use durable_tools::{InferenceClient, ToolExecutor, WorkerOptions};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::tools::EchoTool;
use crate::wrapper::ClientToolWrapper;

/// Configuration for the autopilot worker.
pub struct AutopilotWorkerConfig {
    /// Database pool for the durable task queue (shared with gateway).
    pub pool: PgPool,
    /// Queue name for durable tasks (default: "autopilot").
    pub queue_name: String,
    /// Inference client for calling TensorZero endpoints.
    pub inference_client: Arc<dyn InferenceClient>,
}

impl AutopilotWorkerConfig {
    /// Create config with required components.
    ///
    /// # Arguments
    ///
    /// * `pool` - Database pool for the durable task queue
    /// * `inference_client` - Inference client for TensorZero operations
    ///
    /// Environment variables:
    /// - `TENSORZERO_AUTOPILOT_QUEUE_NAME`: Queue name (default: "autopilot")
    pub fn new(pool: PgPool, inference_client: Arc<dyn InferenceClient>) -> Self {
        let mut queue_name = autopilot_client::DEFAULT_SPAWN_QUEUE_NAME.to_string();
        if cfg!(feature = "e2e_tests")
            && let Some(name) = std::env::var("TENSORZERO_AUTOPILOT_QUEUE_NAME").ok()
        {
            queue_name = name;
        }

        Self {
            pool,
            queue_name,
            inference_client,
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
            .inference_client(config.inference_client)
            .build()
            .await?;

        Ok(Self {
            executor: Arc::new(executor),
        })
    }

    /// Register all autopilot tools with the executor.
    pub async fn register_tools(&self) -> Result<()> {
        // Register the echo tool for testing
        self.executor
            .register_task_tool::<ClientToolWrapper<EchoTool>>()
            .await?;

        // Additional tools will be registered here as they are implemented
        Ok(())
    }

    /// Get a clone of the Arc-wrapped executor for tool spawning.
    pub fn executor(&self) -> Arc<ToolExecutor> {
        self.executor.clone()
    }

    /// Start the worker and run until cancellation.
    pub async fn run(&self, cancel_token: CancellationToken) -> Result<()> {
        let worker = self.executor.start_worker(WorkerOptions::default()).await?;

        tokio::select! {
            () = cancel_token.cancelled() => {
                tracing::info!("Autopilot worker received shutdown signal");
                worker.shutdown().await;
            }
        }
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
/// Returns `Ok(AutopilotWorkerHandle)` if the worker was successfully spawned,
pub async fn spawn_autopilot_worker(
    deferred_tasks: &TaskTracker,
    cancel_token: CancellationToken,
    config: AutopilotWorkerConfig,
) -> Result<AutopilotWorkerHandle> {
    let worker = AutopilotWorker::new(config).await?;
    worker.register_tools().await?;

    // Create the handle with a shared reference to the executor
    let handle = AutopilotWorkerHandle {
        executor: worker.executor(),
    };
    deferred_tasks.spawn(async move { worker.run(cancel_token).await });
    Ok(handle)
}
