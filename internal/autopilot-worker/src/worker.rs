//! Autopilot worker implementation.

use std::sync::Arc;

use anyhow::Result;
use durable_tools::{ToolExecutor, WorkerOptions, http_gateway_client};
use sqlx::PgPool;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use url::Url;

use crate::tools::EchoTool;
use crate::wrapper::ClientToolWrapper;

/// Configuration for the autopilot worker.
pub struct AutopilotWorkerConfig {
    /// Database pool for the durable task queue (shared with gateway).
    pub pool: PgPool,
    /// Queue name for durable tasks (default: "autopilot").
    pub queue_name: String,
    /// URL of the TensorZero gateway for inference (default: http://127.0.0.1:3000).
    pub gateway_url: Url,
}

/// Default gateway URL for the autopilot worker.
const DEFAULT_GATEWAY_URL: &str = "http://127.0.0.1:3000";

impl AutopilotWorkerConfig {
    /// Create config with an existing pool and optional overrides from environment.
    ///
    /// Environment variables:
    /// - `TENSORZERO_AUTOPILOT_QUEUE_NAME`: Queue name (default: "autopilot")
    /// - `TENSORZERO_AUTOPILOT_GATEWAY_URL`: Gateway URL (default: http://127.0.0.1:3000)
    ///
    /// # Errors
    ///
    /// Returns an error if `TENSORZERO_AUTOPILOT_GATEWAY_URL` is set but invalid.
    pub fn new(pool: PgPool) -> Result<Self> {
        let queue_name = std::env::var("TENSORZERO_AUTOPILOT_QUEUE_NAME")
            .unwrap_or_else(|_| "autopilot".to_string());

        let gateway_url = match std::env::var("TENSORZERO_AUTOPILOT_GATEWAY_URL") {
            Ok(s) => s
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid TENSORZERO_AUTOPILOT_GATEWAY_URL: {e}"))?,
            Err(_) => DEFAULT_GATEWAY_URL
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid default gateway URL: {e}"))?,
        };

        Ok(Self {
            pool,
            queue_name,
            gateway_url,
        })
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
    /// Returns an error if the inference client cannot be created.
    pub async fn new(config: AutopilotWorkerConfig) -> Result<Self> {
        let inference_client = http_gateway_client(config.gateway_url)?;

        let executor = ToolExecutor::builder()
            .pool(config.pool)
            .queue_name(&config.queue_name)
            .inference_client(inference_client)
            .build()
            .await?;

        Ok(Self {
            executor: Arc::new(executor),
        })
    }

    /// Register all autopilot tools with the executor.
    pub async fn register_tools(&self) {
        // Register the echo tool for testing
        self.executor
            .register_task_tool::<ClientToolWrapper<EchoTool>>()
            .await;

        // Additional tools will be registered here as they are implemented
    }

    /// Get a clone of the Arc-wrapped executor for tool spawning.
    pub fn executor(&self) -> Arc<ToolExecutor> {
        self.executor.clone()
    }

    /// Start the worker and run until cancellation.
    ///
    /// # Errors
    ///
    /// Returns an error if the worker fails to start.
    pub async fn run(&self, cancel_token: CancellationToken) -> Result<()> {
        let worker = self.executor.start_worker(WorkerOptions::default()).await;

        tokio::select! {
            () = cancel_token.cancelled() => {
                tracing::info!("Autopilot worker received shutdown signal");
                worker.shutdown().await;
                Ok(())
            }
        }
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
/// * `config` - Worker configuration (as a Result from `AutopilotWorkerConfig::new()`)
///
/// # Returns
///
/// Returns `Some(AutopilotWorkerHandle)` if the worker was successfully spawned,
/// `None` if spawning failed (error is logged).
pub fn spawn_autopilot_worker(
    deferred_tasks: &TaskTracker,
    cancel_token: CancellationToken,
    config: Result<AutopilotWorkerConfig>,
) -> Option<AutopilotWorkerHandle> {
    // We use a oneshot channel to get the handle back from the spawned task
    let (tx, rx) = tokio::sync::oneshot::channel();

    deferred_tasks.spawn(async move {
        let config = match config {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Invalid autopilot worker config: {e}");
                let _ = tx.send(None);
                return;
            }
        };
        match AutopilotWorker::new(config).await {
            Ok(worker) => {
                tracing::info!("Autopilot worker started");
                worker.register_tools().await;

                // Create the handle with a shared reference to the executor
                let handle = AutopilotWorkerHandle {
                    executor: worker.executor(),
                };

                // Send the handle back before starting the run loop
                let _ = tx.send(Some(handle));

                if let Err(e) = worker.run(cancel_token).await {
                    tracing::error!("Autopilot worker error: {e}");
                }

                tracing::info!("Autopilot worker stopped");
            }
            Err(e) => {
                tracing::error!("Failed to create autopilot worker: {e}");
                let _ = tx.send(None);
            }
        }
    });

    // Block briefly to get the handle (this happens during startup)
    // Using a small timeout to avoid blocking forever if something goes wrong
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(30), rx)
                .await
                .ok()
                .and_then(|r| r.ok())
                .flatten()
        })
    })
}
