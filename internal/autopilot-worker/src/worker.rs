//! Autopilot worker implementation.

use std::sync::Arc;

use durable_tools::{InferenceClient, ToolExecutor, WorkerOptions, http_gateway_client};
use secrecy::SecretString;
use tensorzero_core::utils::gateway::{AppStateData, GatewayHandle};
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::registry::build_default_registry;
use crate::state::AutopilotExtension;

/// Configuration for the autopilot worker.
pub struct AutopilotWorkerConfig {
    /// Database URL for the durable task queue.
    pub durable_database_url: SecretString,
    /// Queue name for durable tasks.
    pub queue_name: String,
    /// URL of the TensorZero gateway for inference.
    pub gateway_url: Url,
    /// The autopilot client for sending results.
    pub autopilot_client: Arc<autopilot_client::AutopilotClient>,
}

/// The autopilot worker that executes client tools.
pub struct AutopilotWorker {
    executor: ToolExecutor,
}

impl AutopilotWorker {
    /// Create a new autopilot worker.
    ///
    /// # Arguments
    ///
    /// * `config` - Worker configuration
    /// * `gateway_state` - Shared gateway state for accessing TensorZero functionality
    ///
    /// # Errors
    ///
    /// Returns an error if the worker fails to initialize.
    pub async fn new(
        config: AutopilotWorkerConfig,
        gateway_state: AppStateData,
    ) -> anyhow::Result<Self> {
        // Create inference client pointing to the gateway
        let inference_client: Arc<dyn InferenceClient> =
            http_gateway_client(config.gateway_url)?;

        // Create the autopilot extension with the client and gateway state
        let extension = AutopilotExtension::new(config.autopilot_client, gateway_state);

        // Create the tool executor with the extension
        let executor = ToolExecutor::builder()
            .database_url(config.durable_database_url)
            .queue_name(&config.queue_name)
            .inference_client(inference_client)
            .extension(extension)
            .build()
            .await?;

        Ok(Self { executor })
    }

    /// Register all autopilot client tools with the executor.
    pub async fn register_tools(&self) {
        build_default_registry(&self.executor).await;
    }

    /// Start the worker and run until cancellation.
    ///
    /// # Arguments
    ///
    /// * `cancel_token` - Token to signal graceful shutdown
    ///
    /// # Errors
    ///
    /// Returns an error if the worker fails.
    pub async fn run(self, cancel_token: CancellationToken) -> anyhow::Result<()> {
        let worker = self.executor.start_worker(WorkerOptions::default()).await;

        tokio::select! {
            () = cancel_token.cancelled() => {
                tracing::info!("Autopilot worker received shutdown signal");
                worker.shutdown().await;
                Ok(())
            }
        }
    }

    /// Get a reference to the tool executor.
    pub fn executor(&self) -> &ToolExecutor {
        &self.executor
    }
}

/// Spawn the autopilot worker as a tracked background task.
///
/// The worker will run alongside the gateway and shut down gracefully
/// when the gateway shuts down.
///
/// # Arguments
///
/// * `gateway_handle` - Handle to the running gateway
/// * `config` - Worker configuration
pub fn spawn_autopilot_worker(gateway_handle: &GatewayHandle, config: AutopilotWorkerConfig) {
    let gateway_state = gateway_handle.app_state.clone();
    let cancel_token = gateway_handle.cancel_token.clone();

    gateway_handle.app_state.deferred_tasks.spawn(async move {
        match AutopilotWorker::new(config, gateway_state).await {
            Ok(worker) => {
                tracing::info!("Autopilot worker started");
                worker.register_tools().await;

                if let Err(e) = worker.run(cancel_token).await {
                    tracing::error!("Autopilot worker error: {e}");
                }

                tracing::info!("Autopilot worker stopped");
            }
            Err(e) => {
                tracing::error!("Failed to create autopilot worker: {e}");
            }
        }
    });
}
