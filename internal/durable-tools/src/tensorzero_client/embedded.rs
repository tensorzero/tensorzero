//! Embedded TensorZero client that uses gateway state directly.
//!
//! This implementation is used when the worker runs inside the gateway process
//! and wants to call inference and autopilot endpoints without HTTP overhead.

use async_trait::async_trait;
use moka::sync::Cache;
use std::sync::Arc;
use tensorzero::{ClientInferenceParams, InferenceOutput, InferenceResponse, TensorZeroError};
use tensorzero_core::config::Config;
use tensorzero_core::config::snapshot::SnapshotHash;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::http::TensorzeroHttpClient;
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use super::{
    CreateEventRequest, CreateEventResponse, ListEventsParams, ListEventsResponse,
    ListSessionsParams, ListSessionsResponse, TensorZeroClient, TensorZeroClientError,
};

/// TensorZero client that uses an existing gateway's state directly.
///
/// This is used when the worker runs inside the gateway process and wants to
/// call inference and autopilot endpoints without HTTP overhead.
pub struct EmbeddedClient {
    config: Arc<Config>,
    http_client: TensorzeroHttpClient,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    postgres_connection_info: PostgresConnectionInfo,
    deferred_tasks: TaskTracker,
    autopilot_client: Option<Arc<autopilot_client::AutopilotClient>>,
    /// Cache for historical config snapshots, used by the action endpoint.
    config_snapshot_cache: Option<Cache<SnapshotHash, Arc<Config>>>,
}

impl EmbeddedClient {
    /// Create a new embedded client from gateway state components.
    pub fn new(
        config: Arc<Config>,
        http_client: TensorzeroHttpClient,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        deferred_tasks: TaskTracker,
        autopilot_client: Option<Arc<autopilot_client::AutopilotClient>>,
        config_snapshot_cache: Option<Cache<SnapshotHash, Arc<Config>>>,
    ) -> Self {
        Self {
            config,
            http_client,
            clickhouse_connection_info,
            postgres_connection_info,
            deferred_tasks,
            autopilot_client,
            config_snapshot_cache,
        }
    }
}

#[async_trait]
impl TensorZeroClient for EmbeddedClient {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        let internal_params = params
            .try_into()
            .map_err(|e: tensorzero_core::error::Error| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;

        let result = Box::pin(tensorzero_core::endpoints::inference::inference(
            self.config.clone(),
            &self.http_client,
            self.clickhouse_connection_info.clone(),
            self.postgres_connection_info.clone(),
            self.deferred_tasks.clone(),
            internal_params,
            None, // No API key in embedded mode
        ))
        .await
        .map_err(|e| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
        })?;

        match result.output {
            InferenceOutput::NonStreaming(response) => Ok(response),
            InferenceOutput::Streaming(_) => Err(TensorZeroClientError::StreamingNotSupported),
        }
    }

    async fn create_autopilot_event(
        &self,
        session_id: Uuid,
        request: CreateEventRequest,
    ) -> Result<CreateEventResponse, TensorZeroClientError> {
        let autopilot_client = self
            .autopilot_client
            .as_ref()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        tensorzero_core::endpoints::internal::autopilot::create_event(
            autopilot_client,
            session_id,
            request,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn list_autopilot_events(
        &self,
        session_id: Uuid,
        params: ListEventsParams,
    ) -> Result<ListEventsResponse, TensorZeroClientError> {
        let autopilot_client = self
            .autopilot_client
            .as_ref()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        tensorzero_core::endpoints::internal::autopilot::list_events(
            autopilot_client,
            session_id,
            params,
        )
        .await
        .map_err(|e| TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() }))
    }

    async fn list_autopilot_sessions(
        &self,
        params: ListSessionsParams,
    ) -> Result<ListSessionsResponse, TensorZeroClientError> {
        let autopilot_client = self
            .autopilot_client
            .as_ref()
            .ok_or(TensorZeroClientError::AutopilotUnavailable)?;

        tensorzero_core::endpoints::internal::autopilot::list_sessions(autopilot_client, params)
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })
    }

    async fn action(
        &self,
        snapshot_hash: SnapshotHash,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, TensorZeroClientError> {
        use tensorzero_core::config::RuntimeOverlay;
        use tensorzero_core::db::ConfigQueries;

        // Get the config snapshot cache, or return an error if not available
        let cache = self.config_snapshot_cache.as_ref().ok_or_else(|| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other {
                source: tensorzero_core::error::Error::new(
                    tensorzero_core::error::ErrorDetails::InternalError {
                        message: "Config snapshot cache is not enabled".to_string(),
                    },
                )
                .into(),
            })
        })?;

        // Check cache first
        let config = if let Some(config) = cache.get(&snapshot_hash) {
            config
        } else {
            // Cache miss: load from ClickHouse
            let snapshot = self
                .clickhouse_connection_info
                .get_config_snapshot(snapshot_hash.clone())
                .await
                .map_err(|e| {
                    TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
                })?;

            let runtime_overlay = RuntimeOverlay::from_config(&self.config);

            let unwritten_config = Config::load_from_snapshot(
                snapshot,
                runtime_overlay,
                false, // Don't validate credentials for historical configs
            )
            .await
            .map_err(|e| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;

            let config = Arc::new(unwritten_config.dangerous_into_config_without_writing());

            cache.insert(snapshot_hash, config.clone());

            config
        };

        // Convert params and call inference with the snapshot config
        let internal_params = params
            .try_into()
            .map_err(|e: tensorzero_core::error::Error| {
                TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
            })?;

        let result = Box::pin(tensorzero_core::endpoints::inference::inference(
            config,
            &self.http_client,
            self.clickhouse_connection_info.clone(),
            self.postgres_connection_info.clone(),
            self.deferred_tasks.clone(),
            internal_params,
            None, // No API key in embedded mode
        ))
        .await
        .map_err(|e| {
            TensorZeroClientError::TensorZero(TensorZeroError::Other { source: e.into() })
        })?;

        match result.output {
            InferenceOutput::NonStreaming(response) => Ok(response),
            InferenceOutput::Streaming(_) => Err(TensorZeroClientError::StreamingNotSupported),
        }
    }
}
