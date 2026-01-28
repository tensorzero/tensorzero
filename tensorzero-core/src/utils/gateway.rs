use std::collections::HashSet;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{DefaultBodyLimit, FromRequest, Json, Request, rejection::JsonRejection};
use moka::sync::Cache;
use serde::de::DeserializeOwned;
use sqlx::postgres::PgPoolOptions;
use tensorzero_auth::postgres::AuthResult;
use tokio::runtime::Handle;
use tokio::sync::oneshot::Sender;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::instrument;

use crate::config::{Config, ConfigFileGlob, snapshot::SnapshotHash, unwritten::UnwrittenConfig};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::migration_manager::{self, RunMigrationManagerArgs};
use crate::db::feedback::FeedbackQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::valkey::ValkeyConnectionInfo;
use crate::endpoints;
use crate::endpoints::openai_compatible::RouterExt;
use crate::error::{Error, ErrorDetails};
use crate::howdy::setup_howdy;
use crate::http::TensorzeroHttpClient;
use crate::rate_limiting::RateLimitingManager;
use autopilot_client::AutopilotClient;

#[cfg(test)]
use crate::db::clickhouse::ClickHouseClient;

/// A wrapper function called when we drop a `GatewayHandle`.
/// Note the double function type (an `fn` that takes a `Box<dyn FnOnce() + Send + '_>`).
/// The 'Box<dyn FnOnce() + Send + '_>` represents the drop logic for a `GatewayHandle`,
/// which needs to be inside a Tokio runtime.
/// The outer `fn` is responsible for performing any needed setup
/// (entering the Tokio runtime, releasing the Python GIL, etc)
/// that needs to wrap the actual drop logic.
pub type DropWrapper = fn(Box<dyn FnOnce() + Send + '_>);

/// Represents an active gateway (either standalone or embedded)
/// The contained `app_state` can be freely cloned and dropped.
/// However, dropping the `GatewayHandle` itself will wait for any
/// needed background tasks to exit (in the future, this will
/// include the ClickHouse batch insert task).
///
/// It's insufficient to put this kind of drop logic in `AppStateData` - since
/// it can be freely cloned, any contained `Arc`s might only be dropped when
/// the Tokio runtime is shutting down (e.g. if a background `tokio::spawn`
/// task looped forever without using a `CancellationToken` to exit).
/// During runtime shutdown, it's too late to call things like `tokio::spawn_blocking`,
/// so we may be unable to safely wait for our batch insert task to finish writing.
///
/// `GatewayHandle` should *not* be wrapped in an `Arc` (or given a `Clone` impl),
/// so that it's easy for us to tell where it gets dropped.
pub struct GatewayHandle {
    pub app_state: AppStateData,
    drop_wrapper: Option<DropWrapper>,
    _private: (),
}

impl Drop for GatewayHandle {
    fn drop(&mut self) {
        let drop_wrapper = self.drop_wrapper.take();
        let mut drop_self = || {
            self.app_state.shutdown_token.cancel();
            let handle = self
                .app_state
                .clickhouse_connection_info
                .batcher_join_handle();
            // Drop our `ClickHouseConnectionInfo`, so that we stop holding on to the `Arc<BatchSender>`
            // This allows the batch writer task to exit (once all of the remaining `ClickhouseConnectionInfo`s are dropped)
            self.app_state.clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
            if let Some(handle) = handle {
                tracing::info!("Waiting for ClickHouse batch writer to finish");
                // This could block forever if:
                // * We spawn a long-lived `tokio::task` that holds on to a `ClickhouseConnectionInfo`,
                //   and isn't using our `CancellationToken` to exit.
                // * The `GatewayHandle` is dropped from a task that's running other futures
                //   concurrently (e.g. a `try_join_all` where one of the futures somehow drops a `GatewayHandle`).
                //   In this case, the `block_in_place` call would prevent those futures from ever making progress,
                //   causing a `ClickhouseConnectionInfo` (and therefore the `Arc<BatchSender>`) to never be dropped.
                //   This is very unlikely, as we only create a `GatewayHandle` in a few places (the main gateway
                //   and embedded client), and drop it when we're exiting.
                //
                // We err on the side of hanging the server on shutdown, rather than potentially exiting while
                // we still have batched writes in-flight (or about to be written via an active `ClickhouseConnectionInfo`).
                tokio::task::block_in_place(|| {
                    if let Err(e) = Handle::current().block_on(handle) {
                        tracing::error!("Error in batch writer: {e}");
                    }
                });
                tracing::info!("ClickHouse batch writer finished");
            }
            // Return unused rate limit tokens to the database
            if !self.app_state.rate_limiting_manager.is_empty() {
                tracing::info!("Returning unused rate limit tokens to database");
                if let Err(e) = self.app_state.rate_limiting_manager.shutdown() {
                    tracing::warn!("Error returning rate limit tokens on shutdown: {e}");
                }
                tracing::info!("Rate limit token return complete");
            }

            self.app_state.deferred_tasks.close();
            // The 'wait' future will resolve immediately if the pool is empty.
            // Closing the pool doesn't block more futures from being added, so checking
            // if it's empty doesn't introduce any new race conditions (it's still possible
            // for some existing tokio task to spawn something in this pool after 'wait' resolves).
            // This check makes it easier to write tests (since we don't need to use a multi-threaded runtime),
            // as well as reducing the number of log messages in the common case.
            if !self.app_state.deferred_tasks.is_empty() {
                tokio::task::block_in_place(|| {
                    tracing::info!("Waiting for deferred tasks to finish");
                    Handle::current().block_on(self.app_state.deferred_tasks.wait());
                    tracing::info!("Deferred tasks finished");
                });
            }
        };
        // If we have a `DropWrapper` configured, call it with the `drop_self` function,
        // so that the `DropWrapper` can perform any needed setup (entering the Tokio runtime, releasing the Python GIL, etc)
        // that needs to wrap the actual drop logic.
        if let Some(drop_wrapper) = drop_wrapper {
            drop_wrapper(Box::new(drop_self));
        } else {
            drop_self();
        }
    }
}

/// State for the API
#[derive(Clone)]
// `#[non_exhaustive]` only affects downstream crates, so we can't use it here
#[expect(clippy::manual_non_exhaustive)]
pub struct AppStateData {
    pub config: Arc<Config>,
    pub http_client: TensorzeroHttpClient,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
    pub postgres_connection_info: PostgresConnectionInfo,
    pub valkey_connection_info: ValkeyConnectionInfo,
    /// Holds any background tasks that we want to wait on during shutdown
    /// We wait for these tasks to finish when `GatewayHandle` is dropped
    pub deferred_tasks: TaskTracker,
    /// Optional cache for TensorZero API key authentication
    pub auth_cache: Option<Cache<String, AuthResult>>,
    /// Optional cache for historical config snapshots loaded from ClickHouse
    pub config_snapshot_cache: Option<Cache<SnapshotHash, Arc<Config>>>,
    /// Optional Autopilot API client for proxying requests to the Autopilot API
    pub autopilot_client: Option<Arc<AutopilotClient>>,
    /// The deployment ID from ClickHouse (64-char hex string)
    pub deployment_id: Option<String>,
    /// Token pool manager for rate limiting pre-borrowing
    pub rate_limiting_manager: Arc<RateLimitingManager>,
    pub shutdown_token: CancellationToken,
    // Prevent `AppStateData` from being directly constructed outside of this module
    // This ensures that `AppStateData` is only ever constructed via explicit `new` methods,
    // which can ensure that we update global state.
    _private: (),
}
pub type AppState = axum::extract::State<AppStateData>;

/// Creates an auth cache based on the configuration.
/// Returns None if auth is disabled or cache is disabled.
fn create_auth_cache_from_config(config: &Config) -> Option<Cache<String, AuthResult>> {
    if !config.gateway.auth.enabled {
        return None;
    }

    let default_cache_config = Default::default();
    let cache_config = config
        .gateway
        .auth
        .cache
        .as_ref()
        .unwrap_or(&default_cache_config);

    if !cache_config.enabled {
        return None;
    }

    Some(
        Cache::builder()
            .time_to_live(Duration::from_millis(cache_config.ttl_ms))
            .build(),
    )
}

impl GatewayHandle {
    pub async fn new(
        config: UnwrittenConfig,
        available_tools: HashSet<String>,
    ) -> Result<Self, Error> {
        let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL").ok();
        let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").ok();
        let valkey_url = std::env::var("TENSORZERO_VALKEY_URL").ok();
        Box::pin(Self::new_with_databases(
            config,
            clickhouse_url,
            postgres_url,
            valkey_url,
            available_tools,
        ))
        .await
    }

    async fn new_with_databases(
        config: UnwrittenConfig,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
        valkey_url: Option<String>,
        available_tools: HashSet<String>,
    ) -> Result<Self, Error> {
        let clickhouse_connection_info = setup_clickhouse(&config, clickhouse_url, false).await?;
        let config = Arc::new(Box::pin(config.into_config(&clickhouse_connection_info)).await?);
        let postgres_connection_info = setup_postgres(&config, postgres_url).await?;
        let valkey_connection_info = setup_valkey(valkey_url.as_deref()).await?;
        let http_client = config.http_client.clone();
        Self::new_with_database_and_http_client(
            config,
            clickhouse_connection_info,
            postgres_connection_info,
            valkey_connection_info,
            http_client,
            None,
            available_tools,
        )
        .await
    }

    /// # Panics
    /// Panics if a `TensorzeroHttpClient` cannot be constructed
    #[cfg(test)]
    pub fn new_unit_test_data(config: Arc<Config>, test_options: GatewayHandleTestOptions) -> Self {
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info =
            ClickHouseConnectionInfo::new_mock(test_options.clickhouse_client);
        let postgres_connection_info =
            PostgresConnectionInfo::new_mock(test_options.postgres_healthy);
        let cancel_token = CancellationToken::new();
        let auth_cache = create_auth_cache_from_config(&config);
        // In unit tests, use Postgres for rate limiting (Valkey disabled)
        let rate_limiting_manager = Arc::new(
            RateLimitingManager::new_from_connections(
                Arc::new(config.rate_limiting.clone()),
                &ValkeyConnectionInfo::Disabled,
                &postgres_connection_info,
            )
            .unwrap(),
        );
        Self {
            app_state: AppStateData {
                config,
                http_client,
                clickhouse_connection_info,
                postgres_connection_info,
                valkey_connection_info: ValkeyConnectionInfo::Disabled,
                deferred_tasks: TaskTracker::new(),
                auth_cache,
                config_snapshot_cache: None,
                autopilot_client: None,
                deployment_id: None,
                rate_limiting_manager,
                shutdown_token: cancel_token,
                _private: (),
            },
            drop_wrapper: None,
            _private: (),
        }
    }

    pub async fn new_with_database_and_http_client(
        config: Arc<Config>,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        valkey_connection_info: ValkeyConnectionInfo,
        http_client: TensorzeroHttpClient,
        drop_wrapper: Option<DropWrapper>,
        available_tools: HashSet<String>,
    ) -> Result<Self, Error> {
        let rate_limiting_manager = Arc::new(RateLimitingManager::new_from_connections(
            Arc::new(config.rate_limiting.clone()),
            &valkey_connection_info,
            &postgres_connection_info,
        )?);

        let cancel_token = CancellationToken::new();
        setup_howdy(
            &config,
            clickhouse_connection_info.clone(),
            cancel_token.clone(),
        );

        // Fetch the deployment ID from ClickHouse (if available)
        let deployment_id = crate::howdy::get_deployment_id(&clickhouse_connection_info)
            .await
            .ok();

        for (function_name, function_config) in &config.functions {
            function_config
                .experimentation()
                .setup(
                    Arc::new(clickhouse_connection_info.clone())
                        as Arc<dyn FeedbackQueries + Send + Sync>,
                    function_name,
                    &postgres_connection_info,
                    cancel_token.clone(),
                )
                .await?;
        }
        let auth_cache = create_auth_cache_from_config(&config);

        // Create config snapshot cache with TTL of 5 minutes and max 100 entries
        let config_snapshot_cache = Some(
            Cache::builder()
                .time_to_live(Duration::from_secs(300))
                .max_capacity(10)
                .build(),
        );

        let autopilot_client = setup_autopilot_client(
            &postgres_connection_info,
            deployment_id.as_ref(),
            available_tools,
        )
        .await?;

        Ok(Self {
            app_state: AppStateData {
                config,
                http_client,
                clickhouse_connection_info,
                postgres_connection_info,
                valkey_connection_info,
                deferred_tasks: TaskTracker::new(),
                auth_cache,
                config_snapshot_cache,
                autopilot_client,
                deployment_id,
                rate_limiting_manager,
                shutdown_token: cancel_token,
                _private: (),
            },
            drop_wrapper,
            _private: (),
        })
    }
}

impl AppStateData {
    /// Create an AppStateData for use with a historical config snapshot.
    /// This version does not include auth_cache, config_snapshot_cache, autopilot_client,
    /// or deployment_id since those are specific to the live gateway.
    pub fn new_for_snapshot(
        config: Arc<Config>,
        http_client: TensorzeroHttpClient,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        valkey_connection_info: ValkeyConnectionInfo,
        deferred_tasks: TaskTracker,
        shutdown_token: CancellationToken,
    ) -> Result<Self, Error> {
        let rate_limiting_manager = Arc::new(RateLimitingManager::new_from_connections(
            Arc::new(config.rate_limiting.clone()),
            &valkey_connection_info,
            &postgres_connection_info,
        )?);
        Ok(Self {
            config,
            http_client,
            clickhouse_connection_info,
            postgres_connection_info,
            valkey_connection_info,
            deferred_tasks,
            auth_cache: None,
            config_snapshot_cache: None,
            autopilot_client: None,
            deployment_id: None,
            rate_limiting_manager,
            shutdown_token,
            _private: (),
        })
    }
}

pub async fn setup_clickhouse_without_config(
    clickhouse_url: String,
) -> Result<ClickHouseConnectionInfo, Error> {
    setup_clickhouse(
        &Box::pin(Config::new_empty()).await?,
        Some(clickhouse_url),
        true,
    )
    .await
}

pub async fn setup_clickhouse(
    config: &UnwrittenConfig,
    clickhouse_url: Option<String>,
    embedded_client: bool,
) -> Result<ClickHouseConnectionInfo, Error> {
    let clickhouse_connection_info = match (config.gateway.observability.enabled, clickhouse_url) {
        // Observability disabled by config
        (Some(false), _) => {
            tracing::info!(
                "Disabling observability: `gateway.observability.enabled` is set to false in config."
            );
            ClickHouseConnectionInfo::new_disabled()
        }
        // Observability enabled but no ClickHouse URL
        (Some(true), None) => {
            return Err(ErrorDetails::AppState {
                message: "Missing environment variable TENSORZERO_CLICKHOUSE_URL".to_string(),
            }
            .into());
        }
        // Observability enabled and ClickHouse URL provided
        (Some(true), Some(clickhouse_url)) => {
            ClickHouseConnectionInfo::new(
                &clickhouse_url,
                config.gateway.observability.batch_writes.clone(),
            )
            .await?
        }
        // Observability default and no ClickHouse URL
        (None, None) => {
            let msg_suffix = if embedded_client {
                "`clickhouse_url` was not provided."
            } else {
                "`TENSORZERO_CLICKHOUSE_URL` is not set."
            };
            tracing::warn!(
                "Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and {msg_suffix}"
            );
            ClickHouseConnectionInfo::new_disabled()
        }
        // Observability default and ClickHouse URL provided
        (None, Some(clickhouse_url)) => {
            ClickHouseConnectionInfo::new(
                &clickhouse_url,
                config.gateway.observability.batch_writes.clone(),
            )
            .await?
        }
    };

    // Run ClickHouse migrations (if any) if we have a production ClickHouse connection
    if clickhouse_connection_info.client_type() == ClickHouseClientType::Production {
        migration_manager::run(RunMigrationManagerArgs {
            clickhouse: &clickhouse_connection_info,
            is_manual_run: false,
            disable_automatic_migrations: config.gateway.observability.disable_automatic_migrations,
        })
        .await?;
    }
    Ok(clickhouse_connection_info)
}

async fn create_postgres_connection(
    postgres_url: &str,
    connection_pool_size: u32,
) -> Result<PostgresConnectionInfo, Error> {
    let pool = PgPoolOptions::new()
        .max_connections(connection_pool_size)
        .connect(postgres_url)
        .await
        .map_err(|err| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: err.to_string(),
            })
        })?;

    let connection_info = PostgresConnectionInfo::new_with_pool(pool);
    connection_info.check_migrations().await?;
    Ok(connection_info)
}

// TODO(#5764): We should test that on startup we issue the correct SQL for write_retention_config,
// but this is currently structured that's difficult to swap in a Mock.
pub async fn setup_postgres(
    config: &Config,
    postgres_url: Option<String>,
) -> Result<PostgresConnectionInfo, Error> {
    let postgres_connection_info = match (config.postgres.enabled, postgres_url.as_deref()) {
        // Postgres disabled by config
        (Some(false), _) => {
            tracing::info!("Disabling Postgres: `postgres.enabled` is set to false in config.");
            PostgresConnectionInfo::Disabled
        }
        // Postgres enabled but no URL
        (Some(true), None) => {
            return Err(ErrorDetails::AppState {
                message: "Missing environment variable `TENSORZERO_POSTGRES_URL`.".to_string(),
            }
            .into());
        }
        // Postgres enabled and URL provided
        (Some(true), Some(postgres_url)) => {
            create_postgres_connection(postgres_url, config.postgres.connection_pool_size).await?
        }
        // Postgres default and no URL
        (None, None) => {
            tracing::debug!(
                "Disabling Postgres: `postgres.enabled` is not explicitly specified in config and `TENSORZERO_POSTGRES_URL` is not set."
            );
            PostgresConnectionInfo::Disabled
        }
        // Postgres default and URL provided
        (None, Some(postgres_url)) => {
            create_postgres_connection(postgres_url, config.postgres.connection_pool_size).await?
        }
    };

    // Write retention config to Postgres (syncs tensorzero.toml -> database)
    postgres_connection_info
        .write_retention_config(config.postgres.inference_retention_days)
        .await?;

    Ok(postgres_connection_info)
}

/// Sets up the Valkey connection from the provided URL.
///
/// Valkey is optional; if no URL is provided, rate limiting will fall back to PostgreSQL.
///
/// # Arguments
/// * `valkey_url` - Optional Valkey URL (from `TENSORZERO_VALKEY_URL` env var)
pub async fn setup_valkey(valkey_url: Option<&str>) -> Result<ValkeyConnectionInfo, Error> {
    match valkey_url {
        Some(url) => ValkeyConnectionInfo::new(url).await,
        None => {
            tracing::debug!("Disabling Valkey: `TENSORZERO_VALKEY_URL` is not set.");
            Ok(ValkeyConnectionInfo::Disabled)
        }
    }
}

/// Sets up the Autopilot API client from the environment.
/// Returns `Ok(Some(client))` if TENSORZERO_AUTOPILOT_API_KEY is set,
/// `Ok(None)` if not set, or an error if client construction fails.
/// Requires Postgres and ClickHouse (for deployment_id) to be enabled.
///
/// Environment variables:
/// - `TENSORZERO_AUTOPILOT_API_KEY`: Required to enable the client
/// - `TENSORZERO_AUTOPILOT_BASE_URL`: Optional custom base URL (for testing)
/// - `TENSORZERO_AUTOPILOT_QUEUE_NAME`: Optional queue name for tool dispatching
async fn setup_autopilot_client(
    postgres_connection_info: &PostgresConnectionInfo,
    deployment_id: Option<&String>,
    available_tools: HashSet<String>,
) -> Result<Option<Arc<AutopilotClient>>, Error> {
    match std::env::var("TENSORZERO_AUTOPILOT_API_KEY") {
        Ok(api_key) => {
            let pool = postgres_connection_info.get_pool().ok_or_else(|| {
                Error::new(ErrorDetails::AppState {
                    message: "Autopilot client requires Postgres; set `TENSORZERO_POSTGRES_URL`."
                        .to_string(),
                })
            })?;

            // Require `deployment_id` (from ClickHouse) for autopilot
            if deployment_id.is_none() {
                return Err(Error::new(ErrorDetails::AppState {
                    message:
                        "Failed to fetch the deployment ID from ClickHouse. Please make sure that ClickHouse is running and accessible."
                            .to_string(),
                }));
            }
            let queue_name = std::env::var("TENSORZERO_AUTOPILOT_QUEUE_NAME")
                .unwrap_or_else(|_| "autopilot".to_string());

            let mut builder = AutopilotClient::builder()
                .api_key(api_key)
                .spawn_pool(pool.clone())
                .spawn_queue_name(queue_name)
                .available_tools(available_tools);

            // Allow custom base URL for testing
            if let Ok(base_url) = std::env::var("TENSORZERO_AUTOPILOT_BASE_URL") {
                let url = base_url.parse().map_err(|e| {
                    Error::new(ErrorDetails::AppState {
                        message: format!("Invalid TENSORZERO_AUTOPILOT_BASE_URL: {e}"),
                    })
                })?;
                builder = builder.base_url(url);
                tracing::info!("Autopilot client using custom base URL: {}", base_url);
            }

            let client = builder.build().await.map_err(Error::from)?;
            // TODO: Handshake with API to validate credentials
            tracing::info!("Autopilot client initialized");
            Ok(Some(Arc::new(client)))
        }
        Err(std::env::VarError::NotPresent) => {
            tracing::debug!(
                "Autopilot client not configured: TENSORZERO_AUTOPILOT_API_KEY not set"
            );
            Ok(None)
        }
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::new(ErrorDetails::AppState {
            message: "TENSORZERO_AUTOPILOT_API_KEY contains invalid UTF-8".to_string(),
        })),
    }
}

/// Custom Axum extractor that validates the JSON body and deserializes it into a custom type
///
/// When this extractor is present, we don't check if the `Content-Type` header is `application/json`,
/// and instead simply assume that the request body is a JSON object.
pub struct StructuredJson<T>(pub T);

/// Shared JSON deserialization logic used by both `StructuredJson` and `OpenAIStructuredJson`.
///
/// Parses the request body as JSON and deserializes it into the target type `T`,
/// using `serde_path_to_error` for detailed error messages.
pub(crate) async fn deserialize_json_request<S, T>(req: Request, state: &S) -> Result<T, Error>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    // Retrieve the request body as Bytes before deserializing it
    let bytes = bytes::Bytes::from_request(req, state).await.map_err(|e| {
        Error::new(ErrorDetails::JsonRequest {
            message: format!("{} ({})", e, e.status()),
        })
    })?;

    // Convert the entire body into `serde_json::Value`
    let value = Json::<serde_json::Value>::from_bytes(&bytes)
        .map_err(|e| {
            Error::new(ErrorDetails::JsonRequest {
                message: format!("{} ({})", e, e.status()),
            })
        })?
        .0;

    // Now use `serde_path_to_error::deserialize` to attempt deserialization into `T`
    let deserialized: T = serde_path_to_error::deserialize(&value).map_err(|e| {
        Error::new(ErrorDetails::JsonRequest {
            message: e.to_string(),
        })
    })?;

    Ok(deserialized)
}

impl<S, T> FromRequest<S> for StructuredJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
    T: Send + Sync + DeserializeOwned,
{
    type Rejection = Error;

    #[instrument(skip_all, level = "trace", name = "StructuredJson::from_request")]
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        deserialize_json_request(req, state)
            .await
            .map(StructuredJson)
    }
}

// We hold on to these fields so that their Drop impls run when `ShutdownHandle` is dropped
// IMPORTANT: The 'sender' field must come first in the struct definition, so that Rust will
// drop it first: https://doc.rust-lang.org/reference/destructors.html#r-destructors.operation
// This triggers graceful shutdown of the Axum server first, and then waits for the server to
// exit in the `Drop` impl for `GatewayHandle`.
// Declaring these fields in the opposite order will cause a deadlock, since we'll wait on
// an Axum server to shutdown without triggering graceful shutdown first.
pub struct ShutdownHandle {
    #[expect(dead_code)]
    sender: Sender<()>,
    #[expect(dead_code)]
    gateway_handle: GatewayHandle,
}

/// Starts a new HTTP TensorZero gateway on an unused port, with only the openai-compatible endpoint enabled.
/// This is used in by `patch_openai_client` in the Python client to allow pointing the OpenAI client
/// at a local gateway (via `base_url`).
///
/// Returns the address the gateway is listening on, and a future resolves (after the gateway starts up)
/// to a `ShutdownHandle` which shuts down the gateway when dropped.
pub async fn start_openai_compatible_gateway(
    config_file: Option<String>,
    clickhouse_url: Option<String>,
    postgres_url: Option<String>,
    valkey_url: Option<String>,
) -> Result<(SocketAddr, ShutdownHandle), Error> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to bind to a port: {e}"),
            })
        })?;
    let bind_addr = listener.local_addr().map_err(|e| {
        Error::new(ErrorDetails::InternalError {
            message: format!("Failed to get local address: {e}"),
        })
    })?;
    let config_load_info = if let Some(config_file) = config_file {
        Box::pin(Config::load_and_verify_from_path(&ConfigFileGlob::new(
            config_file,
        )?))
        .await?
    } else {
        Box::pin(Config::new_empty()).await?
    };
    let gateway_handle = Box::pin(GatewayHandle::new_with_databases(
        config_load_info,
        clickhouse_url,
        postgres_url,
        valkey_url,
        HashSet::new(), // available_tools
    ))
    .await?;

    let router = Router::new()
        .register_openai_compatible_routes()
        .fallback(endpoints::fallback::handle_404)
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // increase the default body limit from 2MB to 100MB
        .layer(axum::middleware::from_fn_with_state(
            crate::observability::request_logging::InFlightRequestsData::new(),
            crate::observability::request_logging::request_logging_middleware,
        ))
        .with_state(gateway_handle.app_state.clone());

    let (sender, recv) = tokio::sync::oneshot::channel::<()>();
    let shutdown_fut = async move {
        let _ = recv.await;
    };

    // Note - this will cause `gateway_handle` to block on the Axum server shutting down
    // when `gateway_handle` is dropped.
    // See the comment on `ShutdownHandle` for more details.
    gateway_handle.app_state.deferred_tasks.spawn(
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_fut)
            .into_future(),
    );
    Ok((
        bind_addr,
        ShutdownHandle {
            sender,
            gateway_handle,
        },
    ))
}

#[cfg(test)]
pub struct GatewayHandleTestOptions {
    pub clickhouse_client: Arc<dyn ClickHouseClient>,
    pub postgres_healthy: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ObservabilityConfig, PostgresConfig, gateway::GatewayConfig, snapshot::ConfigSnapshot,
        unwritten::UnwrittenConfig,
    };
    #[tokio::test]
    async fn test_setup_clickhouse() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Disabled observability
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(false),
                async_writes: false,
                batch_writes: Default::default(),
                disable_automatic_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
            disable_pseudonymous_usage_analytics: false,
            fetch_and_encode_input_files_before_inference: false,
            auth: Default::default(),
            global_outbound_http_timeout: Default::default(),
            relay: None,
            metrics: Default::default(),
        };

        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let config = UnwrittenConfig::new(config, ConfigSnapshot::new_empty_for_test());

        let clickhouse_connection_info = setup_clickhouse(&config, None, false).await.unwrap();
        assert_eq!(
            clickhouse_connection_info.client_type(),
            ClickHouseClientType::Disabled
        );
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));

        // Default observability and no ClickHouse URL
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: None,
                async_writes: false,
                batch_writes: Default::default(),
                disable_automatic_migrations: false,
            },
            fetch_and_encode_input_files_before_inference: false,
            unstable_error_json: false,
            ..Default::default()
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(config, ConfigSnapshot::new_empty_for_test());
        let clickhouse_connection_info = setup_clickhouse(&unwritten_config, None, false)
            .await
            .unwrap();
        assert_eq!(
            clickhouse_connection_info.client_type(),
            ClickHouseClientType::Disabled
        );
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));
        assert!(logs_contain(
            "Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and `TENSORZERO_CLICKHOUSE_URL` is not set."
        ));

        // We do not test the case where a ClickHouse URL is provided but observability is default,
        // as this would require a working ClickHouse and we don't have one in unit tests.

        // Observability enabled but ClickHouse URL is missing
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
                async_writes: false,
                batch_writes: Default::default(),
                disable_automatic_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
            disable_pseudonymous_usage_analytics: false,
            fetch_and_encode_input_files_before_inference: false,
            auth: Default::default(),
            global_outbound_http_timeout: Default::default(),
            relay: None,
            metrics: Default::default(),
        };

        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(config, ConfigSnapshot::new_empty_for_test());

        let err = setup_clickhouse(&unwritten_config, None, false)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Missing environment variable TENSORZERO_CLICKHOUSE_URL")
        );

        // Bad URL
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
                async_writes: false,
                batch_writes: Default::default(),
                disable_automatic_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
            disable_pseudonymous_usage_analytics: false,
            fetch_and_encode_input_files_before_inference: false,
            auth: Default::default(),
            global_outbound_http_timeout: Default::default(),
            relay: None,
            metrics: Default::default(),
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(config, ConfigSnapshot::new_empty_for_test());
        setup_clickhouse(&unwritten_config, Some("bad_url".to_string()), false)
            .await
            .expect_err("ClickHouse setup should fail given a bad URL");
        assert!(logs_contain("Invalid ClickHouse database URL"));
    }

    #[tokio::test]
    async fn test_unhealthy_clickhouse() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Sensible URL that doesn't point to ClickHouse
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
                async_writes: false,
                batch_writes: Default::default(),
                disable_automatic_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
            disable_pseudonymous_usage_analytics: false,
            fetch_and_encode_input_files_before_inference: false,
            auth: Default::default(),
            global_outbound_http_timeout: Default::default(),
            relay: None,
            metrics: Default::default(),
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(config, ConfigSnapshot::new_empty_for_test());
        setup_clickhouse(
            &unwritten_config,
            Some("https://tensorzero.invalid:8123".to_string()),
            false,
        )
        .await
        .expect_err("ClickHouse setup should fail given a URL that doesn't point to ClickHouse");
        assert!(logs_contain(
            "Error connecting to ClickHouse: ClickHouse is not healthy"
        ));
        // We do not test the case where a ClickHouse URL is provided and observability is on,
        // as this would require a working ClickHouse and we don't have one in unit tests.
    }

    #[tokio::test]
    async fn test_setup_postgres_disabled() {
        let logs_contain = crate::utils::testing::capture_logs();

        // Postgres disabled by config
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(false),
                connection_pool_size: 20,
                inference_retention_days: None,
            },
            ..Default::default()
        }));

        let postgres_connection_info = setup_postgres(config, None).await.unwrap();
        assert!(matches!(
            postgres_connection_info,
            PostgresConnectionInfo::Disabled
        ));
        assert!(logs_contain(
            "Disabling Postgres: `postgres.enabled` is set to false in config."
        ));

        // Postgres disabled even with URL provided
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(false),
                connection_pool_size: 20,
                inference_retention_days: None,
            },
            ..Default::default()
        }));

        let postgres_connection_info = setup_postgres(
            config,
            Some("postgresql://user:pass@localhost:5432/db".to_string()),
        )
        .await
        .unwrap();
        assert!(matches!(
            postgres_connection_info,
            PostgresConnectionInfo::Disabled
        ));
    }

    #[tokio::test]
    async fn test_setup_postgres_default_no_url() {
        // Default postgres config (enabled: None) and no URL
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: None,
                connection_pool_size: 20,
                inference_retention_days: None,
            },
            ..Default::default()
        }));

        let postgres_connection_info = setup_postgres(config, None).await.unwrap();
        assert!(matches!(
            postgres_connection_info,
            PostgresConnectionInfo::Disabled
        ));
    }

    #[tokio::test]
    async fn test_setup_postgres_enabled_no_url() {
        // Postgres enabled but URL is missing
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(true),
                connection_pool_size: 20,
                inference_retention_days: None,
            },
            ..Default::default()
        }));

        let err = setup_postgres(config, None).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Missing environment variable `TENSORZERO_POSTGRES_URL`.")
        );
    }

    #[tokio::test]
    async fn test_setup_postgres_bad_url() {
        // Postgres enabled with bad URL
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(true),
                connection_pool_size: 20,
                inference_retention_days: None,
            },
            ..Default::default()
        }));

        setup_postgres(config, Some("bad_url".to_string()))
            .await
            .expect_err("Postgres setup should fail given a bad URL");
    }

    #[tokio::test]
    async fn test_no_rate_limiting_does_not_require_postgres_or_valkey() {
        // Rate limiting enabled=false should not fail validation (no rules configured)
        let config_no_rules = Arc::new(Config {
            postgres: PostgresConfig {
                enabled: Some(false),
                connection_pool_size: 20,
                inference_retention_days: None,
            },
            rate_limiting: Default::default(),
            ..Default::default()
        });

        let http_client = TensorzeroHttpClient::new_testing().unwrap();

        // This should succeed because rate limiting has no rules
        let _gateway = GatewayHandle::new_with_database_and_http_client(
            config_no_rules,
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(), // available_tools
        )
        .await
        .expect("Gateway setup should succeed when rate limiting has no rules");
    }
}
