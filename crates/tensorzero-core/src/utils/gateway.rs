// This module defines and implements SwappableAppStateData and GatewayHandle.
#![expect(
    clippy::disallowed_types,
    reason = "definition and implementation module for SwappableAppStateData"
)]

use std::collections::HashSet;
use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::{
    DefaultBodyLimit, FromRef, FromRequest, Json, Request, rejection::JsonRejection,
};
use moka::sync::Cache;
use serde::de::DeserializeOwned;
use sqlx::ConnectOptions;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use tensorzero_auth::postgres::AuthResult;
use tokio::runtime::Handle;
use tokio::sync::oneshot::Sender;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::instrument;

use crate::cache::CacheManager;
use crate::config::gateway::{
    default_gateway_auth_cache_enabled, default_gateway_auth_cache_ttl_ms,
};
use crate::config::{
    BatchWritesConfig, Config, ConfigFileGlob, DEFAULT_POSTGRES_CONNECTION_POOL_SIZE,
    RuntimeOverlay, snapshot::ConfigSnapshot, snapshot::SnapshotHash, unwritten::UnwrittenConfig,
};
use crate::db::ConfigQueries;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::migration_manager::{self, RunMigrationManagerArgs};
use crate::db::delegating_connection::{DelegatingDatabaseConnection, PrimaryDatastore};
use crate::db::postgres::PostgresConnectionInfo;
use crate::db::postgres::batching::PostgresBatchSender;
use crate::db::rate_limiting::DisabledRateLimitQueries;
use crate::db::valkey::ValkeyConnectionInfo;
use crate::endpoints;
use crate::endpoints::openai_compatible::RouterExt;
use crate::error::{DelayedError, Error, ErrorDetails};
use crate::howdy::{get_deployment_id, setup_howdy};
use crate::http::TensorzeroHttpClient;
use crate::rate_limiting::{RateLimitingConfig, RateLimitingManager};
use autopilot_client::AutopilotClient;
use durable_tools_spawn::SpawnClient;

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
    pub app_state: SwappableAppStateData,
    drop_wrapper: Option<DropWrapper>,
    _private: (),
}

impl Drop for GatewayHandle {
    fn drop(&mut self) {
        let drop_wrapper = self.drop_wrapper.take();
        let mut drop_self = || {
            self.app_state.shutdown_token.cancel();
            let disabled_placeholder = self.app_state.disabled_for_shutdown_placeholder();
            let mut app_state = std::mem::replace(&mut self.app_state, disabled_placeholder);

            // Move the deferred task tracker out before dropping app state so we can
            // still close/wait on it below.
            let deferred_tasks =
                std::mem::replace(&mut app_state.deferred_tasks, TaskTracker::new());

            // Drop all remaining state before waiting on batch writers. This releases
            // all sender/reference holders (cache backends, rate-limit backends, and any
            // future fields added to `AppStateData`) without requiring manual `drop(...)`
            // calls for each one.
            drop(app_state);

            deferred_tasks.close();
            // The 'wait' future will resolve immediately if the pool is empty.
            // Closing the pool doesn't block more futures from being added, so checking
            // if it's empty doesn't introduce any new race conditions (it's still possible
            // for some existing tokio task to spawn something in this pool after 'wait' resolves).
            // This check makes it easier to write tests (since we don't need to use a multi-threaded runtime),
            // as well as reducing the number of log messages in the common case.
            if !deferred_tasks.is_empty() {
                tokio::task::block_in_place(|| {
                    tracing::info!("Waiting for deferred tasks to finish");
                    Handle::current().block_on(deferred_tasks.wait());
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

/// All hot-swappable state for a running gateway.
/// Stored inside `Arc<ArcSwap<LiveState>>` so config swaps are atomic.
///
/// Not `Clone` — if cloned, when dropped we will wait on the same database connection's batch writer multiple times,
/// which works but is logically wrong.
///
/// This only spawns tasks that *wait* for shutdown - the shutdown happens automatically
/// once all outstanding `ClickHouseConnectionInfo` and `PostgresConnectionInfo` handles are dropped.
/// It's therefore safe for us to hand out cloned `ClickHouseConnectionInfo` and `PostgresConnectionInfo` handles
/// from this struct.
struct LiveState {
    config: Arc<Config>,
    runtime_overlay: Arc<RuntimeOverlay>,
    http_client: TensorzeroHttpClient,
    valkey_connection_info: ValkeyConnectionInfo,
    /// Separate Valkey connection for model inference caching.
    valkey_cache_connection_info: ValkeyConnectionInfo,
    cache_manager: CacheManager,
    primary_datastore: PrimaryDatastore,
    /// Token pool manager for rate limiting pre-borrowing.
    /// Stored inside `LiveState` so that config hot-swaps are atomic:
    /// `swap_config` issues a single `ArcSwap::store`, ensuring every
    /// request sees either the old (config, rate-limiter) pair or the new
    /// one — never a mix of the two.
    rate_limiting_manager: Arc<RateLimitingManager>,
}

#[derive(Clone, Default)]
struct ConnectionUrls {
    valkey_url: Option<String>,
    valkey_cache_url: Option<String>,
}

/// A thin, cloneable handle that lets callers observe the latest `Config` snapshot
/// without holding a reference into `SwappableAppStateData`.
///
/// Internally shares the same `ArcSwap<LiveState>` as the owning `SwappableAppStateData`,
/// so every `load()` sees whatever the most recent `swap_config` published.
#[derive(Clone)]
pub struct SwappableConfig(Arc<ArcSwap<LiveState>>);

impl SwappableConfig {
    fn new(live_state: Arc<ArcSwap<LiveState>>) -> Self {
        Self(live_state)
    }

    pub fn load(&self) -> Arc<Config> {
        self.0.load().config.clone()
    }
}

/// Holds state that needs to have tasks spawned onto `deferred_tasks`
/// when dropped.
/// Currently, we use this to ensure that we wait for the batch writer handles to finish
/// when the gateway shuts down.
/// This only spawns tasks that *wait* for shutdown - the shutdown happens automatically
/// once all outstanding `ClickHouseConnectionInfo` and `PostgresConnectionInfo` handles are dropped.
/// It's therefore safe for us to hand out cloned `ClickHouseConnectionInfo` and `PostgresConnectionInfo` handles
/// from this struct.
struct DeferredShutdown {
    deferred_tasks: TaskTracker,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    postgres_connection_info: PostgresConnectionInfo,
}

impl Drop for DeferredShutdown {
    fn drop(&mut self) {
        if let Some(clickhouse_handle) = self.clickhouse_connection_info.batcher_join_handle() {
            self.deferred_tasks.spawn(async move {
                tracing::info!("Waiting for ClickHouse batch writer to finish");
                if let Err(e) = clickhouse_handle.await {
                    tracing::error!("Error in batch writer: {e}");
                }
                tracing::info!("ClickHouse batch writer finished");
            });
        }
        if let Some(postgres_handle) = self.postgres_connection_info.batcher_join_handle() {
            self.deferred_tasks.spawn(async move {
                tracing::info!("Waiting for Postgres batch writer to finish");
                if let Err(e) = postgres_handle.await {
                    tracing::error!("Error in batch writer: {e}");
                }
                tracing::info!("Postgres batch writer finished");
            });
        }
    }
}

#[derive(Clone)]
// `#[non_exhaustive]` only affects downstream crates, so we can't use it here
#[expect(clippy::manual_non_exhaustive)]
pub struct AppStateData {
    pub config: Arc<Config>,
    /// Runtime overlay captured from the original UninitializedConfig at startup.
    /// Used for snapshot rehydration without the lossy Config → UninitializedConfig round-trip.
    pub runtime_overlay: Arc<RuntimeOverlay>,
    pub http_client: TensorzeroHttpClient,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
    pub postgres_connection_info: PostgresConnectionInfo,
    pub valkey_connection_info: ValkeyConnectionInfo,
    /// Separate Valkey connection for model inference caching, allowing isolation from rate limiting keys.
    /// When `TENSORZERO_VALKEY_CACHE_URL` is set, this points to a dedicated instance;
    /// otherwise, it shares the same connection as `valkey_connection_info`.
    pub valkey_cache_connection_info: ValkeyConnectionInfo,
    pub cache_manager: CacheManager,
    /// Holds any background tasks that we want to wait on during shutdown
    /// We wait for these tasks to finish when `GatewayHandle` is dropped
    pub deferred_tasks: TaskTracker,
    /// Optional cache for TensorZero API key authentication
    pub auth_cache: Option<Cache<String, AuthResult>>,
    /// Optional cache for historical config snapshots loaded from ClickHouse
    pub config_snapshot_cache: Option<Cache<SnapshotHash, Arc<Config>>>,
    /// Optional Autopilot API client for proxying requests to the Autopilot API
    pub autopilot_client: Option<Arc<AutopilotClient>>,
    /// Optional durable task spawning client for GEPA workflows
    pub spawn_client: Option<Arc<SpawnClient>>,
    /// The deployment ID from ClickHouse (64-char hex string)
    pub deployment_id: Option<String>,
    /// Token pool manager for rate limiting pre-borrowing
    pub rate_limiting_manager: Arc<RateLimitingManager>,
    pub shutdown_token: CancellationToken,
    /// Which database backend is the primary datastore for observability data.
    /// Derived from config (`observability.backend`) at startup.
    pub primary_datastore: PrimaryDatastore,
    /// Whether the gateway config was loaded from the database (as opposed to a file on disk).
    /// Used by the UI to decide whether to show the config editor.
    pub config_in_database: bool,
    // Prevent `AppStateData` from being directly constructed outside of this module
    // This ensures that `AppStateData` is only ever constructed via explicit `new` methods,
    // which can ensure that we update global state.
    _private: (),
}

#[derive(Clone)]
pub struct SwappableAppStateData {
    live_state: Arc<ArcSwap<LiveState>>,
    connection_urls: Arc<ConnectionUrls>,
    /// Holds clickhouse and postgres handles, which are intentionally excluded from the swappable LiveState bundle.
    /// Hot-swapping them would interfere with the batch-writer drain logic in GatewayHandle::drop and may cause
    /// issues with connection pool sizes.
    deferred_shutdown: Arc<DeferredShutdown>,
    /// Holds any background tasks that we want to wait on during shutdown
    /// We wait for these tasks to finish when `GatewayHandle` is dropped
    pub deferred_tasks: TaskTracker,
    /// Optional cache for TensorZero API key authentication
    pub auth_cache: Option<Cache<String, AuthResult>>,
    /// Optional cache for historical config snapshots loaded from ClickHouse
    pub config_snapshot_cache: Option<Cache<SnapshotHash, Arc<Config>>>,
    /// Optional Autopilot API client for proxying requests to the Autopilot API
    pub autopilot_client: Option<Arc<AutopilotClient>>,
    /// Optional durable task spawning client for GEPA workflows
    pub spawn_client: Option<Arc<SpawnClient>>,
    /// The deployment ID from ClickHouse (64-char hex string)
    pub deployment_id: Option<String>,
    pub shutdown_token: CancellationToken,
    /// Whether the gateway config was loaded from the database (as opposed to a file on disk).
    /// Used by the UI to decide whether to show the config editor.
    pub config_in_database: bool,
}

/// `AppStateData` with a concrete config snapshot, used by route handlers and business logic.
pub type ResolvedAppStateData = AppStateData;

/// Axum extractor that loads the latest config from the `SwappableConfig`
/// and produces a `ResolvedAppStateData`.
pub type LatestAppStateData = axum::extract::State<AppStateData>;
pub type AppState = LatestAppStateData;

/// Opaque bundle produced by `SwappableAppStateData::prepare_config_swap`.
/// Holds everything needed for an infallible `swap_config` call.
/// Building this succeeds before any database transaction is committed,
/// so a failure here is still fully recoverable.
pub struct PreparedConfigSwap {
    config: Arc<Config>,
    runtime_overlay: Arc<RuntimeOverlay>,
    http_client: TensorzeroHttpClient,
    valkey_connection_info: ValkeyConnectionInfo,
    valkey_cache_connection_info: ValkeyConnectionInfo,
    cache_manager: CacheManager,
    primary_datastore: PrimaryDatastore,
    rate_limiting_manager: Arc<RateLimitingManager>,
}

impl PreparedConfigSwap {
    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }
}

impl SwappableAppStateData {
    /// A cloneable handle that observes the latest `Config` snapshot. Each call
    /// returns a fresh handle that shares the underlying `ArcSwap<LiveState>`,
    /// so any subsequent `swap_config` is visible to existing handles.
    pub fn config(&self) -> SwappableConfig {
        SwappableConfig::new(self.live_state.clone())
    }

    /// Builds new runtime dependencies for the incoming config, and writes a new
    /// config snapshot to the database. Returns an opaque [`PreparedConfigSwap`].
    ///
    /// Callers should invoke this **before** committing any surrounding
    /// database transaction so that a failure here can still be rolled back.
    /// Once the transaction commits, pass the result to [`Self::swap_config`],
    /// which is infallible.
    pub async fn prepare_config_swap(
        &self,
        unwritten: UnwrittenConfig,
        db: &impl ConfigQueries,
    ) -> Result<PreparedConfigSwap, DelayedError> {
        let (config, runtime_overlay) = Box::pin(unwritten.into_config(db)).await?;
        let config = Arc::new(config);
        let runtime_overlay = Arc::new(runtime_overlay);

        let valkey_connection_info =
            setup_valkey(self.connection_urls.valkey_url.as_deref()).await?;
        let valkey_cache_connection_info = setup_valkey_cache(
            self.connection_urls.valkey_cache_url.as_deref(),
            &valkey_connection_info,
        )
        .await?;
        let primary_datastore = PrimaryDatastore::resolve(
            &config.gateway.observability,
            &self.deferred_shutdown.clickhouse_connection_info,
            &self.deferred_shutdown.postgres_connection_info,
        )?;
        let cache_manager = CacheManager::new_from_connections(
            &valkey_cache_connection_info,
            &self.deferred_shutdown.clickhouse_connection_info,
            &config.gateway.cache,
            primary_datastore,
        )?;
        let rate_limiting_manager = Arc::new(RateLimitingManager::new_from_connections(
            Arc::new(config.rate_limiting.clone()),
            &valkey_connection_info,
            &self.deferred_shutdown.postgres_connection_info,
        )?);
        let http_client = config.http_client.clone();

        Ok(PreparedConfigSwap {
            config,
            runtime_overlay,
            http_client,
            valkey_connection_info,
            valkey_cache_connection_info,
            cache_manager,
            primary_datastore,
            rate_limiting_manager,
        })
    }

    /// Atomically hot-swap the in-memory `Config` snapshot and runtime
    /// dependencies. Old dependencies (DB pools, batch writers, rate limiter,
    /// cache manager, HTTP client) remain alive for any in-flight requests
    /// that have already resolved a snapshot and will be dropped once those
    /// requests complete.
    ///
    /// This is infallible; all fallible work is done up front in
    /// [`Self::prepare_config_swap`].
    pub fn swap_config(&self, prepared: PreparedConfigSwap) {
        self.live_state.store(Arc::new(LiveState {
            config: prepared.config,
            runtime_overlay: prepared.runtime_overlay,
            http_client: prepared.http_client,
            valkey_connection_info: prepared.valkey_connection_info,
            valkey_cache_connection_info: prepared.valkey_cache_connection_info,
            cache_manager: prepared.cache_manager,
            primary_datastore: prepared.primary_datastore,
            rate_limiting_manager: prepared.rate_limiting_manager,
        }));
    }

    /// Load the latest config snapshot, producing a concrete `AppStateData`.
    pub fn load_latest(&self) -> AppStateData {
        let live_state = self.live_state.load_full();
        AppStateData {
            config: live_state.config.clone(),
            runtime_overlay: live_state.runtime_overlay.clone(),
            http_client: live_state.http_client.clone(),
            clickhouse_connection_info: self.deferred_shutdown.clickhouse_connection_info.clone(),
            postgres_connection_info: self.deferred_shutdown.postgres_connection_info.clone(),
            valkey_connection_info: live_state.valkey_connection_info.clone(),
            valkey_cache_connection_info: live_state.valkey_cache_connection_info.clone(),
            cache_manager: live_state.cache_manager.clone(),
            deferred_tasks: self.deferred_tasks.clone(),
            auth_cache: self.auth_cache.clone(),
            config_snapshot_cache: self.config_snapshot_cache.clone(),
            autopilot_client: self.autopilot_client.clone(),
            spawn_client: self.spawn_client.clone(),
            deployment_id: self.deployment_id.clone(),
            rate_limiting_manager: live_state.rate_limiting_manager.clone(),
            shutdown_token: self.shutdown_token.clone(),
            primary_datastore: live_state.primary_datastore,
            config_in_database: self.config_in_database,
            _private: (),
        }
    }

    pub fn primary_datastore(&self) -> PrimaryDatastore {
        self.live_state.load().primary_datastore
    }

    pub fn postgres_connection_info(&self) -> PostgresConnectionInfo {
        self.deferred_shutdown.postgres_connection_info.clone()
    }

    pub fn clickhouse_connection_info(&self) -> ClickHouseConnectionInfo {
        self.deferred_shutdown.clickhouse_connection_info.clone()
    }

    pub fn valkey_connection_info(&self) -> ValkeyConnectionInfo {
        self.live_state.load().valkey_connection_info.clone()
    }

    pub fn valkey_cache_connection_info(&self) -> ValkeyConnectionInfo {
        self.live_state.load().valkey_cache_connection_info.clone()
    }

    pub fn http_client(&self) -> TensorzeroHttpClient {
        self.live_state.load().http_client.clone()
    }

    pub fn rate_limiting_manager(&self) -> Arc<RateLimitingManager> {
        self.live_state.load().rate_limiting_manager.clone()
    }
}

impl FromRef<SwappableAppStateData> for AppStateData {
    fn from_ref(state: &SwappableAppStateData) -> Self {
        state.load_latest()
    }
}

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

    if !cache_config
        .enabled
        .unwrap_or_else(default_gateway_auth_cache_enabled)
    {
        return None;
    }

    Some(
        Cache::builder()
            .time_to_live(Duration::from_millis(
                cache_config
                    .ttl_ms
                    .unwrap_or_else(default_gateway_auth_cache_ttl_ms),
            ))
            .build(),
    )
}

impl GatewayHandle {
    pub async fn new(
        config: UnwrittenConfig,
        available_tools: HashSet<String>,
        tool_whitelist: HashSet<String>,
        config_in_database: bool,
    ) -> Result<Self, DelayedError> {
        let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL").ok();
        let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").ok();
        let valkey_url = std::env::var("TENSORZERO_VALKEY_URL").ok();
        let valkey_cache_url = std::env::var("TENSORZERO_VALKEY_CACHE_URL").ok();
        Box::pin(Self::new_with_databases(
            config,
            clickhouse_url,
            postgres_url,
            valkey_url,
            valkey_cache_url,
            available_tools,
            tool_whitelist,
            config_in_database,
        ))
        .await
    }

    #[expect(clippy::too_many_arguments)]
    async fn new_with_databases(
        config: UnwrittenConfig,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
        valkey_url: Option<String>,
        valkey_cache_url: Option<String>,
        available_tools: HashSet<String>,
        tool_whitelist: HashSet<String>,
        config_in_database: bool,
    ) -> Result<Self, DelayedError> {
        let clickhouse_connection_info = setup_clickhouse(&config, clickhouse_url.clone()).await?;
        let postgres_connection_info = setup_postgres(&config, postgres_url.as_deref()).await?;

        let primary_datastore = PrimaryDatastore::resolve(
            &config.gateway.observability,
            &clickhouse_connection_info,
            &postgres_connection_info,
        )?;
        let db = DelegatingDatabaseConnection::new(
            clickhouse_connection_info.clone(),
            postgres_connection_info.clone(),
            primary_datastore,
        );
        let (config, runtime_overlay) = Box::pin(config.into_config(&db)).await?;
        let config = Arc::new(config);
        let runtime_overlay = Arc::new(runtime_overlay);
        let valkey_connection_info = setup_valkey(valkey_url.as_deref()).await?;
        let valkey_cache_connection_info =
            setup_valkey_cache(valkey_cache_url.as_deref(), &valkey_connection_info).await?;
        let http_client = config.http_client.clone();
        Self::new_with_database_and_http_client_and_urls(
            config,
            runtime_overlay,
            clickhouse_connection_info,
            postgres_connection_info,
            valkey_connection_info,
            valkey_cache_connection_info,
            http_client,
            ConnectionUrls {
                valkey_url,
                valkey_cache_url,
            },
            None,
            available_tools,
            tool_whitelist,
            config_in_database,
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
            .expect("Should be able to construct RateLimitingManager"),
        );
        let cache_manager = CacheManager::new_from_connections(
            &ValkeyConnectionInfo::Disabled,
            &clickhouse_connection_info,
            &config.gateway.cache,
            PrimaryDatastore::ClickHouse,
        )
        .expect("Should be able to construct CacheManager");
        let live_state = Arc::new(ArcSwap::from_pointee(LiveState {
            config: config.clone(),
            runtime_overlay: Arc::new(RuntimeOverlay::default()),
            http_client,
            valkey_connection_info: ValkeyConnectionInfo::Disabled,
            valkey_cache_connection_info: ValkeyConnectionInfo::Disabled,
            cache_manager,
            primary_datastore: PrimaryDatastore::ClickHouse,
            rate_limiting_manager,
        }));
        let deferred_tasks = TaskTracker::new();
        Self {
            app_state: SwappableAppStateData {
                live_state,
                connection_urls: Arc::new(ConnectionUrls::default()),
                deferred_shutdown: Arc::new(DeferredShutdown {
                    deferred_tasks: deferred_tasks.clone(),
                    clickhouse_connection_info,
                    postgres_connection_info,
                }),
                deferred_tasks,
                auth_cache,
                config_snapshot_cache: None,
                autopilot_client: None,
                spawn_client: None,
                deployment_id: None,
                shutdown_token: cancel_token,
                config_in_database: false,
            },
            drop_wrapper: None,
            _private: (),
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn new_with_database_and_http_client(
        config: Arc<Config>,
        runtime_overlay: Arc<RuntimeOverlay>,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        valkey_connection_info: ValkeyConnectionInfo,
        valkey_cache_connection_info: ValkeyConnectionInfo,
        http_client: TensorzeroHttpClient,
        drop_wrapper: Option<DropWrapper>,
        available_tools: HashSet<String>,
        tool_whitelist: HashSet<String>,
        config_in_database: bool,
    ) -> Result<Self, DelayedError> {
        Self::new_with_database_and_http_client_and_urls(
            config,
            runtime_overlay,
            clickhouse_connection_info,
            postgres_connection_info,
            valkey_connection_info,
            valkey_cache_connection_info,
            http_client,
            ConnectionUrls::default(),
            drop_wrapper,
            available_tools,
            tool_whitelist,
            config_in_database,
        )
        .await
    }

    #[expect(clippy::too_many_arguments)]
    async fn new_with_database_and_http_client_and_urls(
        config: Arc<Config>,
        runtime_overlay: Arc<RuntimeOverlay>,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        valkey_connection_info: ValkeyConnectionInfo,
        valkey_cache_connection_info: ValkeyConnectionInfo,
        http_client: TensorzeroHttpClient,
        connection_urls: ConnectionUrls,
        drop_wrapper: Option<DropWrapper>,
        available_tools: HashSet<String>,
        tool_whitelist: HashSet<String>,
        config_in_database: bool,
    ) -> Result<Self, DelayedError> {
        let rate_limiting_manager = Arc::new(RateLimitingManager::new_from_connections(
            Arc::new(config.rate_limiting.clone()),
            &valkey_connection_info,
            &postgres_connection_info,
        )?);
        let primary_datastore = PrimaryDatastore::resolve(
            &config.gateway.observability,
            &clickhouse_connection_info,
            &postgres_connection_info,
        )?;
        let cache_manager = CacheManager::new_from_connections(
            &valkey_cache_connection_info,
            &clickhouse_connection_info,
            &config.gateway.cache,
            primary_datastore,
        )?;

        let cancel_token = CancellationToken::new();
        setup_howdy(
            &config,
            clickhouse_connection_info.clone(),
            postgres_connection_info.clone(),
            primary_datastore,
            cancel_token.clone(),
        );

        let deployment_id = if primary_datastore == PrimaryDatastore::Disabled {
            None
        } else {
            get_deployment_id(
                &clickhouse_connection_info,
                &postgres_connection_info,
                primary_datastore,
            )
            .await
            .ok()
        };

        let db = Arc::new(DelegatingDatabaseConnection::new(
            clickhouse_connection_info.clone(),
            postgres_connection_info.clone(),
            primary_datastore,
        ));
        for (function_name, function_config) in &config.functions {
            let experimentation = function_config.experimentation_with_namespaces();
            experimentation
                .base
                .setup(
                    db.clone(),
                    function_name,
                    &postgres_connection_info,
                    cancel_token.clone(),
                )
                .await?;
            for namespace_config in experimentation.namespaces.values() {
                namespace_config
                    .setup(
                        db.clone(),
                        function_name,
                        &postgres_connection_info,
                        cancel_token.clone(),
                    )
                    .await?;
            }
        }
        let auth_cache = create_auth_cache_from_config(&config);

        let config_snapshot_cache = Some(
            Cache::builder()
                .time_to_live(Duration::from_secs(300))
                .max_capacity(10)
                .build(),
        );

        let unknown_whitelist_tools: Vec<&str> = tool_whitelist
            .iter()
            .filter(|name| !available_tools.contains(name.as_str()))
            .map(|s| s.as_str())
            .collect();
        if !unknown_whitelist_tools.is_empty() {
            return Err(DelayedError::new(ErrorDetails::AppState {
                message: format!(
                    "Unknown tool names in `autopilot.tool_whitelist`: {unknown_whitelist_tools:?}. \
                     These tools do not exist and will never be auto-approved. \
                     Check for typos in your configuration."
                ),
            }));
        }

        let spawn_client = if let Some(pool) = postgres_connection_info.get_pool() {
            let queue_name = std::env::var("TENSORZERO_AUTOPILOT_QUEUE_NAME")
                .unwrap_or_else(|_| "autopilot".to_string());
            match SpawnClient::builder()
                .pool(pool.clone())
                .queue_name(&queue_name)
                .build()
                .await
            {
                Ok(client) => Some(Arc::new(client)),
                Err(e) => {
                    tracing::warn!("Failed to create `SpawnClient`: {e}");
                    None
                }
            }
        } else {
            None
        };

        let autopilot_client = setup_autopilot_client(
            &postgres_connection_info,
            deployment_id.as_ref(),
            available_tools,
            tool_whitelist,
        )
        .await?;

        if config.gateway.auth.enabled
            && matches!(postgres_connection_info, PostgresConnectionInfo::Disabled)
        {
            return Err(DelayedError::new(ErrorDetails::AppState {
                message:
                    "Authentication is enabled (`gateway.auth.enabled = true`) but Postgres is not available. \
                     Authentication requires Postgres. Set `TENSORZERO_POSTGRES_URL` or disable auth."
                        .to_string(),
            }));
        }

        let live_state = Arc::new(ArcSwap::from_pointee(LiveState {
            config: config.clone(),
            runtime_overlay,
            http_client,
            valkey_connection_info,
            valkey_cache_connection_info,
            cache_manager,
            primary_datastore,
            rate_limiting_manager,
        }));
        let deferred_tasks = TaskTracker::new();
        Ok(Self {
            app_state: SwappableAppStateData {
                live_state,
                connection_urls: Arc::new(connection_urls),
                deferred_shutdown: Arc::new(DeferredShutdown {
                    deferred_tasks: deferred_tasks.clone(),
                    clickhouse_connection_info,
                    postgres_connection_info,
                }),
                deferred_tasks,
                auth_cache,
                config_snapshot_cache,
                autopilot_client,
                spawn_client,
                deployment_id,
                shutdown_token: cancel_token,
                config_in_database,
            },
            drop_wrapper,
            _private: (),
        })
    }
}

impl SwappableAppStateData {
    /// Returns a new AppStateData with all connections disabled. This is only used in
    /// `GatewayHandle::drop` so we can wait for batch writer handles without worrying
    /// about anything else in AppStateData holding a database connection.
    fn disabled_for_shutdown_placeholder(&self) -> Self {
        let current = self.live_state.load_full();
        let live_state = Arc::new(ArcSwap::from_pointee(LiveState {
            config: current.config.clone(),
            runtime_overlay: current.runtime_overlay.clone(),
            http_client: current.config.http_client.clone(),
            valkey_connection_info: ValkeyConnectionInfo::Disabled,
            valkey_cache_connection_info: ValkeyConnectionInfo::Disabled,
            cache_manager: CacheManager::disabled(),
            primary_datastore: current.primary_datastore,
            rate_limiting_manager: Arc::new(RateLimitingManager::new(
                Arc::new(RateLimitingConfig::default()),
                Arc::new(DisabledRateLimitQueries),
            )),
        }));
        Self {
            live_state,
            connection_urls: Arc::new(ConnectionUrls::default()),
            deferred_shutdown: Arc::new(DeferredShutdown {
                deferred_tasks: TaskTracker::new(),
                clickhouse_connection_info: ClickHouseConnectionInfo::new_disabled(),
                postgres_connection_info: PostgresConnectionInfo::new_disabled(),
            }),
            deferred_tasks: TaskTracker::new(),
            auth_cache: None,
            config_snapshot_cache: None,
            autopilot_client: None,
            spawn_client: None,
            deployment_id: None,
            shutdown_token: CancellationToken::new(),
            config_in_database: self.config_in_database,
        }
    }

    pub fn get_delegating_database(&self) -> DelegatingDatabaseConnection {
        let live_state = self.live_state.load_full();
        DelegatingDatabaseConnection::new(
            self.deferred_shutdown.clickhouse_connection_info.clone(),
            self.deferred_shutdown.postgres_connection_info.clone(),
            live_state.primary_datastore,
        )
    }
}

impl AppStateData {
    pub fn get_delegating_database(&self) -> DelegatingDatabaseConnection {
        DelegatingDatabaseConnection::new(
            self.clickhouse_connection_info.clone(),
            self.postgres_connection_info.clone(),
            self.primary_datastore,
        )
    }
}

impl AppStateData {
    /// Validate a config snapshot and write it to the database.
    ///
    /// This is the single entry point for writing config snapshots. It validates
    /// the config by running the full loading pipeline (with credential validation
    /// disabled) before persisting, ensuring invalid configs (e.g. missing templates
    /// for JSON schema functions) are rejected.
    pub async fn validate_and_write_config_snapshot(
        &self,
        snapshot: &ConfigSnapshot,
    ) -> Result<(), crate::error::Error> {
        Config::load_from_snapshot(snapshot.clone(), (*self.runtime_overlay).clone(), false)
            .await?;

        let db = self.get_delegating_database();
        #[expect(clippy::disallowed_methods)]
        db.write_config_snapshot(snapshot)
            .await
            .map_err(|e| e.log())
    }

    /// Create an AppStateData for use with a historical config snapshot.
    /// This version does not include auth_cache, config_snapshot_cache, autopilot_client,
    /// or deployment_id since those are specific to the live gateway.
    #[expect(clippy::too_many_arguments)]
    pub fn new_for_snapshot(
        config: Arc<Config>,
        runtime_overlay: Arc<RuntimeOverlay>,
        http_client: TensorzeroHttpClient,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        valkey_connection_info: ValkeyConnectionInfo,
        valkey_cache_connection_info: ValkeyConnectionInfo,
        deferred_tasks: TaskTracker,
        shutdown_token: CancellationToken,
        primary_datastore: PrimaryDatastore,
    ) -> Result<Self, Error> {
        let rate_limiting_manager = Arc::new(
            RateLimitingManager::new_from_connections(
                Arc::new(config.rate_limiting.clone()),
                &valkey_connection_info,
                &postgres_connection_info,
            )
            .map_err(|e| e.log())?,
        );
        let cache_manager = CacheManager::new_from_connections(
            &valkey_cache_connection_info,
            &clickhouse_connection_info,
            &config.gateway.cache,
            primary_datastore,
        )
        .map_err(|e| e.log())?;
        Ok(Self {
            config,
            runtime_overlay,
            http_client,
            valkey_connection_info,
            valkey_cache_connection_info,
            cache_manager,
            deferred_tasks,
            clickhouse_connection_info,
            postgres_connection_info,
            auth_cache: None,
            config_snapshot_cache: None,
            autopilot_client: None,
            spawn_client: None,
            deployment_id: None,
            rate_limiting_manager,
            shutdown_token,
            primary_datastore,
            config_in_database: false,
            _private: (),
        })
    }
}

impl SwappableAppStateData {
    /// Validate a config snapshot and write it to the database.
    ///
    /// Delegates to the `AppStateData<Arc<Config>>` implementation after loading
    /// the latest config snapshot.
    pub async fn validate_and_write_config_snapshot(
        &self,
        snapshot: &ConfigSnapshot,
    ) -> Result<(), crate::error::Error> {
        self.load_latest()
            .validate_and_write_config_snapshot(snapshot)
            .await
    }
}

pub async fn setup_clickhouse_without_config(
    clickhouse_url: String,
) -> Result<ClickHouseConnectionInfo, Error> {
    setup_clickhouse(
        &Box::pin(Config::new_empty())
            .await?
            .dangerous_into_config_without_writing(),
        Some(clickhouse_url),
    )
    .await
    .map_err(|e| e.log())
}

pub async fn setup_clickhouse(
    config: &Config,
    clickhouse_url: Option<String>,
) -> Result<ClickHouseConnectionInfo, DelayedError> {
    // TODO(#5691): we should stop checking an explicit observability.enabled config when setting up
    // ClickHouse.
    let clickhouse_connection_info = match (config.gateway.observability.enabled, clickhouse_url) {
        (Some(false), _) => {
            // Observability disabled by config
            tracing::info!(
                "Disabling ClickHouse: `gateway.observability.enabled` is set to false in config."
            );
            ClickHouseConnectionInfo::new_disabled()
        }
        (Some(true), None) => {
            // Observability enabled but no ClickHouse URL
            // This is allowed if Postgres is the primary datastore; we validate after both ClickHouse
            // and Postgres are initialized.
            ClickHouseConnectionInfo::new_disabled()
        }
        // Observability enabled and ClickHouse URL provided
        (Some(true), Some(clickhouse_url)) => {
            ClickHouseConnectionInfo::new(
                &clickhouse_url,
                config
                    .gateway
                    .observability
                    .batch_writes
                    .clone()
                    .unwrap_or_default(),
            )
            .await?
        }
        // Observability default and no ClickHouse URL
        (None, None) => {
            tracing::debug!("Disabling ClickHouse: `TENSORZERO_CLICKHOUSE_URL` is not set.");
            ClickHouseConnectionInfo::new_disabled()
        }
        // Observability default and ClickHouse URL provided
        (None, Some(clickhouse_url)) => {
            ClickHouseConnectionInfo::new(
                &clickhouse_url,
                config
                    .gateway
                    .observability
                    .batch_writes
                    .clone()
                    .unwrap_or_default(),
            )
            .await?
        }
    };

    // Run ClickHouse migrations (if any) if we have a production ClickHouse connection
    if clickhouse_connection_info.client_type() == ClickHouseClientType::Production {
        migration_manager::run(RunMigrationManagerArgs {
            clickhouse: &clickhouse_connection_info,
            is_manual_run: false,
            disable_automatic_migrations: config
                .clickhouse
                .disable_automatic_migrations
                .unwrap_or(false),
        })
        .await?;
    }
    Ok(clickhouse_connection_info)
}

async fn create_postgres_connection(
    postgres_url: &str,
    connection_pool_size: u32,
    batch_writes: &BatchWritesConfig,
) -> Result<PostgresConnectionInfo, DelayedError> {
    let connect_options: PgConnectOptions = postgres_url.parse().map_err(|err: sqlx::Error| {
        DelayedError::new(ErrorDetails::PostgresConnectionInitialization {
            message: err.to_string(),
        })
    })?;
    // Demote sqlx's built-in statement logging from INFO to TRACE.
    // This avoids flooding logs with full SQL (including thousands of bind params)
    // during bulk INSERT operations.
    let connect_options = connect_options.log_statements(tracing::log::LevelFilter::Trace);

    let pool = PgPoolOptions::new()
        .max_connections(connection_pool_size)
        .connect_with(connect_options)
        .await
        .map_err(|err| {
            DelayedError::new(ErrorDetails::PostgresConnectionInitialization {
                message: err.to_string(),
            })
        })?;

    let connection_info = if batch_writes.enabled {
        let batch_sender = Arc::new(PostgresBatchSender::new(
            pool.clone(),
            batch_writes.clone(),
        )?);
        tracing::debug!(
            write_queue_capacity = ?batch_writes.write_queue_capacity,
            "Postgres batch writer enabled"
        );
        PostgresConnectionInfo::new_with_pool_and_batcher(pool, batch_sender)
    } else {
        PostgresConnectionInfo::new_with_pool(pool)
    };
    connection_info.check_migrations().await?;
    Ok(connection_info)
}

// TODO(#5764): We should test that on startup we issue the correct SQL for write_retention_config,
// but this is currently structured that's difficult to swap in a Mock.
#[expect(deprecated)]
pub async fn setup_postgres(
    config: &Config,
    postgres_url: Option<&str>,
) -> Result<PostgresConnectionInfo, DelayedError> {
    if config.postgres.enabled.is_some() {
        crate::utils::deprecation_warning(
            "`postgres.enabled` is deprecated (2026.3+) and will be removed in a future release. \
             Postgres connectivity is now determined by the `TENSORZERO_POSTGRES_URL` environment variable. \
             Remove `postgres.enabled` from your config.",
        );
    }

    let postgres_connection_info = match (config.postgres.enabled, postgres_url) {
        // Postgres disabled by config (deprecated)
        (Some(false), _) => {
            tracing::info!("Disabling Postgres: `postgres.enabled` is set to false in config.");
            PostgresConnectionInfo::Disabled
        }
        // Postgres enabled but no URL (deprecated)
        (Some(true), None) => {
            return Err(DelayedError::new(ErrorDetails::AppState {
                message: "Missing environment variable `TENSORZERO_POSTGRES_URL`.".to_string(),
            }));
        }
        // Postgres enabled and URL provided (deprecated)
        (Some(true), Some(postgres_url)) => {
            create_postgres_connection(
                postgres_url,
                config
                    .postgres
                    .connection_pool_size
                    .unwrap_or(DEFAULT_POSTGRES_CONNECTION_POOL_SIZE),
                &config
                    .gateway
                    .observability
                    .batch_writes
                    .clone()
                    .unwrap_or_default(),
            )
            .await?
        }
        // Postgres default and no URL
        (None, None) => {
            tracing::debug!("Disabling Postgres: `TENSORZERO_POSTGRES_URL` is not set.");
            PostgresConnectionInfo::Disabled
        }
        // Postgres default and URL provided
        (None, Some(postgres_url)) => {
            create_postgres_connection(
                postgres_url,
                config
                    .postgres
                    .connection_pool_size
                    .unwrap_or(DEFAULT_POSTGRES_CONNECTION_POOL_SIZE),
                &config
                    .gateway
                    .observability
                    .batch_writes
                    .clone()
                    .unwrap_or_default(),
            )
            .await?
        }
    };

    // Write retention config to Postgres (syncs tensorzero.toml -> database)
    postgres_connection_info
        .write_retention_config(
            config.postgres.inference_metadata_retention_days,
            config.postgres.inference_data_retention_days,
        )
        .await?;

    Ok(postgres_connection_info)
}

/// Sets up the Valkey connection from the provided URL.
///
/// Valkey is optional; if no URL is provided, rate limiting will fall back to Postgres.
///
/// # Arguments
/// * `valkey_url` - Optional Valkey URL (from `TENSORZERO_VALKEY_URL` env var)
pub async fn setup_valkey(valkey_url: Option<&str>) -> Result<ValkeyConnectionInfo, DelayedError> {
    match valkey_url {
        Some(url) => ValkeyConnectionInfo::new(url).await,
        None => {
            tracing::debug!("Disabling Valkey: `TENSORZERO_VALKEY_URL` is not set.");
            Ok(ValkeyConnectionInfo::Disabled)
        }
    }
}

/// Sets up the Valkey connection for model inference caching.
///
/// If `valkey_cache_url` is provided, creates a dedicated connection for caching.
/// Otherwise, falls back to the shared `valkey_connection_info`.
///
/// A dedicated caching Valkey instance allows operators to use eviction policies like
/// `allkeys-lru` or `volatile-ttl` for cache entries while keeping the main instance
/// configured with `noeviction` to protect rate limiting keys.
pub async fn setup_valkey_cache(
    valkey_cache_url: Option<&str>,
    valkey_connection_info: &ValkeyConnectionInfo,
) -> Result<ValkeyConnectionInfo, DelayedError> {
    match valkey_cache_url {
        Some(url) => {
            tracing::info!(
                "Using dedicated Valkey instance for caching (`TENSORZERO_VALKEY_CACHE_URL` is set)."
            );
            ValkeyConnectionInfo::new_cache_only(url).await
        }
        None => Ok(valkey_connection_info.clone()),
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
    tool_whitelist: HashSet<String>,
) -> Result<Option<Arc<AutopilotClient>>, DelayedError> {
    match std::env::var("TENSORZERO_AUTOPILOT_API_KEY") {
        Ok(api_key) => {
            let pool = postgres_connection_info.get_pool().ok_or_else(|| {
                DelayedError::new(ErrorDetails::AppState {
                    message: "Autopilot client requires Postgres; set `TENSORZERO_POSTGRES_URL`."
                        .to_string(),
                })
            })?;

            // Require `deployment_id` (from ClickHouse) for autopilot
            if deployment_id.is_none() {
                return Err(DelayedError::new(ErrorDetails::AppState {
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
                .available_tools(available_tools)
                .tool_whitelist(tool_whitelist)
                .deployment_id(deployment_id.cloned().unwrap_or_default())
                .tensorzero_version(crate::endpoints::status::TENSORZERO_VERSION.to_string());

            // Allow custom base URL for testing
            if let Ok(base_url) = std::env::var("TENSORZERO_AUTOPILOT_BASE_URL") {
                let url = base_url.parse().map_err(|e| {
                    DelayedError::new(ErrorDetails::AppState {
                        message: format!("Invalid TENSORZERO_AUTOPILOT_BASE_URL: {e}"),
                    })
                })?;
                builder = builder.base_url(url);
                tracing::info!("Autopilot client using custom base URL: {}", base_url);
            }

            let client = builder.build().await.map_err(|e| {
                DelayedError::new(ErrorDetails::AppState {
                    message: format!("Failed to build autopilot client: {e}"),
                })
            })?;
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
        Err(std::env::VarError::NotUnicode(_)) => Err(DelayedError::new(ErrorDetails::AppState {
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
/// Returns the address the gateway is listening on and a `ShutdownHandle` which shuts down
/// the gateway when dropped.
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
        None, // Embedded gateways use the same Valkey instance for rate limiting and caching
        HashSet::new(), // available_tools
        HashSet::new(), // tool_whitelist
        false,
    ))
    .await
    .map_err(|e| e.log())?;

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
        ObservabilityBackend, ObservabilityConfig, PostgresConfig, UninitializedConfig,
        gateway::{GatewayConfig, ModelInferenceCacheConfig},
        snapshot::ConfigSnapshot,
        unwritten::UnwrittenConfig,
    };
    #[tokio::test]
    async fn test_setup_clickhouse() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Disabled observability
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(false),
                backend: Some(ObservabilityBackend::Auto),
                async_writes: Some(false),
                batch_writes: Default::default(),
                ..Default::default()
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
            global_outbound_http_intra_stream_read_timeout: None,
            relay: None,
            metrics: Default::default(),
            cache: Default::default(),
        };

        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let config = UnwrittenConfig::new(
            config,
            UninitializedConfig::default(),
            ConfigSnapshot::new_empty_for_test(),
            RuntimeOverlay::default(),
        );

        let clickhouse_connection_info = setup_clickhouse(&config, None).await.unwrap();
        assert_eq!(
            clickhouse_connection_info.client_type(),
            ClickHouseClientType::Disabled
        );
        assert!(!logs_contain(
            "Missing environment variable `TENSORZERO_CLICKHOUSE_URL`"
        ));

        // Default observability and no ClickHouse URL
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: None,
                backend: Some(ObservabilityBackend::Auto),
                async_writes: Some(false),
                batch_writes: Default::default(),
                ..Default::default()
            },
            fetch_and_encode_input_files_before_inference: false,
            unstable_error_json: false,
            ..Default::default()
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(
            config,
            UninitializedConfig::default(),
            ConfigSnapshot::new_empty_for_test(),
            RuntimeOverlay::default(),
        );
        let clickhouse_connection_info = setup_clickhouse(&unwritten_config, None).await.unwrap();
        assert_eq!(
            clickhouse_connection_info.client_type(),
            ClickHouseClientType::Disabled
        );
        assert!(!logs_contain(
            "Missing environment variable `TENSORZERO_CLICKHOUSE_URL`"
        ));
        assert!(!logs_contain("Disabling observability"));

        // We do not test the case where a ClickHouse URL is provided but observability is default,
        // as this would require a working ClickHouse and we don't have one in unit tests.

        // Observability enabled but ClickHouse URL is missing returns disabled
        // (validation happens in `new_with_database_and_http_client`)
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
                backend: Some(ObservabilityBackend::Auto),
                async_writes: Some(false),
                batch_writes: Default::default(),
                ..Default::default()
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
            global_outbound_http_intra_stream_read_timeout: None,
            relay: None,
            metrics: Default::default(),
            cache: Default::default(),
        };

        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(
            config,
            UninitializedConfig::default(),
            ConfigSnapshot::new_empty_for_test(),
            RuntimeOverlay::default(),
        );

        let clickhouse_connection_info = setup_clickhouse(&unwritten_config, None).await.unwrap();
        assert_eq!(
            clickhouse_connection_info.client_type(),
            ClickHouseClientType::Disabled,
            "ClickHouse should be disabled when observability is enabled but no URL is provided"
        );

        // Bad URL
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
                backend: Some(ObservabilityBackend::Auto),
                async_writes: Some(false),
                batch_writes: Default::default(),
                ..Default::default()
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
            global_outbound_http_intra_stream_read_timeout: None,
            relay: None,
            metrics: Default::default(),
            cache: Default::default(),
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(
            config,
            UninitializedConfig::default(),
            ConfigSnapshot::new_empty_for_test(),
            RuntimeOverlay::default(),
        );
        setup_clickhouse(&unwritten_config, Some("bad_url".to_string()))
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
                backend: Some(ObservabilityBackend::Auto),
                async_writes: Some(false),
                batch_writes: Default::default(),
                ..Default::default()
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
            global_outbound_http_intra_stream_read_timeout: None,
            relay: None,
            metrics: Default::default(),
            cache: Default::default(),
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        let unwritten_config = UnwrittenConfig::new(
            config,
            UninitializedConfig::default(),
            ConfigSnapshot::new_empty_for_test(),
            RuntimeOverlay::default(),
        );
        setup_clickhouse(
            &unwritten_config,
            Some("https://tensorzero.invalid:8123".to_string()),
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
    #[expect(deprecated)]
    async fn test_setup_postgres_disabled() {
        let logs_contain = crate::utils::testing::capture_logs();

        // Postgres disabled by config
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(false),
                ..Default::default()
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
                ..Default::default()
            },
            ..Default::default()
        }));

        let postgres_connection_info =
            setup_postgres(config, Some("postgresql://user:pass@localhost:5432/db"))
                .await
                .unwrap();
        assert!(matches!(
            postgres_connection_info,
            PostgresConnectionInfo::Disabled
        ));
    }

    #[tokio::test]
    #[expect(deprecated)]
    async fn test_setup_postgres_default_no_url() {
        // Default postgres config (enabled: None) and no URL
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: None,
                ..Default::default()
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
    #[expect(deprecated)]
    async fn test_setup_postgres_enabled_no_url() {
        // Postgres enabled but URL is missing
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(true),
                ..Default::default()
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
    #[expect(deprecated)]
    async fn test_setup_postgres_bad_url() {
        // Postgres enabled with bad URL
        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(true),
                ..Default::default()
            },
            ..Default::default()
        }));

        setup_postgres(config, Some("bad_url"))
            .await
            .expect_err("Postgres setup should fail given a bad URL");
    }

    #[tokio::test]
    async fn test_observability_enabled_requires_clickhouse_when_not_postgres_primary() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                observability: ObservabilityConfig {
                    enabled: Some(true),
                    backend: Some(ObservabilityBackend::ClickHouse),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let result = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await;
        let err = result
            .err()
            .expect("Gateway should fail when observability is enabled but ClickHouse is missing");
        assert!(
            err.to_string()
                .contains("Missing environment variable `TENSORZERO_CLICKHOUSE_URL`"),
            "error should mention the missing ClickHouse URL: {err}"
        );
    }

    #[tokio::test]
    async fn test_observability_enabled_requires_postgres_when_postgres_primary() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                observability: ObservabilityConfig {
                    enabled: Some(true),
                    backend: Some(ObservabilityBackend::Postgres),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let result = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await;
        let err = result
            .err()
            .expect("Gateway should fail when Postgres is primary but disabled");
        assert!(
            err.to_string().contains("Postgres") && err.to_string().contains("primary datastore"),
            "error should mention that Postgres is the primary datastore: {err}"
        );
    }

    #[tokio::test]
    async fn test_observability_disabled_does_not_require_datastore() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                observability: ObservabilityConfig {
                    enabled: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let _gateway = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await
        .expect("Gateway should start when observability is disabled");
    }

    #[tokio::test]
    async fn test_observability_default_does_not_require_datastore() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                observability: ObservabilityConfig {
                    enabled: None,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let _gateway = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await
        .expect("Gateway should start when observability is default (not explicitly enabled)");
    }

    #[tokio::test]
    #[expect(deprecated)]
    async fn test_no_rate_limiting_does_not_require_postgres_or_valkey() {
        // Rate limiting enabled=false should not fail validation (no rules configured)
        let config_no_rules = Arc::new(Config {
            postgres: PostgresConfig {
                enabled: Some(false),
                ..Default::default()
            },
            rate_limiting: Default::default(),
            ..Default::default()
        });

        let http_client = TensorzeroHttpClient::new_testing().unwrap();

        // This should succeed because rate limiting has no rules
        let _gateway = GatewayHandle::new_with_database_and_http_client(
            config_no_rules,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(), // available_tools
            HashSet::new(), // tool_whitelist
            false,
        )
        .await
        .expect("Gateway setup should succeed when rate limiting has no rules");
    }

    #[tokio::test]
    async fn test_cache_enabled_true_fails_without_backend() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                cache: ModelInferenceCacheConfig {
                    enabled: Some(true),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let result = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await;
        let err = result
            .err()
            .expect("Gateway should fail when cache.enabled=true but no backend available");
        assert!(
            err.to_string().contains("cache.enabled"),
            "error should mention cache.enabled: {err}"
        );
    }

    #[tokio::test]
    async fn test_cache_enabled_false_starts_without_backend() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                cache: ModelInferenceCacheConfig {
                    enabled: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let _gateway = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await
        .expect("Gateway should start when cache is explicitly disabled");
    }

    #[tokio::test]
    async fn test_cache_default_starts_without_backend() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                cache: ModelInferenceCacheConfig {
                    enabled: None,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let _gateway = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await
        .expect("Gateway should start when cache.enabled is default (null)");
    }

    #[tokio::test]
    #[expect(deprecated)]
    async fn test_setup_postgres_deprecated_enabled_true_emits_warning() {
        let logs_contain = crate::utils::testing::capture_logs();

        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(true),
                ..Default::default()
            },
            ..Default::default()
        }));

        // Will fail because the URL is bad, but the deprecation warning should still fire
        let _ = setup_postgres(config, Some("bad_url")).await;
        assert!(
            logs_contain("`postgres.enabled` is deprecated"),
            "should emit deprecation warning when postgres.enabled is explicitly set to true"
        );
    }

    #[tokio::test]
    #[expect(deprecated)]
    async fn test_setup_postgres_deprecated_enabled_false_emits_warning() {
        let logs_contain = crate::utils::testing::capture_logs();

        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: Some(false),
                ..Default::default()
            },
            ..Default::default()
        }));

        let postgres_connection_info = setup_postgres(config, None).await.unwrap();
        assert!(
            matches!(postgres_connection_info, PostgresConnectionInfo::Disabled),
            "postgres should be disabled when enabled=false"
        );
        assert!(
            logs_contain("`postgres.enabled` is deprecated"),
            "should emit deprecation warning when postgres.enabled is explicitly set to false"
        );
    }

    #[tokio::test]
    #[expect(deprecated)]
    async fn test_setup_postgres_default_no_deprecation_warning() {
        let logs_contain = crate::utils::testing::capture_logs();

        let config = Box::leak(Box::new(Config {
            postgres: PostgresConfig {
                enabled: None,
                ..Default::default()
            },
            ..Default::default()
        }));

        let _ = setup_postgres(config, None).await.unwrap();
        assert!(
            !logs_contain("`postgres.enabled` is deprecated"),
            "should not emit deprecation warning when postgres.enabled is not set"
        );
    }

    #[tokio::test]
    async fn test_auth_enabled_fails_without_postgres() {
        use crate::config::gateway::AuthConfig;

        let config = Arc::new(Config {
            gateway: GatewayConfig {
                auth: AuthConfig {
                    enabled: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let result = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await;
        let err = result
            .err()
            .expect("Gateway should fail when auth is enabled but Postgres is unavailable");
        assert!(
            err.to_string().contains("Authentication is enabled")
                && err.to_string().contains("Postgres"),
            "error should mention auth and Postgres: {err}"
        );
    }

    #[tokio::test]
    async fn test_auth_disabled_starts_without_postgres() {
        let config = Arc::new(Config {
            gateway: GatewayConfig {
                auth: Default::default(), // auth.enabled = false
                ..Default::default()
            },
            ..Default::default()
        });
        let http_client = TensorzeroHttpClient::new_testing().unwrap();
        let _gateway = GatewayHandle::new_with_database_and_http_client(
            config,
            Arc::new(RuntimeOverlay::default()),
            ClickHouseConnectionInfo::new_disabled(),
            PostgresConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            ValkeyConnectionInfo::Disabled,
            http_client,
            None,
            HashSet::new(),
            HashSet::new(),
            false,
        )
        .await
        .expect("Gateway should start when auth is disabled even without Postgres");
    }
}
