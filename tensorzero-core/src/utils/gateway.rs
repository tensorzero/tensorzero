use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use crate::db::postgres::PostgresConnectionInfo;
use crate::endpoints::openai_compatible::RouterExt;
use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, FromRequest, Json, Request};
use axum::Router;
use moka::sync::Cache;
use serde::de::DeserializeOwned;
use sqlx::postgres::PgPoolOptions;
use tensorzero_auth::postgres::AuthResult;
use tokio::runtime::Handle;
use tokio::sync::oneshot::Sender;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::instrument;

use crate::config::{Config, ConfigFileGlob};
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::migration_manager::{self, RunMigrationManagerArgs};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::feedback::FeedbackQueries;
use crate::endpoints;
use crate::error::{Error, ErrorDetails};
use crate::howdy::setup_howdy;
use crate::http::TensorzeroHttpClient;

#[cfg(test)]
use crate::db::clickhouse::ClickHouseClient;

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
///
// Using `#[non_exhaustive]` has no effect within the crate
#[expect(clippy::manual_non_exhaustive)]
pub struct GatewayHandle {
    pub app_state: AppStateData,
    pub cancel_token: CancellationToken,
    _private: (),
}

impl Drop for GatewayHandle {
    fn drop(&mut self) {
        self.cancel_token.cancel();
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
    /// Holds any background tasks that we want to wait on during shutdown
    /// We wait for these tasks to finish when `GatewayHandle` is dropped
    pub deferred_tasks: TaskTracker,
    /// Optional cache for TensorZero API key authentication
    pub auth_cache: Option<Cache<String, AuthResult>>,
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
    pub async fn new(config: Arc<Config>) -> Result<Self, Error> {
        let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL").ok();
        let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").ok();
        Self::new_with_databases(config, clickhouse_url, postgres_url).await
    }

    async fn new_with_databases(
        config: Arc<Config>,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
    ) -> Result<Self, Error> {
        let clickhouse_connection_info = setup_clickhouse(&config, clickhouse_url, false).await?;
        let postgres_connection_info = setup_postgres(&config, postgres_url).await?;
        let http_client = TensorzeroHttpClient::new(config.gateway.global_outbound_http_timeout)?;
        Self::new_with_database_and_http_client(
            config,
            clickhouse_connection_info,
            postgres_connection_info,
            http_client,
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
        Self {
            app_state: AppStateData {
                config,
                http_client,
                clickhouse_connection_info,
                postgres_connection_info,
                deferred_tasks: TaskTracker::new(),
                auth_cache,
                _private: (),
            },
            cancel_token,
            _private: (),
        }
    }

    #[cfg(feature = "pyo3")]
    pub fn new_dummy(http_client: TensorzeroHttpClient) -> Self {
        let config = Arc::new(Config::new_dummy_for_pyo3());
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_fake();
        #[cfg(test)]
        let postgres_connection_info = PostgresConnectionInfo::new_mock(true);
        #[cfg(not(test))]
        let postgres_connection_info = PostgresConnectionInfo::new_disabled();
        let cancel_token = CancellationToken::new();
        let auth_cache = create_auth_cache_from_config(&config);
        Self {
            app_state: AppStateData {
                config,
                http_client,
                clickhouse_connection_info,
                postgres_connection_info,
                deferred_tasks: TaskTracker::new(),
                auth_cache,
                _private: (),
            },
            cancel_token,
            _private: (),
        }
    }

    pub async fn new_with_database_and_http_client(
        config: Arc<Config>,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        postgres_connection_info: PostgresConnectionInfo,
        http_client: TensorzeroHttpClient,
    ) -> Result<Self, Error> {
        let cancel_token = CancellationToken::new();
        setup_howdy(
            &config,
            clickhouse_connection_info.clone(),
            cancel_token.clone(),
        );
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
        Ok(Self {
            app_state: AppStateData {
                config,
                http_client,
                clickhouse_connection_info,
                postgres_connection_info,
                deferred_tasks: TaskTracker::new(),
                auth_cache,
                _private: (),
            },
            cancel_token,
            _private: (),
        })
    }
}

pub async fn setup_clickhouse_without_config(
    clickhouse_url: String,
) -> Result<ClickHouseConnectionInfo, Error> {
    setup_clickhouse(&Config::new_empty().await?, Some(clickhouse_url), true).await
}

pub async fn setup_clickhouse(
    config: &Config,
    clickhouse_url: Option<String>,
    embedded_client: bool,
) -> Result<ClickHouseConnectionInfo, Error> {
    let clickhouse_connection_info = match (config.gateway.observability.enabled, clickhouse_url) {
        // Observability disabled by config
        (Some(false), _) => {
            tracing::info!("Disabling observability: `gateway.observability.enabled` is set to false in config.");
            ClickHouseConnectionInfo::new_disabled()
        }
        // Observability enabled but no ClickHouse URL
        (Some(true), None) => {
            return Err(ErrorDetails::AppState {
                message: "Missing environment variable TENSORZERO_CLICKHOUSE_URL".to_string(),
            }
            .into())
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
            tracing::warn!("Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and {msg_suffix}");
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

pub async fn setup_postgres(
    config: &Config,
    postgres_url: Option<String>,
) -> Result<PostgresConnectionInfo, Error> {
    let Some(postgres_url) = postgres_url else {
        // Check if rate limiting is configured but Postgres is not available
        if config.rate_limiting.enabled() && !config.rate_limiting.rules().is_empty() {
            return Err(Error::new(ErrorDetails::Config {
                message: "Rate limiting is configured but PostgreSQL is not available. Rate limiting requires PostgreSQL to be configured. Please set the TENSORZERO_POSTGRES_URL environment variable or disable rate limiting.".to_string(),
            }));
        }
        return Ok(PostgresConnectionInfo::Disabled);
    };

    // TODO - decide how we should handle apply `connection_pool_size` to two pools
    // Hopefully, sqlx does a stable release before we actually start using `alpha_pool`
    let pool = PgPoolOptions::new()
        .max_connections(config.postgres.connection_pool_size)
        .connect(&postgres_url)
        .await
        .map_err(|err| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: err.to_string(),
            })
        })?;

    let alpha_pool = sqlx_alpha::postgres::PgPoolOptions::new()
        .max_connections(config.postgres.connection_pool_size)
        .connect(&postgres_url)
        .await
        .map_err(|err| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: err.to_string(),
            })
        })?;

    let connection_info = PostgresConnectionInfo::new_with_pool(pool, Some(alpha_pool));
    connection_info.check_migrations().await?;
    Ok(connection_info)
}

/// Custom Axum extractor that validates the JSON body and deserializes it into a custom type
///
/// When this extractor is present, we don't check if the `Content-Type` header is `application/json`,
/// and instead simply assume that the request body is a JSON object.
pub struct StructuredJson<T>(pub T);

impl<S, T> FromRequest<S> for StructuredJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
    T: Send + Sync + DeserializeOwned,
{
    type Rejection = Error;

    #[instrument(skip_all, level = "trace", name = "StructuredJson::from_request")]
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
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

        Ok(StructuredJson(deserialized))
    }
}

// We hold on to these fields so that their Drop impls run when `ShutdownHandle` is dropped
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

    let config = if let Some(config_file) = config_file {
        Arc::new(Config::load_and_verify_from_path(&ConfigFileGlob::new(config_file)?).await?)
    } else {
        Arc::new(Config::new_empty().await?)
    };
    let gateway_handle =
        GatewayHandle::new_with_databases(config, clickhouse_url, postgres_url).await?;

    let router = Router::new()
        .register_openai_compatible_routes()
        .fallback(endpoints::fallback::handle_404)
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // increase the default body limit from 2MB to 100MB
        .layer(axum::middleware::from_fn(
            crate::observability::warn_early_drop::warn_on_early_connection_drop,
        ))
        .with_state(gateway_handle.app_state.clone());

    let (sender, recv) = tokio::sync::oneshot::channel::<()>();
    let shutdown_fut = async move {
        let _ = recv.await;
    };

    // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(
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
    use crate::config::{gateway::GatewayConfig, ObservabilityConfig};

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
        };

        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));

        let clickhouse_connection_info = setup_clickhouse(config, None, false).await.unwrap();
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
        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));
        let clickhouse_connection_info = setup_clickhouse(config, None, false).await.unwrap();
        assert_eq!(
            clickhouse_connection_info.client_type(),
            ClickHouseClientType::Disabled
        );
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));
        assert!(logs_contain("Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and `TENSORZERO_CLICKHOUSE_URL` is not set."));

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
        };

        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));

        let err = setup_clickhouse(config, None, false).await.unwrap_err();
        assert!(err
            .to_string()
            .contains("Missing environment variable TENSORZERO_CLICKHOUSE_URL"));

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
        };
        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));
        setup_clickhouse(config, Some("bad_url".to_string()), false)
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
        };
        let config = Config {
            gateway: gateway_config,
            ..Default::default()
        };
        setup_clickhouse(
            &config,
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
}
