use std::future::IntoFuture;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use axum::extract::{rejection::JsonRejection, FromRequest, Json, Request};
use axum::routing::post;
use axum::Router;
use reqwest::{Client, Proxy};
use serde::de::DeserializeOwned;
use tokio::runtime::Handle;
use tokio::sync::oneshot::Sender;
use tokio_util::sync::CancellationToken;
use tracing::instrument;

use crate::clickhouse::migration_manager::{self, RunMigrationManagerArgs};
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::endpoints;
use crate::error::{Error, ErrorDetails};
use crate::howdy::setup_howdy;

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
        tracing::info!("Shutting down gateway");
        self.cancel_token.cancel();
        let handle = self
            .app_state
            .clickhouse_connection_info
            .batcher_join_handle();
        // Drop our `ClickHouseConnectionInfo`, so that we stop holding on to the `Arc<BatchSender>`
        // This allows the batch writer task to exit (once all of the remaining `ClickhouseConnectionInfo`s are dropped)
        self.app_state.clickhouse_connection_info = ClickHouseConnectionInfo::Disabled;
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
    }
}

/// State for the API
#[derive(Clone)]
// `#[non_exhaustive]` only affects downstream crates, so we can't use it here
#[expect(clippy::manual_non_exhaustive)]
pub struct AppStateData {
    pub config: Arc<Config>,
    pub http_client: Client,
    pub clickhouse_connection_info: ClickHouseConnectionInfo,
    // Prevent `AppStateData` from being directly constructed outside of this module
    // This ensures that `AppStateData` is only ever constructed via explicit `new` methods,
    // which can ensure that we update global state.
    _private: (),
}
pub type AppState = axum::extract::State<AppStateData>;

impl GatewayHandle {
    pub async fn new(config: Arc<Config>) -> Result<Self, Error> {
        let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL")
            .ok()
            .or_else(|| {
                std::env::var("CLICKHOUSE_URL").ok().inspect(|_| {
                    tracing::warn!("Deprecation Warning: The environment variable \"CLICKHOUSE_URL\" has been renamed to \"TENSORZERO_CLICKHOUSE_URL\" and will be removed in a future version. Please update your environment to use \"TENSORZERO_CLICKHOUSE_URL\" instead.");
                })
            });
        Self::new_with_clickhouse(config, clickhouse_url).await
    }

    async fn new_with_clickhouse(
        config: Arc<Config>,
        clickhouse_url: Option<String>,
    ) -> Result<Self, Error> {
        let clickhouse_connection_info = setup_clickhouse(&config, clickhouse_url, false).await?;
        let http_client = setup_http_client()?;
        Ok(Self::new_with_clickhouse_and_http_client(
            config,
            clickhouse_connection_info,
            http_client,
        ))
    }

    #[cfg(test)]
    pub fn new_unit_test_data(config: Arc<Config>, clickhouse_healthy: bool) -> Self {
        let http_client = reqwest::Client::new();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_mock(clickhouse_healthy);
        Self::new_with_clickhouse_and_http_client(config, clickhouse_connection_info, http_client)
    }

    pub fn new_with_clickhouse_and_http_client(
        config: Arc<Config>,
        clickhouse_connection_info: ClickHouseConnectionInfo,
        http_client: Client,
    ) -> Self {
        let cancel_token = CancellationToken::new();
        setup_howdy(clickhouse_connection_info.clone(), cancel_token.clone());
        Self {
            app_state: AppStateData {
                config,
                http_client,
                clickhouse_connection_info,
                _private: (),
            },
            cancel_token,
            _private: (),
        }
    }
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
    if let ClickHouseConnectionInfo::Production { .. } = &clickhouse_connection_info {
        migration_manager::run(RunMigrationManagerArgs {
            clickhouse: &clickhouse_connection_info,
            skip_completed_migrations: config.gateway.observability.skip_completed_migrations,
            manual_run: false,
        })
        .await?;
    }
    Ok(clickhouse_connection_info)
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

// This is set high enough that it should never be hit for a normal model response.
// In the future, we may want to allow overriding this at the model provider level.
const DEFAULT_HTTP_CLIENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5 * 60);

pub fn setup_http_client() -> Result<Client, Error> {
    let mut http_client_builder = Client::builder().timeout(DEFAULT_HTTP_CLIENT_TIMEOUT);

    if cfg!(feature = "e2e_tests") {
        if let Ok(proxy_url) = std::env::var("TENSORZERO_E2E_PROXY") {
            tracing::info!("Using proxy URL from TENSORZERO_E2E_PROXY: {proxy_url}");
            http_client_builder = http_client_builder
                .proxy(Proxy::all(proxy_url).map_err(|e| {
                    Error::new(ErrorDetails::AppState {
                        message: format!("Invalid proxy URL: {e}"),
                    })
                })?)
                // When running e2e tests, we use `provider-proxy` as an MITM proxy
                // for caching, so we need to accept the invalid (self-signed) cert.
                .danger_accept_invalid_certs(true);
        }
    }

    http_client_builder.build().map_err(|e| {
        Error::new(ErrorDetails::AppState {
            message: format!("Failed to build HTTP client: {e}"),
        })
    })
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
        Arc::new(Config::load_and_verify_from_path(Path::new(&config_file)).await?)
    } else {
        Arc::new(Config::default())
    };
    let gateway_handle = GatewayHandle::new_with_clickhouse(config, clickhouse_url).await?;

    let router = Router::new()
        .route(
            "/openai/v1/chat/completions",
            post(endpoints::openai_compatible::inference_handler),
        )
        .fallback(endpoints::fallback::handle_404)
        .with_state(gateway_handle.app_state.clone());

    let (sender, recv) = tokio::sync::oneshot::channel::<()>();
    let shutdown_fut = async move {
        let _ = recv.await;
    };

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
mod tests {
    use tracing_test::traced_test;

    use super::*;
    use crate::config_parser::{gateway::GatewayConfig, ObservabilityConfig};

    #[tokio::test]
    #[traced_test]
    async fn test_setup_clickhouse() {
        // Disabled observability
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(false),
                async_writes: false,
                batch_writes: Default::default(),
                skip_completed_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
        };

        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));

        let clickhouse_connection_info = setup_clickhouse(config, None, false).await.unwrap();
        assert!(matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ));
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));

        // Default observability and no ClickHouse URL
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: None,
                async_writes: false,
                batch_writes: Default::default(),
                skip_completed_migrations: false,
            },
            unstable_error_json: false,
            ..Default::default()
        };
        let config = Box::leak(Box::new(Config {
            gateway: gateway_config,
            ..Default::default()
        }));
        let clickhouse_connection_info = setup_clickhouse(config, None, false).await.unwrap();
        assert!(matches!(
            clickhouse_connection_info,
            ClickHouseConnectionInfo::Disabled
        ));
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
                skip_completed_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
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
                skip_completed_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
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
    #[traced_test]
    async fn test_unhealthy_clickhouse() {
        // Sensible URL that doesn't point to ClickHouse
        let gateway_config = GatewayConfig {
            observability: ObservabilityConfig {
                enabled: Some(true),
                async_writes: false,
                batch_writes: Default::default(),
                skip_completed_migrations: false,
            },
            bind_address: None,
            debug: false,
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: None,
            unstable_error_json: false,
            unstable_disable_feedback_target_validation: false,
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
