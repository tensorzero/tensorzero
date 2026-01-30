use std::collections::HashSet;
use std::{env, fmt::Display, future::Future, path::PathBuf, sync::Arc, time::Duration};

use crate::config::ConfigFileGlob;
use crate::config::RuntimeOverlay;
use crate::config::snapshot::ConfigSnapshot;
use crate::config::unwritten::UnwrittenConfig;
use crate::endpoints::openai_compatible::types::embeddings::OpenAICompatibleEmbeddingParams;
use crate::endpoints::openai_compatible::types::embeddings::OpenAIEmbeddingResponse;
use crate::feature_flags;
use crate::http::TensorzeroResponseWrapper;
use crate::http::{DEFAULT_HTTP_CLIENT_TIMEOUT, TensorzeroHttpClient, TensorzeroRequestBuilder};
use crate::inference::types::stored_input::StoragePathResolver;
use crate::observability::{
    TENSORZERO_OTLP_ATTRIBUTE_PREFIX, TENSORZERO_OTLP_HEADERS_PREFIX,
    TENSORZERO_OTLP_RESOURCE_PREFIX,
};
use crate::utils::gateway::DropWrapper;
use crate::{
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    db::postgres::PostgresConnectionInfo,
    db::valkey::ValkeyConnectionInfo,
    error::{Error, ErrorDetails},
    utils::gateway::{GatewayHandle, setup_clickhouse, setup_postgres, setup_valkey},
};
use reqwest::header::HeaderMap;
use reqwest_sse_stream::Event;
use secrecy::{ExposeSecret, SecretString};
use std::fmt::Debug;
use tokio::time::error::Elapsed;
use tokio_stream::StreamExt;
use url::Url;

pub use client_inference_params::{ClientInferenceParams, ClientSecretString};
pub use input_handling::resolved_input_to_client_input;

pub use crate::cache::CacheParamsOptions;
pub use crate::endpoints::feedback::FeedbackResponse;
pub use crate::endpoints::feedback::Params as FeedbackParams;
pub use crate::endpoints::inference::{
    InferenceOutput, InferenceParams, InferenceResponse, InferenceResponseChunk, InferenceStream,
};
pub use crate::endpoints::object_storage::ObjectResponse;
pub use crate::inference::types::storage::{StorageKind, StoragePath};
pub use crate::inference::types::{Base64File, File, ObjectStoragePointer, UrlFile};
pub use crate::inference::types::{
    ContentBlockChunk, Input, InputMessage, InputMessageContent, Role, System, Unknown,
};
pub use crate::tool::{DynamicToolParams, Tool};

pub mod client_inference_params;
pub mod input_handling;

pub enum ClientMode {
    HTTPGateway(HTTPGateway),
    EmbeddedGateway {
        gateway: EmbeddedGateway,
        timeout: Option<Duration>,
    },
}

impl Debug for ClientMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientMode::HTTPGateway(_) => write!(f, "HTTPGateway"),
            ClientMode::EmbeddedGateway {
                gateway: _,
                timeout,
            } => {
                write!(f, "EmbeddedGateway {{ timeout: {timeout:?} }}")
            }
        }
    }
}

pub struct HttpResponse<T> {
    pub response: T,
    pub raw_request: String,
    pub raw_response: Option<String>,
}

pub struct HTTPGateway {
    pub base_url: Url,
    pub http_client: TensorzeroHttpClient,
    headers: HeaderMap,
    timeout: Option<Duration>,
    verbose_errors: bool,
}

impl HTTPGateway {
    pub async fn check_http_response(
        &self,
        resp: Result<TensorzeroResponseWrapper, reqwest::Error>,
    ) -> Result<TensorzeroResponseWrapper, TensorZeroError> {
        let resp = resp.map_err(|e| {
            if e.is_timeout() {
                TensorZeroError::RequestTimeout
            } else {
                TensorZeroError::Other {
                    source: Error::new(ErrorDetails::JsonRequest {
                        message: format!(
                            "Error from server: {}",
                            DisplayOrDebug {
                                val: e,
                                debug: self.verbose_errors,
                            }
                        ),
                    })
                    .into(),
                }
            }
        })?;

        if let Err(e) = resp.error_for_status_ref() {
            let status_code = resp.status().as_u16();
            let text = resp.text().await.ok();
            return Err(TensorZeroError::Http {
                status_code,
                text,
                source: Error::new(ErrorDetails::JsonRequest {
                    message: format!(
                        "Request failed: {}",
                        DisplayOrDebug {
                            val: e,
                            debug: self.verbose_errors,
                        }
                    ),
                })
                .into(),
            });
        }
        Ok(resp)
    }

    fn customize_builder<'a>(
        &self,
        mut builder: TensorzeroRequestBuilder<'a>,
    ) -> TensorzeroRequestBuilder<'a> {
        if let Some(timeout) = self.timeout {
            builder = builder.timeout(timeout);
        }
        builder.headers(self.headers.clone())
    }

    pub async fn send_request(
        &self,
        builder: TensorzeroRequestBuilder<'_>,
    ) -> Result<String, TensorZeroError> {
        let builder = self.customize_builder(builder);
        let resp = builder.send().await;
        self.check_http_response(resp)
            .await?
            .text()
            .await
            .map_err(|e| TensorZeroError::Other {
                source: Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error deserializing response: {}",
                        DisplayOrDebug {
                            val: e,
                            debug: self.verbose_errors,
                        }
                    ),
                })
                .into(),
            })
    }

    pub async fn send_and_parse_http_response<T: serde::de::DeserializeOwned>(
        &self,
        builder: TensorzeroRequestBuilder<'_>,
    ) -> Result<(T, String), TensorZeroError> {
        let builder = self.customize_builder(builder);
        let resp = self.check_http_response(builder.send().await).await?;
        let raw_response = resp.text().await.map_err(|e| TensorZeroError::Other {
            source: Error::new(ErrorDetails::Serialization {
                message: format!(
                    "Error deserializing response: {}",
                    DisplayOrDebug {
                        val: e,
                        debug: self.verbose_errors,
                    }
                ),
            })
            .into(),
        })?;

        let response: T =
            serde_json::from_str(&raw_response).map_err(|e| TensorZeroError::Other {
                source: Error::new(ErrorDetails::Serialization {
                    message: format!(
                        "Error deserializing response: {}",
                        DisplayOrDebug {
                            val: e,
                            debug: self.verbose_errors,
                        }
                    ),
                })
                .into(),
            })?;
        Ok((response, raw_response))
    }

    async fn send_http_stream_inference(
        &self,
        builder: TensorzeroRequestBuilder<'_>,
    ) -> Result<InferenceStream, TensorZeroError> {
        let event_source = match self.customize_builder(builder).eventsource().await {
            Ok(es) => es,
            Err(e) => {
                let err_str = format!("Error in streaming response: {e:?}");
                let inner_err = Error::new(ErrorDetails::StreamError {
                    source: Box::new(Error::new(ErrorDetails::Serialization { message: err_str })),
                });
                if let reqwest_sse_stream::ReqwestSseStreamError::InvalidStatusCode(code, resp) = e
                {
                    return Err(TensorZeroError::Http {
                        status_code: code.as_u16(),
                        text: resp.text().await.ok(),
                        source: inner_err.into(),
                    });
                }
                return Err(TensorZeroError::Other {
                    source: Error::new(ErrorDetails::StreamError {
                        source: Box::new(inner_err),
                    })
                    .into(),
                });
            }
        };

        let mut event_source = event_source.peekable();
        let first = event_source.peek().await;
        if let Some(Err(_)) = first {
            // Discard the stream if it has an error
            let res = event_source.next().await;
            #[expect(clippy::panic)]
            let Some(Err(e)) = res else {
                panic!("Peeked error but got non-err {res:?}");
            };
            let err_str = format!("Error in streaming response: {e:?}");
            let inner_err = Error::new(ErrorDetails::StreamError {
                source: Box::new(Error::new(ErrorDetails::Serialization { message: err_str })),
            });
            if let reqwest_sse_stream::ReqwestSseStreamError::InvalidStatusCode(code, resp) = *e {
                return Err(TensorZeroError::Http {
                    status_code: code.as_u16(),
                    text: resp.text().await.ok(),
                    source: inner_err.into(),
                });
            }
            return Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::StreamError {
                    source: Box::new(inner_err),
                })
                .into(),
            });
        }
        let verbose_errors = self.verbose_errors;
        Ok(Box::pin(async_stream::stream! {
            while let Some(ev) = event_source.next().await {
                match ev {
                    Err(e) => {
                        yield Err(Error::new(ErrorDetails::StreamError {
                            source: Box::new(Error::new(ErrorDetails::Serialization {
                                message: format!("Error in streaming response: {}", DisplayOrDebug {
                                    val: e,
                                    debug: verbose_errors,
                                })
                            }))
                        }))
                    }
                    Ok(e) => match e {
                        Event::Open => continue,
                        Event::Message(message) => {
                            if message.data == "[DONE]" {
                                break;
                            }
                            let json: serde_json::Value = serde_json::from_str(&message.data).map_err(|e| {
                                Error::new(ErrorDetails::Serialization {
                                    message: format!("Error deserializing inference response chunk: {}", DisplayOrDebug {
                                        val: e,
                                        debug: verbose_errors,
                                    }),
                                })
                            })?;
                            if let Some(err) = json.get("error") {
                                yield Err(Error::new(ErrorDetails::StreamError {
                                    source: Box::new(Error::new(ErrorDetails::Serialization {
                                        message: format!("Stream produced an error: {}", DisplayOrDebug {
                                            val: err,
                                            debug: verbose_errors,
                                        }),
                                    }))
                                }));
                            } else {
                                let data: InferenceResponseChunk =
                                serde_json::from_value(json).map_err(|e| {
                                    Error::new(ErrorDetails::Serialization {
                                        message: format!("Error deserializing json value as InferenceResponseChunk: {}", DisplayOrDebug {
                                            val: e,
                                            debug: verbose_errors,
                                        }),
                                    })
                                })?;
                                yield Ok(data);
                            }

                        }
                    }
                }
            }
        }))
    }
}

pub struct EmbeddedGateway {
    pub handle: GatewayHandle,
}

/// Used to construct a `Client`
/// Call `ClientBuilder::new` to create a new `ClientBuilder`
/// in either `HTTPGateway` or `EmbeddedGateway` mode
pub struct ClientBuilder {
    mode: ClientBuilderMode,
    http_client: Option<TensorzeroHttpClient>,
    verbose_errors: bool,
    api_key: Option<SecretString>,
    timeout: Option<Duration>,
    drop_wrapper: Option<DropWrapper>,
}

/// An error type representing an error from within the TensorZero gateway
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum TensorZeroError {
    Http {
        status_code: u16,
        text: Option<String>,
        #[source]
        source: TensorZeroInternalError,
    },
    Other {
        #[source]
        source: TensorZeroInternalError,
    },
    RequestTimeout,
    Git {
        #[source]
        source: git2::Error,
    },
}

impl Display for TensorZeroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorZeroError::Http {
                status_code,
                text,
                source: _,
            } => {
                if let Some(text) = text {
                    write!(f, "HTTP Error (status code {status_code}): {text}")
                } else {
                    write!(f, "HTTP Error (status code {status_code})")
                }
            }
            TensorZeroError::Other { source } => write!(f, "{source}"),
            TensorZeroError::RequestTimeout => write!(f, "HTTP Error: request timed out"),
            TensorZeroError::Git { source } => write!(f, "Failed to get git info: {source}"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Internal TensorZero Error: {0}")]
pub struct TensorZeroInternalError(#[from] Error);

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum ClientBuilderError {
    #[error(
        "Missing configuration: you must call `with_config_file` before calling `build` in EmbeddedGateway mode"
    )]
    MissingConfig,
    #[error(
        "Missing gateway URL: you must call `with_gateway_url` before calling `build` in HTTPGateway mode"
    )]
    MissingGatewayUrl,
    #[error("Called ClientBuilder.build_http() when not in HTTPGateway mode")]
    NotHTTPGateway,
    #[error("Failed to configure ClickHouse: {0}")]
    Clickhouse(TensorZeroError),
    #[error("Failed to configure PostgreSQL: {0}")]
    Postgres(TensorZeroError),
    #[error("Authentication is not supported in embedded gateway mode: {0}")]
    AuthNotSupportedInEmbeddedMode(TensorZeroError),
    #[error("Failed to parse config: {0}")]
    ConfigParsingPreGlob(TensorZeroError),
    #[error("Failed to parse config: {error}. Config file glob `{glob}` resolved to the following files:\n{paths}", glob = glob.glob,paths = glob.paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n"))]
    ConfigParsing {
        error: TensorZeroError,
        glob: ConfigFileGlob,
    },
    #[error("Failed to build HTTP client: {0}")]
    HTTPClientBuild(TensorZeroError),
    #[error("Failed to get gateway version: {0}")]
    GatewayVersion(String),
    #[error("Failed to set up embedded gateway: {0}")]
    EmbeddedGatewaySetup(TensorZeroError),
    #[error("Failed to initialize feature flags: {0}")]
    FeatureFlags(TensorZeroError),
}

// Helper type to choose between using Debug or Display for a type
#[doc(hidden)]
pub struct DisplayOrDebug<T: Debug + Display> {
    pub val: T,
    pub debug: bool,
}

impl<T: Debug + Display> Display for DisplayOrDebug<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.debug {
            write!(f, "{:?}", self.val)
        } else {
            write!(f, "{}", self.val)
        }
    }
}

pub enum PostgresConfig {
    /// Constructs a new Postgres pool from the given url
    Url(String),
    /// Re-uses an existing PostgresConnectionInfo
    ExistingConnectionInfo(PostgresConnectionInfo),
}

/// Controls how a `Client` is run
pub enum ClientBuilderMode {
    /// In HTTPGateway mode, we make HTTP requests to a TensorZero gateway server.
    HTTPGateway { url: Url },
    /// In EmbeddedGateway mode, we run an embedded gateway using a config file.
    /// We do not launch an HTTP server - we only make outgoing HTTP requests to model providers and to ClickHouse.
    EmbeddedGateway {
        config_file: Option<PathBuf>,
        clickhouse_url: Option<String>,
        postgres_config: Option<PostgresConfig>,
        valkey_url: Option<String>,
        /// A timeout for all TensorZero gateway processing.
        /// If this timeout is hit, any in-progress LLM requests may be aborted.
        timeout: Option<std::time::Duration>,
        verify_credentials: bool,
        /// Allow turning on batch writes - used in e2e tests.
        /// We don't expose this through the Python client, since we're having deadlock issues
        /// there.
        allow_batch_writes: bool,
    },
    /// Construct a client from already-initialized components.
    /// Used when the gateway infrastructure is already set up (e.g., in optimizers).
    /// This avoids re-parsing config files or re-initializing database connections.
    FromComponents {
        /// Pre-parsed TensorZero configuration
        config: Arc<Config>,
        /// Use the settings from this `ClickHouseConnectionInfo` to create a *new* ClickHouseConnectionInfo
        /// We do *not* re-use this directly,since we block when an embedded client `GatewayHandle` is dropped,
        /// waiting on all outstanding `ClickHouseConnectionInfo` to get dropped.
        /// This does not work if two different embedded clients can use the same `ClickHouseConnectionInfo`,
        /// since one might be in use by Python code with the GIL held
        clickhouse_connection_info: ClickHouseConnectionInfo,
        /// Already-initialized Postgres connection
        postgres_connection_info: PostgresConnectionInfo,
        /// Already-initialized Valkey connection
        valkey_connection_info: ValkeyConnectionInfo,
        /// Pre-configured HTTP client for model inference
        http_client: TensorzeroHttpClient,
        /// A timeout for all TensorZero gateway processing.
        /// If this timeout is hit, any in-progress LLM requests may be aborted.
        timeout: Option<Duration>,
    },
}

/// A `ClientBuilder` is used to construct a `Client`.
impl ClientBuilder {
    pub fn new(mode: ClientBuilderMode) -> Self {
        Self {
            mode,
            http_client: None,
            verbose_errors: false,
            api_key: None,
            timeout: None,
            drop_wrapper: None,
        }
    }

    /// Sets the `TensorzeroHttpClient` to be used when making any HTTP requests.
    /// In `EmbeddedGateway` mode, this is used for making requests to model endpoints,
    /// as well as ClickHouse.
    /// In `HTTPGateway` mode, this is used for making requests to the gateway.
    pub fn with_http_client(mut self, client: TensorzeroHttpClient) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Sets whether error messages should be more verbose (more `Debug` impls are used).
    /// This increases the chances of exposing sensitive information (e.g. model responses)
    /// in error messages.
    ///
    /// This is `false` by default.
    pub fn with_verbose_errors(mut self, verbose_errors: bool) -> Self {
        self.verbose_errors = verbose_errors;
        self
    }

    /// Sets the API key to use for authentication with the TensorZero gateway.
    /// This is only used in `HTTPGateway` mode.
    /// If not set, the client will attempt to read from the `TENSORZERO_API_KEY` environment variable.
    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the timeout for HTTP requests to the TensorZero gateway.
    /// This is only used in `HTTPGateway` mode.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the drop wrapper for embedded gateway mode.
    /// This is used internally by the Python client, since we need to enter the Tokio runtime
    /// and release the Python GIL (the drop may have been originally triggered from the Python
    /// interpreter performing garbage collection of the wrapping Python object)
    /// When the embedded gateway shuts down, the provided `DropWrapper` will be called
    /// with a closure representing the actual drop logic (e.g. waiting for the ClickHouse batch insert task to complete)
    #[cfg(feature = "pyo3")]
    pub fn with_drop_wrapper(mut self, drop_wrapper: DropWrapper) -> Self {
        self.drop_wrapper = Some(drop_wrapper);
        self
    }

    /// Constructs a `Client`, returning an error if the configuration is invalid.
    pub async fn build(self) -> Result<Client, ClientBuilderError> {
        // Initialize feature flags (for embedded clients).
        feature_flags::init_flags().map_err(|e| {
            ClientBuilderError::FeatureFlags(TensorZeroError::Other { source: e.into() })
        })?;

        match &self.mode {
            ClientBuilderMode::HTTPGateway { .. } => {
                let client = self.build_http()?;
                Ok(client)
            }
            ClientBuilderMode::EmbeddedGateway {
                config_file,
                clickhouse_url,
                postgres_config,
                valkey_url,
                timeout,
                verify_credentials,
                allow_batch_writes,
            } => Box::pin(async move {
                let unwritten_config = if let Some(config_file) = config_file {
                    let glob = ConfigFileGlob::new(config_file.to_string_lossy().to_string())
                        .map_err(|e| {
                            ClientBuilderError::ConfigParsingPreGlob(TensorZeroError::Other {
                                source: e.into(),
                            })
                        })?;
                    Box::pin(Config::load_from_path_optional_verify_credentials(
                        &glob,
                        *verify_credentials,
                    ))
                    .await
                    .map_err(|e| ClientBuilderError::ConfigParsing {
                        error: TensorZeroError::Other { source: e.into() },
                        glob,
                    })?
                } else {
                    tracing::info!(
                        "No config file provided, so only default functions will be available. Set `config_file` to specify your `tensorzero.toml`"
                    );
                    Config::new_empty()
                        .await
                        .map_err(|e| ClientBuilderError::ConfigParsing {
                            error: TensorZeroError::Other { source: e.into() },
                            glob: ConfigFileGlob::new_empty(),
                        })?
                };
                let clickhouse_connection_info =
                    setup_clickhouse(&unwritten_config, clickhouse_url.clone(), true)
                        .await
                        .map_err(|e| {
                            ClientBuilderError::Clickhouse(TensorZeroError::Other {
                                source: e.into(),
                            })
                        })?;
                let config = Box::pin(unwritten_config.into_config(&clickhouse_connection_info))
                    .await
                    .map_err(|e| {
                        ClientBuilderError::Clickhouse(TensorZeroError::Other { source: e.into() })
                    })?;
                let config = Arc::new(config);
                Self::validate_embedded_gateway_config(&config, *allow_batch_writes)?;
                let postgres_connection_info = match postgres_config {
                    Some(PostgresConfig::Url(url)) => {
                        setup_postgres(&config, Some(url.clone())).await.map_err(|e| {
                            ClientBuilderError::Postgres(TensorZeroError::Other { source: e.into() })
                        })?
                    }
                    Some(PostgresConfig::ExistingConnectionInfo(connection_info)) => connection_info.clone(),
                    None => setup_postgres(&config, None).await.map_err(|e| {
                        ClientBuilderError::Postgres(TensorZeroError::Other { source: e.into() })
                    })?
                };

                // Set up Valkey connection from explicit URL
                let valkey_connection_info = setup_valkey(valkey_url.as_deref()).await.map_err(|e| {
                    ClientBuilderError::EmbeddedGatewaySetup(TensorZeroError::Other {
                        source: e.into(),
                    })
                })?;

                let http_client = if self.http_client.is_some() {
                    return Err(ClientBuilderError::HTTPClientBuild(
                        TensorZeroError::Other {
                            source: TensorZeroInternalError(Error::new(ErrorDetails::AppState {
                                message: "HTTP client cannot be provided in EmbeddedGateway mode"
                                    .to_string(),
                            })),
                        },
                    ));
                } else {
                    config.http_client.clone()
                };
                Ok(Client {
                    mode: Arc::new(ClientMode::EmbeddedGateway {
                        gateway: EmbeddedGateway {
                            handle: GatewayHandle::new_with_database_and_http_client(
                                config,
                                clickhouse_connection_info,
                                postgres_connection_info,
                                valkey_connection_info,
                                http_client,
                                self.drop_wrapper,
                                HashSet::new(), // available_tools not needed for embedded client
                            )
                            .await
                            .map_err(|e| {
                                ClientBuilderError::EmbeddedGatewaySetup(TensorZeroError::Other {
                                    source: e.into(),
                                })
                            })?,
                        },
                        timeout: *timeout,
                    }),
                    verbose_errors: self.verbose_errors,
                })
            }).await,
            ClientBuilderMode::FromComponents {
                config,
                clickhouse_connection_info,
                postgres_connection_info,
                valkey_connection_info,
                http_client,
                timeout,
            } => {
                // Validate embedded gateway configuration
                // Note: FromComponents doesn't have an allow_batch_writes parameter,
                // so we pass false to enforce that batch writes must be explicitly forced
                // via __force_allow_embedded_batch_writes if enabled.
                Self::validate_embedded_gateway_config(config, false)?;

                // Construct Client directly from provided components
                Ok(Client {
                    mode: Arc::new(ClientMode::EmbeddedGateway {
                        gateway: EmbeddedGateway {
                            handle: GatewayHandle::new_with_database_and_http_client(
                                config.clone(),
                                // We create a new independent `ClickHouseConnectionInfo` here,
                                // and do *not* directly use the existing `clickhouse_connection_info`
                                // See `ClientBuilderMode::FromComponents` for more details
                                clickhouse_connection_info.recreate().await.map_err(|e| {
                                    ClientBuilderError::Clickhouse(TensorZeroError::Other {
                                        source: e.into(),
                                    })
                                })?,
                                postgres_connection_info.clone(),
                                valkey_connection_info.clone(),
                                http_client.clone(),
                                self.drop_wrapper,
                                HashSet::new(), // available_tools not needed for embedded client
                            )
                            .await
                            .map_err(|e| {
                                ClientBuilderError::EmbeddedGatewaySetup(TensorZeroError::Other {
                                    source: e.into(),
                                })
                            })?,
                        },
                        timeout: *timeout,
                    }),
                    verbose_errors: self.verbose_errors,
                })
            }
        }
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub fn build_from_state(handle: GatewayHandle) -> Result<Client, ClientBuilderError> {
        Ok(Client {
            mode: Arc::new(ClientMode::EmbeddedGateway {
                gateway: EmbeddedGateway { handle },
                timeout: None,
            }),
            verbose_errors: false,
        })
    }

    /// Creates a client from a historical ConfigSnapshot.
    ///
    /// This allows replaying inferences with the exact configuration that was used
    /// at the time the snapshot was created. The semantic configuration (functions,
    /// models, variants, templates, etc.) comes from the snapshot, while runtime
    /// infrastructure settings are overlaid from the live config.
    ///
    /// # Parameters
    /// - `snapshot`: The ConfigSnapshot to load from (historical semantic config)
    /// - `live_config`: Reference to the current live gateway config. Runtime fields
    ///   (`gateway`, `object_store_info`, `postgres`, `rate_limiting`, `http_client`)
    ///   are copied from this config to override the snapshot's values, since these
    ///   represent current infrastructure rather than historical behavior.
    /// - `clickhouse_url`: Current ClickHouse connection (not from snapshot)
    /// - `postgres_url`: Current Postgres connection (not from snapshot)
    /// - `verify_credentials`: Whether to validate model provider credentials
    /// - `timeout`: Optional timeout for gateway operations
    ///
    /// # Returns
    /// A Client configured with historical semantic settings but current runtime parameters
    pub async fn from_config_snapshot(
        snapshot: ConfigSnapshot,
        live_config: &Config,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
        valkey_url: Option<String>,
        verify_credentials: bool,
        timeout: Option<Duration>,
    ) -> Result<Client, ClientBuilderError> {
        // Create runtime overlay from live config.
        // This ensures infrastructure settings (gateway, postgres, rate limiting, etc.)
        // reflect the current environment rather than historical snapshot values.
        let runtime_overlay = RuntimeOverlay::from_config(live_config);

        // Load config from snapshot with runtime overlay applied
        let unwritten_config = Box::pin(Config::load_from_snapshot(
            snapshot,
            runtime_overlay,
            verify_credentials,
        ))
        .await
        .map_err(|e| ClientBuilderError::ConfigParsing {
            error: TensorZeroError::Other { source: e.into() },
            glob: ConfigFileGlob::new_empty(),
        })?;

        // Setup ClickHouse with runtime URL
        let clickhouse_connection_info = setup_clickhouse(&unwritten_config, clickhouse_url, true)
            .await
            .map_err(|e| {
                ClientBuilderError::Clickhouse(TensorZeroError::Other { source: e.into() })
            })?;

        // Convert config_load_info into Config with hash
        let config = Box::pin(unwritten_config.into_config(&clickhouse_connection_info))
            .await
            .map_err(|e| {
                ClientBuilderError::Clickhouse(TensorZeroError::Other { source: e.into() })
            })?;

        let config = Arc::new(config);

        // Validate embedded gateway configuration
        Self::validate_embedded_gateway_config(&config, false)?;

        // Setup Postgres with runtime URL
        let postgres_connection_info =
            setup_postgres(&config, postgres_url).await.map_err(|e| {
                ClientBuilderError::Postgres(TensorZeroError::Other { source: e.into() })
            })?;

        // Setup Valkey with runtime URL
        let valkey_connection_info = setup_valkey(valkey_url.as_deref()).await.map_err(|e| {
            ClientBuilderError::EmbeddedGatewaySetup(TensorZeroError::Other { source: e.into() })
        })?;

        // Use HTTP client from config (now overlaid from live_config)
        let http_client = config.http_client.clone();

        // Build client using FromComponents pattern
        let builder = ClientBuilder::new(ClientBuilderMode::FromComponents {
            config,
            clickhouse_connection_info,
            postgres_connection_info,
            valkey_connection_info,
            http_client,
            timeout,
        });

        Box::pin(builder.build()).await
    }

    /// Validates configuration for embedded gateway mode.
    /// Checks for unsupported features like batch writes and authentication.
    fn validate_embedded_gateway_config(
        config: &Config,
        allow_batch_writes: bool,
    ) -> Result<(), ClientBuilderError> {
        // Validate batch writes configuration
        if !allow_batch_writes
            && config.gateway.observability.batch_writes.enabled
            && !config
                .gateway
                .observability
                .batch_writes
                .__force_allow_embedded_batch_writes
        {
            return Err(ClientBuilderError::Clickhouse(TensorZeroError::Other {
                source: Error::new(ErrorDetails::Config {
                    message: "`[gateway.observability.batch_writes]` is not yet supported in embedded gateway mode".to_string(),
                })
                .into(),
            }));
        }

        // Validate auth configuration
        if config.gateway.auth.enabled {
            return Err(ClientBuilderError::AuthNotSupportedInEmbeddedMode(
                TensorZeroError::Other {
                    source: Error::new(ErrorDetails::Config {
                        message: "`[gateway.auth]` is not supported in embedded gateway mode. Authentication is only available when using HTTP gateway mode. Please either disable authentication by setting `gateway.auth.enabled = false` or use HTTP mode instead.".to_string(),
                    })
                    .into(),
                },
            ));
        }

        Ok(())
    }

    /// Builds a `Client` in HTTPGateway mode, erroring if the mode is not HTTPGateway
    /// This allows avoiding calling the async `build` method
    pub fn build_http(self) -> Result<Client, ClientBuilderError> {
        let ClientBuilderMode::HTTPGateway { mut url } = self.mode else {
            return Err(ClientBuilderError::NotHTTPGateway);
        };
        // This is only used when dropping a `GatewayHandle` in embedded gateway mode,
        // and does not currently make sense in HTTPGateway mode.
        if self.drop_wrapper.is_some() {
            return Err(ClientBuilderError::HTTPClientBuild(
                TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InternalError {
                        message:
                            "ClientBuilder.with_drop_wrapper is not allowed in HTTPGateway mode"
                                .to_string(),
                    })
                    .into(),
                },
            ));
        }
        // Enforce that the URL has a trailing slash, so that joining endpoints works correctly
        // This means that passing in a url that looks like 'http://example.com/some/prefix'
        // will result in inference requests being sent to 'http://example.com/some/prefix/inference'
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }

        // Try to get API key from constructor parameter, otherwise try environment variable
        let api_key = self
            .api_key
            .or_else(|| env::var("TENSORZERO_API_KEY").ok().map(|s| s.into()));

        // Build the HTTP client, applying timeout and/or API key
        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            TensorzeroHttpClient::new(
                // The timeout may be overridden in `send_and_parse_http_response`
                DEFAULT_HTTP_CLIENT_TIMEOUT,
            )
            .map_err(|e| {
                ClientBuilderError::HTTPClientBuild(TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to build HTTP client: {e}"),
                    })
                    .into(),
                })
            })?
        };

        let mut headers = HeaderMap::new();
        if let Some(key) = api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", key.expose_secret()))
                    .map_err(|e| {
                        ClientBuilderError::HTTPClientBuild(TensorZeroError::Other {
                            source: Error::new(ErrorDetails::InternalError {
                                message: format!("Failed to create authorization header: {e}"),
                            })
                            .into(),
                        })
                    })?,
            );
        }
        Ok(Client {
            mode: Arc::new(ClientMode::HTTPGateway(HTTPGateway {
                base_url: url,
                http_client,
                headers,
                timeout: self.timeout,
                verbose_errors: self.verbose_errors,
            })),
            verbose_errors: self.verbose_errors,
        })
    }
}

/// A TensorZero client. This is constructed using `ClientBuilder`
#[derive(Debug, Clone)]
pub struct Client {
    mode: Arc<ClientMode>,
    pub verbose_errors: bool,
}

impl StoragePathResolver for Client {
    async fn resolve(&self, storage_path: StoragePath) -> Result<String, Error> {
        Ok(self
            .get_object(storage_path.clone())
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Error resolving object {storage_path}: {e}"),
                })
            })?
            .data)
    }
}

impl Client {
    /// Returns a reference to the client mode for use by extension traits
    pub fn mode(&self) -> &ClientMode {
        &self.mode
    }

    /// Assigns feedback for a TensorZero inference.
    /// See https://www.tensorzero.com/docs/gateway/api-reference#post-feedback
    pub async fn feedback(
        &self,
        params: FeedbackParams,
    ) -> Result<FeedbackResponse, TensorZeroError> {
        match &*self.mode {
            ClientMode::HTTPGateway(client) => {
                let url = client
                    .base_url
                    .join("feedback")
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /feedback endpoint: {e}"
                            ),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.post(url).json(&params);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                // We currently ban auth-enabled configs in embedded gateway mode,
                // so we don't have an API key here
                Ok(with_embedded_timeout(*timeout, async {
                    crate::endpoints::feedback::feedback(
                        gateway.handle.app_state.clone(),
                        params,
                        None,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
        }
    }

    pub async fn http_embeddings(
        &self,
        params: OpenAICompatibleEmbeddingParams,
        api_key: Option<SecretString>,
    ) -> Result<HttpResponse<OpenAIEmbeddingResponse>, TensorZeroError> {
        match &*self.mode {
            ClientMode::HTTPGateway(client) => {
                let url = client.base_url.join("/openai/v1/embeddings").map_err(|e| {
                    TensorZeroError::Other {
                        source: Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /openai/v1/embeddings endpoint: {e}"
                            ),
                        })
                        .into(),
                    }
                })?;
                let body = serde_json::to_string(&params).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Failed to serialize embedding params: {}",
                            DisplayOrDebug {
                                val: e,
                                debug: self.verbose_errors,
                            }
                        ),
                    })
                    .into(),
                })?;
                let mut builder = client
                    .http_client
                    .post(url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body.clone());

                if let Some(api_key) = api_key {
                    builder = builder.header(
                        reqwest::header::AUTHORIZATION,
                        format!("Bearer {}", api_key.expose_secret()),
                    );
                }
                let (response, raw_response) = client.send_and_parse_http_response(builder).await?;
                Ok(HttpResponse {
                    response,
                    raw_request: body,
                    raw_response: Some(raw_response),
                })
            }
            ClientMode::EmbeddedGateway { .. } => Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InternalError {
                    message: "HTTP embeddings is not supported in embedded gateway mode"
                        .to_string(),
                })
                .into(),
            }),
        }
    }

    /// Runs a TensorZero inference over HTTP
    /// This is like `inference`, but only works in HTTPGateway mode
    /// The `HttpResponse` struct contains extra http-specific information (e.g. raw_request and raw_response),
    /// which would not be available when calling `inference` on an embedded gateway.
    pub async fn http_inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<HttpResponse<InferenceOutput>, TensorZeroError> {
        match &*self.mode {
            ClientMode::HTTPGateway(client) => {
                let url =
                    client
                        .base_url
                        .join("inference")
                        .map_err(|e| TensorZeroError::Other {
                            source: Error::new(ErrorDetails::InvalidBaseUrl {
                                message: format!(
                                    "Failed to join base URL with /inference endpoint: {e}"
                                ),
                            })
                            .into(),
                        })?;
                let body = serde_json::to_string(&params).map_err(|e| TensorZeroError::Other {
                    source: Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Failed to serialize inference params: {}",
                            DisplayOrDebug {
                                val: e,
                                debug: self.verbose_errors,
                            }
                        ),
                    })
                    .into(),
                })?;
                let mut builder = client
                    .http_client
                    .post(url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body.clone());

                if let Some(api_key) = params.api_key {
                    builder = builder.header(
                        reqwest::header::AUTHORIZATION,
                        format!("Bearer {}", api_key.expose_secret()),
                    );
                }

                // Add OTLP trace headers with the required prefix
                for (key, value) in &params.otlp_traces_extra_headers {
                    let header_name = format!("{TENSORZERO_OTLP_HEADERS_PREFIX}{key}");
                    builder = builder.header(header_name, value);
                }

                for (key, value) in &params.otlp_traces_extra_attributes {
                    let header_name = format!("{TENSORZERO_OTLP_ATTRIBUTE_PREFIX}{key}");
                    builder = builder.header(header_name, value);
                }

                for (key, value) in &params.otlp_traces_extra_resources {
                    let header_name = format!("{TENSORZERO_OTLP_RESOURCE_PREFIX}{key}");
                    builder = builder.header(header_name, value);
                }

                if params.stream.unwrap_or(false) {
                    Ok(HttpResponse {
                        response: InferenceOutput::Streaming(
                            client.send_http_stream_inference(builder).await?,
                        ),
                        raw_request: body,
                        raw_response: None,
                    })
                } else {
                    let (response, raw_response) =
                        client.send_and_parse_http_response(builder).await?;
                    Ok(HttpResponse {
                        response: InferenceOutput::NonStreaming(response),
                        raw_request: body,
                        raw_response: Some(raw_response),
                    })
                }
            }
            ClientMode::EmbeddedGateway { .. } => Err(TensorZeroError::Other {
                source: Error::new(ErrorDetails::InternalError {
                    message: "HTTP inference is not supported in embedded gateway mode".to_string(),
                })
                .into(),
            }),
        }
    }

    // Runs a TensorZero inference.
    // See https://www.tensorzero.com/docs/gateway/api-reference#post-inference
    pub async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceOutput, TensorZeroError> {
        match &*self.mode {
            ClientMode::HTTPGateway(_) => Ok(self.http_inference(params).await?.response),
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    let res = Box::pin(crate::endpoints::inference::inference(
                        gateway.handle.app_state.config.clone(),
                        &gateway.handle.app_state.http_client,
                        gateway.handle.app_state.clickhouse_connection_info.clone(),
                        gateway.handle.app_state.postgres_connection_info.clone(),
                        gateway.handle.app_state.deferred_tasks.clone(),
                        gateway.handle.app_state.rate_limiting_manager.clone(),
                        params.try_into().map_err(err_to_http)?,
                        // We currently ban auth-enabled configs in embedded gateway mode,
                        // so we don't have an API key here
                        None,
                    ))
                    .await
                    .map_err(err_to_http)?
                    .output;
                    match res {
                        InferenceOutput::NonStreaming(response) => {
                            Ok(InferenceOutput::NonStreaming(response))
                        }
                        InferenceOutput::Streaming(stream) => {
                            Ok(InferenceOutput::Streaming(stream))
                        }
                    }
                })
                .await?)
            }
        }
    }

    pub async fn get_object(
        &self,
        storage_path: StoragePath,
    ) -> Result<ObjectResponse, TensorZeroError> {
        match &*self.mode {
            ClientMode::HTTPGateway(client) => {
                let url = client
                    .base_url
                    .join("internal/object_storage")
                    .map_err(|e| TensorZeroError::Other {
                        source: Error::new(
                            ErrorDetails::InvalidBaseUrl {
                                message: format!(
                                    "Failed to join base URL with /internal/object_storage endpoint: {e}"
                                ),
                            },
                        )
                        .into(),
                    })?;
                let storage_path_json =
                    serde_json::to_string(&storage_path).map_err(|e| TensorZeroError::Other {
                        source: Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to serialize storage path: {e}"),
                        })
                        .into(),
                    })?;
                let builder = client
                    .http_client
                    .get(url)
                    .query(&[("storage_path", storage_path_json)]);
                Ok(client.send_and_parse_http_response(builder).await?.0)
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    crate::endpoints::object_storage::get_object(
                        gateway.handle.app_state.config.object_store_info.as_ref(),
                        storage_path,
                    )
                    .await
                    .map_err(err_to_http)
                })
                .await?)
            }
        }
    }
}

#[doc(hidden)]
pub async fn with_embedded_timeout<R, F: Future<Output = Result<R, TensorZeroError>>>(
    timeout: Option<Duration>,
    fut: F,
) -> Result<R, TensorZeroError> {
    if let Some(timeout) = timeout {
        tokio::time::timeout(timeout, fut)
            .await
            .map_err(|_: Elapsed| TensorZeroError::RequestTimeout)?
    } else {
        fut.await
    }
}

/// Load a config from a path.
/// This is a convenience function that wraps `Config::load_from_path_optional_verify_credentials`
/// and returns a `TensorZeroError` instead of a `ConfigError`.
/// This function does NOT verify credentials.
/// If the path is None, it returns the default config.
pub async fn get_config_no_verify_credentials(
    path: Option<PathBuf>,
) -> Result<UnwrittenConfig, TensorZeroError> {
    match path {
        Some(path) => Box::pin(Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new(path.to_string_lossy().to_string())
                .map_err(|e| TensorZeroError::Other { source: e.into() })?,
            false,
        ))
        .await
        .map_err(|e| TensorZeroError::Other { source: e.into() }),
        None => Ok(Box::pin(Config::new_empty())
            .await
            .map_err(|e| TensorZeroError::Other { source: e.into() })?),
    }
}

// This is intentionally not a `From` impl, since we only want to make fake HTTP
// errors for certain embedded gateway errors. For example, a config parsing error
// should be `TensorZeroError::Other`, not `TensorZeroError::Http`.
#[doc(hidden)]
pub fn err_to_http(e: Error) -> TensorZeroError {
    TensorZeroError::Http {
        status_code: e.status_code().as_u16(),
        text: Some(serde_json::json!({"error": e.to_string()}).to_string()),
        source: e.into(),
    }
}

#[cfg(feature = "pyo3")]
pub use crate::observability;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    #[tokio::test]
    async fn test_missing_clickhouse() {
        // This config file requires ClickHouse, so it should fail if no ClickHouse URL is provided
        let err = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(PathBuf::from("../clients/rust/tests/test_config.toml")),
            clickhouse_url: None,
            postgres_config: None,
            valkey_url: None,
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: true,
        })
        .build()
        .await
        .expect_err("ClientBuilder should have failed");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Missing environment variable TENSORZERO_CLICKHOUSE_URL"),
            "Bad error message: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_auth_not_supported_in_embedded() {
        // Create a config that enables auth, which is not supported in embedded mode
        let config = r"
        [gateway.auth]
        enabled = true
        ";
        let tmp_config = NamedTempFile::new().unwrap();
        std::fs::write(tmp_config.path(), config).unwrap();

        let err = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(tmp_config.path().to_owned()),
            clickhouse_url: None,
            postgres_config: None,
            valkey_url: None,
            timeout: None,
            verify_credentials: false, // Skip credential verification
            allow_batch_writes: false,
        })
        .build()
        .await
        .expect_err("ClientBuilder should have failed");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("`[gateway.auth]` is not supported in embedded gateway"),
            "Bad error message: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_from_components_rejects_auth() {
        // Create a config that enables auth, which is not supported in embedded mode
        let config_str = r"
        [gateway.auth]
        enabled = true
        ";
        let tmp_config = NamedTempFile::new().unwrap();
        std::fs::write(tmp_config.path(), config_str).unwrap();

        // Load config
        let config = Arc::new(
            Config::load_from_path_optional_verify_credentials(
                &ConfigFileGlob::new(tmp_config.path().to_string_lossy().to_string()).unwrap(),
                false,
            )
            .await
            .unwrap()
            .into_config_without_writing_for_tests(),
        );

        // Create mock components
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let postgres_connection_info = PostgresConnectionInfo::Disabled;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();

        // Attempt to build client with FromComponents mode
        let err = ClientBuilder::new(ClientBuilderMode::FromComponents {
            config,
            clickhouse_connection_info,
            postgres_connection_info,
            valkey_connection_info: ValkeyConnectionInfo::Disabled,
            http_client,
            timeout: None,
        })
        .build()
        .await
        .expect_err("ClientBuilder should have failed");

        // Verify it returns the correct error
        assert!(
            matches!(err, ClientBuilderError::AuthNotSupportedInEmbeddedMode(_)),
            "Expected AuthNotSupportedInEmbeddedMode error, got: {err:?}"
        );

        let err_msg = err.to_string();
        assert!(
            err_msg.contains("`[gateway.auth]` is not supported in embedded gateway"),
            "Bad error message: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_from_components_rejects_batch_writes() {
        // Create a config that enables batch writes, which is not supported in embedded mode
        let config_str = r"
        [gateway.observability.batch_writes]
        enabled = true
        ";
        let tmp_config = NamedTempFile::new().unwrap();
        std::fs::write(tmp_config.path(), config_str).unwrap();

        // Load config
        let config = Arc::new(
            Config::load_from_path_optional_verify_credentials(
                &ConfigFileGlob::new(tmp_config.path().to_string_lossy().to_string()).unwrap(),
                false,
            )
            .await
            .unwrap()
            .into_config_without_writing_for_tests(),
        );

        // Create mock components
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let postgres_connection_info = PostgresConnectionInfo::Disabled;
        let http_client = TensorzeroHttpClient::new_testing().unwrap();

        // Attempt to build client with FromComponents mode
        let err = ClientBuilder::new(ClientBuilderMode::FromComponents {
            config,
            clickhouse_connection_info,
            postgres_connection_info,
            valkey_connection_info: ValkeyConnectionInfo::Disabled,
            http_client,
            timeout: None,
        })
        .build()
        .await
        .expect_err("ClientBuilder should have failed");

        // Verify it returns the correct error
        assert!(
            matches!(err, ClientBuilderError::Clickhouse(_)),
            "Expected Clickhouse error, got: {err:?}"
        );

        let err_msg = err.to_string();
        assert!(
            err_msg.contains("`[gateway.observability.batch_writes]` is not yet supported"),
            "Bad error message: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_log_no_clickhouse() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Default observability and no ClickHouse URL
        ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(PathBuf::from(
                "../examples/haiku-hidden-preferences/config/tensorzero.toml",
            )),
            clickhouse_url: None,
            postgres_config: None,
            valkey_url: None,
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: true,
        })
        .build()
        .await
        .expect("Failed to build client");
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));
        assert!(logs_contain(
            "Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and `clickhouse_url` was not provided."
        ));
    }

    #[tokio::test]
    async fn test_log_no_config() {
        let logs_contain = crate::utils::testing::capture_logs();
        ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: None,
            clickhouse_url: None,
            postgres_config: None,
            valkey_url: None,
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: true,
        })
        .build()
        .await
        .expect("Failed to build client");
        assert!(!logs_contain(
            "Missing environment variable TENSORZERO_CLICKHOUSE_URL"
        ));
        assert!(logs_contain(
            "No config file provided, so only default functions will be available. Set `config_file` to specify your `tensorzero.toml`"
        ));
        assert!(logs_contain(
            "Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and `clickhouse_url` was not provided."
        ));
    }

    #[tokio::test]
    async fn test_feature_flags_are_initialized() {
        ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: None,
            clickhouse_url: None,
            postgres_config: None,
            valkey_url: None,
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: true,
        })
        .build()
        .await
        .expect("Failed to build client");

        assert!(
            !feature_flags::TEST_FLAG.get(),
            "Should be able to get TEST_FLAG value without panic"
        );
    }
}
