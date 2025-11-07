use std::{env, fmt::Display, future::Future, path::PathBuf, sync::Arc, time::Duration};

use crate::config::ConfigFileGlob;
use crate::http::TensorzeroHttpClient;
use crate::inference::types::stored_input::StoragePathResolver;
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    utils::gateway::{setup_clickhouse, setup_postgres, GatewayHandle},
};
use reqwest::header::HeaderMap;
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use std::fmt::Debug;
use thiserror::Error;
use tokio::{sync::Mutex, time::error::Elapsed};
use tokio_stream::StreamExt;
use url::Url;

pub use client_inference_params::{ClientInferenceParams, ClientSecretString};
pub use client_input::{ClientInput, ClientInputMessage, ClientInputMessageContent};
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
pub mod client_input;
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

pub struct HTTPGateway {
    pub base_url: Url,
    pub http_client: reqwest::Client,
    pub gateway_version: Mutex<Option<String>>, // Needs interior mutability so it can be set on every request
}

impl HTTPGateway {
    /// Sets the gateway version on the HTTPGateway struct.
    /// This should be called if the HTTPGateway is constructed within an async context.
    pub async fn discover_initialize_gateway_version(
        &self,
        client: &Client,
    ) -> Result<(), ClientBuilderError> {
        let status_url = self.base_url.join("status").map_err(|_| {
            ClientBuilderError::GatewayVersion("Failed to construct /status URL".to_string())
        })?;
        // If the client is initialized and the ping for version fails, we simply don't set it.
        let status_response = match self.http_client.get(status_url).send().await {
            Ok(status_response) => status_response,
            Err(_) => return Ok(()),
        };

        client
            .update_gateway_version_from_headers(status_response.headers())
            .await;

        Ok(())
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
    http_client: Option<reqwest::Client>,
    verbose_errors: bool,
    api_key: Option<String>,
    timeout: Option<Duration>,
}

/// An error type representing an error from within the TensorZero gateway
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TensorZeroError {
    #[error("HTTP Error (status code {status_code}): {text:?}")]
    Http {
        status_code: u16,
        text: Option<String>,
        #[source]
        source: TensorZeroInternalError,
    },
    #[error("{source}")] // the `source` has already been formatted (below)
    Other {
        #[source]
        source: TensorZeroInternalError,
    },
    #[error("HTTP Error: request timed out")]
    RequestTimeout,
    #[error("Failed to get git info: {source}")]
    Git {
        #[source]
        source: git2::Error,
    },
}

#[derive(Debug, Error)]
#[error("Internal TensorZero Error: {0}")]
pub struct TensorZeroInternalError(#[from] crate::error::Error);

#[derive(Error, Debug)]
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

/// Controls how a `Client` is run
pub enum ClientBuilderMode {
    /// In HTTPGateway mode, we make HTTP requests to a TensorZero gateway server.
    HTTPGateway { url: Url },
    /// In EmbeddedGateway mode, we run an embedded gateway using a config file.
    /// We do not launch an HTTP server - we only make outgoing HTTP requests to model providers and to ClickHouse.
    EmbeddedGateway {
        config_file: Option<PathBuf>,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
        /// A timeout for all TensorZero gateway processing.
        /// If this timeout is hit, any in-progress LLM requests may be aborted.
        timeout: Option<std::time::Duration>,
        verify_credentials: bool,
        // Allow turning on batch writes - used in e2e tests.
        // We don't expose this through the Python client, since we're having deadlock issues
        // there.
        allow_batch_writes: bool,
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
        }
    }

    /// Sets the `reqwest::Client` to be used when making any HTTP requests.
    /// In `EmbeddedGateway` mode, this is used for making requests to model endpoints,
    /// as well as ClickHouse.
    /// In `HTTPGateway` mode, this is used for making requests to the gateway.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
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
        self.api_key = Some(api_key);
        self
    }

    /// Sets the timeout for HTTP requests to the TensorZero gateway.
    /// This is only used in `HTTPGateway` mode.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Constructs a `Client`, returning an error if the configuration is invalid.
    pub async fn build(self) -> Result<Client, ClientBuilderError> {
        match &self.mode {
            ClientBuilderMode::HTTPGateway { .. } => {
                let client = self.build_http()?;
                if let ClientMode::HTTPGateway(mode) = &*client.mode {
                    mode.discover_initialize_gateway_version(&client).await?;
                }
                Ok(client)
            }
            ClientBuilderMode::EmbeddedGateway {
                config_file,
                clickhouse_url,
                postgres_url,
                timeout,
                verify_credentials,
                allow_batch_writes,
            } => {
                let config = if let Some(config_file) = config_file {
                    let glob = ConfigFileGlob::new(config_file.to_string_lossy().to_string())
                        .map_err(|e| {
                            ClientBuilderError::ConfigParsingPreGlob(TensorZeroError::Other {
                                source: e.into(),
                            })
                        })?;
                    Arc::new(
                        Config::load_from_path_optional_verify_credentials(
                            &glob,
                            *verify_credentials,
                        )
                        .await
                        .map_err(|e| {
                            ClientBuilderError::ConfigParsing {
                                error: TensorZeroError::Other { source: e.into() },
                                glob,
                            }
                        })?,
                    )
                } else {
                    tracing::info!("No config file provided, so only default functions will be available. Set `config_file` to specify your `tensorzero.toml`");
                    Arc::new(Config::new_empty().await.map_err(|e| {
                        ClientBuilderError::ConfigParsing {
                            error: TensorZeroError::Other { source: e.into() },
                            glob: ConfigFileGlob::new_empty(),
                        }
                    })?)
                };
                if !allow_batch_writes
                    && config.gateway.observability.batch_writes.enabled
                    && !config
                        .gateway
                        .observability
                        .batch_writes
                        .__force_allow_embedded_batch_writes
                {
                    return Err(ClientBuilderError::Clickhouse(TensorZeroError::Other {
                        source: crate::error::Error::new(ErrorDetails::Config {
                            message: "`[gateway.observability.batch_writes]` is not yet supported in embedded gateway mode".to_string(),
                        })
                        .into(),
                    }));
                }
                if config.gateway.auth.enabled {
                    return Err(ClientBuilderError::AuthNotSupportedInEmbeddedMode(TensorZeroError::Other {
                        source: crate::error::Error::new(ErrorDetails::Config {
                            message: "`[gateway.auth]` is not supported in embedded gateway mode. Authentication is only available when using HTTP gateway mode. Please either disable authentication by setting `gateway.auth.enabled = false` or use HTTP mode instead.".to_string(),
                        })
                        .into(),
                    }));
                }
                let clickhouse_connection_info =
                    setup_clickhouse(&config, clickhouse_url.clone(), true)
                        .await
                        .map_err(|e| {
                            ClientBuilderError::Clickhouse(TensorZeroError::Other {
                                source: e.into(),
                            })
                        })?;
                let postgres_connection_info = setup_postgres(&config, postgres_url.clone())
                    .await
                    .map_err(|e| {
                        ClientBuilderError::Postgres(TensorZeroError::Other { source: e.into() })
                    })?;

                let http_client = if self.http_client.is_some() {
                    return Err(ClientBuilderError::HTTPClientBuild(
                        TensorZeroError::Other {
                            source: TensorZeroInternalError(crate::error::Error::new(
                                ErrorDetails::AppState {
                                    message:
                                        "HTTP client cannot be provided in EmbeddedGateway mode"
                                            .to_string(),
                                },
                            )),
                        },
                    ));
                } else {
                    TensorzeroHttpClient::new(config.gateway.global_outbound_http_timeout).map_err(
                        |e| {
                            ClientBuilderError::HTTPClientBuild(TensorZeroError::Other {
                                source: e.into(),
                            })
                        },
                    )?
                };
                Ok(Client {
                    mode: Arc::new(ClientMode::EmbeddedGateway {
                        gateway: EmbeddedGateway {
                            handle: GatewayHandle::new_with_database_and_http_client(
                                config,
                                clickhouse_connection_info,
                                postgres_connection_info,
                                http_client,
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
                    #[cfg(feature = "e2e_tests")]
                    last_body: Default::default(),
                })
            }
        }
    }

    /// Builds a dummy client for use in pyo3. Should not otherwise be used
    /// This avoids logging any messages
    ///
    /// # Panics
    /// This will panic if a `TensorzeroHttpClient` cannot be constructed
    /// due to an error when building a `reqwest::Client`
    /// (e.g. if a TLS backend cannot be initialized)
    #[cfg(feature = "pyo3")]
    pub fn build_dummy() -> Client {
        let handle = GatewayHandle::new_dummy(
            // NOTE - we previously called `reqwest::Client::new()`, which panics
            // if a TLS backend cannot be initialized.
            // This explicit `expect` does not actually increase the risk of panics,
            #[expect(clippy::expect_used)]
            TensorzeroHttpClient::new(crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT)
                .expect("Failed to construct TensorzeroHttpClient"),
        );
        Client {
            mode: Arc::new(ClientMode::EmbeddedGateway {
                gateway: EmbeddedGateway { handle },
                timeout: None,
            }),
            verbose_errors: false,
            #[cfg(feature = "e2e_tests")]
            last_body: Default::default(),
        }
    }

    #[cfg(any(test, feature = "e2e_tests"))]
    pub async fn build_from_state(handle: GatewayHandle) -> Result<Client, ClientBuilderError> {
        Ok(Client {
            mode: Arc::new(ClientMode::EmbeddedGateway {
                gateway: EmbeddedGateway { handle },
                timeout: None,
            }),
            verbose_errors: false,
            #[cfg(feature = "e2e_tests")]
            last_body: Default::default(),
        })
    }

    /// Builds a `Client` in HTTPGateway mode, erroring if the mode is not HTTPGateway
    /// This allows avoiding calling the async `build` method
    pub fn build_http(self) -> Result<Client, ClientBuilderError> {
        let ClientBuilderMode::HTTPGateway { mut url } = self.mode else {
            return Err(ClientBuilderError::NotHTTPGateway);
        };
        // Enforce that the URL has a trailing slash, so that joining endpoints works correctly
        // This means that passing in a url that looks like 'http://example.com/some/prefix'
        // will result in inference requests being sent to 'http://example.com/some/prefix/inference'
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }

        // Try to get API key from constructor parameter, otherwise try environment variable
        let api_key = self.api_key.or_else(|| env::var("TENSORZERO_API_KEY").ok());

        // Build the HTTP client, applying timeout and/or API key
        let http_client = if let Some(client) = self.http_client {
            // Use custom client provided by advanced users

            // TODO: Later we can decide if we want to override the custom HTTP clients.

            if self.timeout.is_some() {
                tracing::warn!("A timeout is set but a custom HTTP client is being used. The TensorZero SDK will not automatically apply the timeout to the custom client.");
            }

            if api_key.is_some() {
                tracing::warn!("A TensorZero API key is available but a custom HTTP client is being used. The TensorZero SDK will not automatically apply the authentication header to the custom client.");
            }

            client
        } else {
            // Build client from scratch, composing timeout and api_key
            let mut builder = reqwest::Client::builder();

            // Apply timeout if provided
            if let Some(timeout) = self.timeout {
                builder = builder.timeout(timeout);
            }

            // Apply API key as default Authorization header if provided
            if let Some(ref key) = api_key {
                let mut headers = HeaderMap::new();
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")).map_err(
                        |e| {
                            ClientBuilderError::HTTPClientBuild(TensorZeroError::Other {
                                source: Error::new(ErrorDetails::InternalError {
                                    message: format!("Failed to create authorization header: {e}"),
                                })
                                .into(),
                            })
                        },
                    )?,
                );
                builder = builder.default_headers(headers);
            }

            builder.build().map_err(|e| {
                ClientBuilderError::HTTPClientBuild(TensorZeroError::Other {
                    source: Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to build HTTP client: {e}"),
                    })
                    .into(),
                })
            })?
        };

        Ok(Client {
            mode: Arc::new(ClientMode::HTTPGateway(HTTPGateway {
                base_url: url,
                http_client,
                gateway_version: Mutex::new(None),
            })),
            verbose_errors: self.verbose_errors,
            #[cfg(feature = "e2e_tests")]
            last_body: Default::default(),
        })
    }
}

/// A TensorZero client. This is constructed using `ClientBuilder`
#[derive(Debug, Clone)]
pub struct Client {
    mode: Arc<ClientMode>,
    pub verbose_errors: bool,
    #[cfg(feature = "e2e_tests")]
    pub last_body: Arc<Mutex<Option<String>>>,
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
                        source: crate::error::Error::new(ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /feedback endpoint: {e}"
                            ),
                        })
                        .into(),
                    })?;
                let builder = client.http_client.post(url).json(&params);
                self.parse_http_response(builder.send().await).await
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

    // Runs a TensorZero inference.
    // See https://www.tensorzero.com/docs/gateway/api-reference#post-inference
    pub async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceOutput, TensorZeroError> {
        match &*self.mode {
            ClientMode::HTTPGateway(client) => {
                let url =
                    client
                        .base_url
                        .join("inference")
                        .map_err(|e| TensorZeroError::Other {
                            source: crate::error::Error::new(ErrorDetails::InvalidBaseUrl {
                                message: format!(
                                    "Failed to join base URL with /inference endpoint: {e}"
                                ),
                            })
                            .into(),
                        })?;
                let body = serde_json::to_string(&params).map_err(|e| TensorZeroError::Other {
                    source: crate::error::Error::new(ErrorDetails::Serialization {
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
                #[cfg(feature = "e2e_tests")]
                {
                    *self.last_body.lock().await = Some(body.clone());
                }
                let mut builder = client
                    .http_client
                    .post(url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body);

                // Add OTLP trace headers with the required prefix
                for (key, value) in &params.otlp_traces_extra_headers {
                    let header_name = format!("tensorzero-otlp-traces-extra-header-{key}");
                    builder = builder.header(header_name, value);
                }

                if params.stream.unwrap_or(false) {
                    let event_source =
                        builder.eventsource().map_err(|e| TensorZeroError::Other {
                            source: crate::error::Error::new(ErrorDetails::JsonRequest {
                                message: format!("Error constructing event stream: {e:?}"),
                            })
                            .into(),
                        })?;
                    Ok(InferenceOutput::Streaming(
                        self.http_inference_stream(event_source).await?,
                    ))
                } else {
                    Ok(InferenceOutput::NonStreaming(
                        self.parse_http_response(builder.send().await).await?,
                    ))
                }
            }
            ClientMode::EmbeddedGateway { gateway, timeout } => {
                Ok(with_embedded_timeout(*timeout, async {
                    let res = crate::endpoints::inference::inference(
                        gateway.handle.app_state.config.clone(),
                        &gateway.handle.app_state.http_client,
                        gateway.handle.app_state.clickhouse_connection_info.clone(),
                        gateway.handle.app_state.postgres_connection_info.clone(),
                        gateway.handle.app_state.deferred_tasks.clone(),
                        params.try_into().map_err(err_to_http)?,
                        // We currently ban auth-enabled configs in embedded gateway mode,
                        // so we don't have an API key here
                        None,
                    )
                    .await
                    .map_err(err_to_http)?;
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
                        source: crate::error::Error::new(
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
                        source: crate::error::Error::new(ErrorDetails::Serialization {
                            message: format!("Failed to serialize storage path: {e}"),
                        })
                        .into(),
                    })?;
                let builder = client
                    .http_client
                    .get(url)
                    .query(&[("storage_path", storage_path_json)]);
                self.parse_http_response(builder.send().await).await
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

    pub async fn check_http_response(
        &self,
        resp: Result<reqwest::Response, reqwest::Error>,
    ) -> Result<reqwest::Response, TensorZeroError> {
        let resp = resp.map_err(|e| {
            if e.is_timeout() {
                TensorZeroError::RequestTimeout
            } else {
                TensorZeroError::Other {
                    source: crate::error::Error::new(ErrorDetails::JsonRequest {
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

        self.update_gateway_version_from_headers(resp.headers())
            .await;

        if let Err(e) = resp.error_for_status_ref() {
            let status_code = resp.status().as_u16();
            let text = resp.text().await.ok();
            return Err(TensorZeroError::Http {
                status_code,
                text,
                source: crate::error::Error::new(ErrorDetails::JsonRequest {
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

    pub async fn parse_http_response<T: serde::de::DeserializeOwned>(
        &self,
        resp: Result<reqwest::Response, reqwest::Error>,
    ) -> Result<T, TensorZeroError> {
        self.check_http_response(resp)
            .await?
            .json()
            .await
            .map_err(|e| TensorZeroError::Other {
                source: crate::error::Error::new(ErrorDetails::Serialization {
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

    async fn http_inference_stream(
        &self,
        event_source: EventSource,
    ) -> Result<InferenceStream, TensorZeroError> {
        let mut event_source = event_source.peekable();
        let first = event_source.peek().await;
        if let Some(Err(_)) = first {
            // Discard the stream if it has an error
            let res = event_source.next().await;
            #[expect(clippy::panic)]
            let Some(Err(e)) = res
            else {
                panic!("Peeked error but got non-err {res:?}");
            };
            let err_str = format!("Error in streaming response: {e:?}");
            let inner_err = crate::error::Error::new(ErrorDetails::StreamError {
                source: Box::new(crate::error::Error::new(ErrorDetails::Serialization {
                    message: err_str,
                })),
            });
            if let reqwest_eventsource::Error::InvalidStatusCode(code, resp) = e {
                return Err(TensorZeroError::Http {
                    status_code: code.as_u16(),
                    text: resp.text().await.ok(),
                    source: inner_err.into(),
                });
            }
            return Err(TensorZeroError::Other {
                source: crate::error::Error::new(ErrorDetails::StreamError {
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
                        if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                            break;
                        }
                        yield Err(crate::error::Error::new(ErrorDetails::StreamError {
                            source: Box::new(crate::error::Error::new(ErrorDetails::Serialization {
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
                                crate::error::Error::new(ErrorDetails::Serialization {
                                    message: format!("Error deserializing inference response chunk: {}", DisplayOrDebug {
                                        val: e,
                                        debug: verbose_errors,
                                    }),
                                })
                            })?;
                            if let Some(err) = json.get("error") {
                                yield Err(crate::error::Error::new(ErrorDetails::StreamError {
                                    source: Box::new(crate::error::Error::new(ErrorDetails::Serialization {
                                        message: format!("Stream produced an error: {}", DisplayOrDebug {
                                            val: err,
                                            debug: verbose_errors,
                                        }),
                                    }))
                                }));
                            } else {
                                let data: InferenceResponseChunk =
                                serde_json::from_value(json).map_err(|e| {
                                    crate::error::Error::new(ErrorDetails::Serialization {
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

    async fn update_gateway_version_from_headers(&self, headers: &HeaderMap) {
        let mut version = headers
            .get("x-tensorzero-gateway-version")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if cfg!(feature = "e2e_tests") {
            if let Ok(version_override) = env::var("TENSORZERO_E2E_GATEWAY_VERSION_OVERRIDE") {
                version = Some(version_override);
            }
        };
        if let Some(version) = version {
            self.update_gateway_version(version).await;
        }
    }

    async fn update_gateway_version(&self, version: String) {
        match &*self.mode {
            ClientMode::HTTPGateway(client) => {
                // Acquire the lock on the gateway version
                let mut gateway_version = client.gateway_version.lock().await;
                *gateway_version = Some(version);
            }
            // Should never be called
            ClientMode::EmbeddedGateway { .. } => {}
        }
    }

    #[cfg(feature = "e2e_tests")]
    pub async fn e2e_update_gateway_version(&self, version: String) {
        self.update_gateway_version(version).await;
    }

    #[cfg(feature = "e2e_tests")]
    pub async fn get_gateway_version(&self) -> Option<String> {
        match &*self.mode {
            ClientMode::HTTPGateway(client) => client.gateway_version.lock().await.clone(),
            ClientMode::EmbeddedGateway { .. } => None,
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
) -> Result<Config, TensorZeroError> {
    match path {
        Some(path) => Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new(path.to_string_lossy().to_string())
                .map_err(|e| TensorZeroError::Other { source: e.into() })?,
            false,
        )
        .await
        .map_err(|e| TensorZeroError::Other { source: e.into() }),
        None => Ok(Config::new_empty()
            .await
            .map_err(|e| TensorZeroError::Other { source: e.into() })?),
    }
}

// This is intentionally not a `From` impl, since we only want to make fake HTTP
// errors for certain embedded gateway errors. For example, a config parsing error
// should be `TensorZeroError::Other`, not `TensorZeroError::Http`.
#[doc(hidden)]
pub fn err_to_http(e: crate::error::Error) -> TensorZeroError {
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
            postgres_url: None,
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
            postgres_url: None,
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
    async fn test_log_no_clickhouse() {
        let logs_contain = crate::utils::testing::capture_logs();
        // Default observability and no ClickHouse URL
        ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(PathBuf::from(
                "../examples/haiku-hidden-preferences/config/tensorzero.toml",
            )),
            clickhouse_url: None,
            postgres_url: None,
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
        assert!(logs_contain("Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and `clickhouse_url` was not provided."));
    }

    #[tokio::test]
    async fn test_log_no_config() {
        let logs_contain = crate::utils::testing::capture_logs();
        ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: None,
            clickhouse_url: None,
            postgres_url: None,
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
        assert!(logs_contain("No config file provided, so only default functions will be available. Set `config_file` to specify your `tensorzero.toml`"));
        assert!(logs_contain("Disabling observability: `gateway.observability.enabled` is not explicitly specified in config and `clickhouse_url` was not provided."));
    }
}
