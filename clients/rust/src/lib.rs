use std::{path::PathBuf, sync::Arc};

use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use std::fmt::Debug;
use tensorzero_internal::{
    config_parser::Config,
    error::ErrorDetails,
    gateway_util::{setup_clickhouse, AppStateData},
};
use thiserror::Error;
use tokio_stream::StreamExt;
use url::Url;

mod client_inference_params;

pub use client_inference_params::{ClientInferenceParams, ClientSecretString};

pub use tensorzero_internal::cache::CacheParamsOptions;
pub use tensorzero_internal::endpoints::feedback::FeedbackResponse;
pub use tensorzero_internal::endpoints::feedback::Params as FeedbackParams;
pub use tensorzero_internal::endpoints::inference::{
    InferenceOutput, InferenceParams, InferenceResponse, InferenceResponseChunk, InferenceStream,
};
pub use tensorzero_internal::inference::types::{
    ContentBlockChunk, Input, InputMessage, InputMessageContent, Role,
};
pub use tensorzero_internal::tool::{DynamicToolParams, Tool};

enum ClientMode {
    HTTPGateway(HTTPGateway),
    EmbeddedGateway(EmbeddedGateway),
}

impl Debug for ClientMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientMode::HTTPGateway(_) => write!(f, "HTTPGateway"),
            ClientMode::EmbeddedGateway(_) => write!(f, "EmbeddedGateway"),
        }
    }
}

struct HTTPGateway {
    base_url: Url,
    http_client: reqwest::Client,
}

struct EmbeddedGateway {
    state: AppStateData,
}

/// Used to construct a `Client`
/// Call `ClientBuilder::new` to create a new `ClientBuilder`
/// in either `HTTPGateway` or `EmbeddedGateway` mode
pub struct ClientBuilder {
    mode: ClientBuilderMode,
    http_client: Option<reqwest::Client>,
}

/// An error type representing an error from within the TensorZero gateway
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TensorZeroError {
    #[error("HTTP error: {status_code} {text:?}")]
    Http {
        status_code: u16,
        text: Option<String>,
        #[source]
        source: TensorZeroInternalError,
    },
    #[error("Internal error: {source}")]
    Other {
        #[source]
        source: TensorZeroInternalError,
    },
}

#[derive(Debug, Error)]
#[error("Internal TensorZero error: {0}")]
pub struct TensorZeroInternalError(#[from] tensorzero_internal::error::Error);

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ClientBuilderError {
    #[error(
        "Missing config - you must call `with_config_path` before calling `build` in EmbeddedGateway mode"
    )]
    MissingConfig,
    #[error(
        "Missing gateway URL - you must call `with_gateway_url` before calling `build` in HTTPGateway mode"
    )]
    MissingGatewayUrl,
    #[error("Called ClientBuilder.build_http() when not in HTTPGateway mode")]
    NotHTTPGateway,
    #[error("Failed to configure ClickHouse: {0}")]
    Clickhouse(TensorZeroError),
    #[error("Failed to parse config: {0}")]
    ConfigParsing(TensorZeroError),
}

/// Controls how a `Client` is run
pub enum ClientBuilderMode {
    /// In HTTPGateway mode, we make HTTP requests to a TensorZero gateway server.
    HTTPGateway { url: Url },
    /// In EmbeddedGateway mode, we run an embedded gateway using a config file.
    /// We do not launch an HTTP server - we only make outgoing HTTP requests to model providers and to ClickHouse.
    EmbeddedGateway {
        config_path: PathBuf,
        clickhouse_url: Option<String>,
    },
}

/// A `ClientBuilder` is used to construct a `Client`.
impl ClientBuilder {
    pub fn new(mode: ClientBuilderMode) -> Self {
        Self {
            mode,
            http_client: None,
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

    /// Constructs a `Client`, returning an error if the configuration is invalid.
    pub async fn build(self) -> Result<Client, ClientBuilderError> {
        match &self.mode {
            ClientBuilderMode::HTTPGateway { .. } => self.build_http(),
            ClientBuilderMode::EmbeddedGateway {
                config_path,
                clickhouse_url,
            } => {
                let config = Arc::new(Config::load_from_path(config_path).map_err(|e| {
                    ClientBuilderError::ConfigParsing(TensorZeroError::Other { source: e.into() })
                })?);
                let clickhouse_connection_info = setup_clickhouse(&config, clickhouse_url.clone())
                    .await
                    .map_err(|e| {
                        ClientBuilderError::Clickhouse(TensorZeroError::Other { source: e.into() })
                    })?;
                Ok(Client {
                    mode: ClientMode::EmbeddedGateway(EmbeddedGateway {
                        state: AppStateData {
                            config,
                            http_client: self.http_client.unwrap_or_default(),
                            clickhouse_connection_info,
                        },
                    }),
                })
            }
        }
    }

    /// Builds a `Client` in HTTPGateway mode, erroring if the mode is not HTTPGateway
    /// This allows avoiding calling the async `build` method
    pub fn build_http(self) -> Result<Client, ClientBuilderError> {
        let ClientBuilderMode::HTTPGateway { url } = self.mode else {
            return Err(ClientBuilderError::NotHTTPGateway);
        };
        Ok(Client {
            mode: ClientMode::HTTPGateway(HTTPGateway {
                base_url: url,
                http_client: self.http_client.unwrap_or_default(),
            }),
        })
    }
}

/// A TensorZero client. This is constructed using `ClientBuilder`
#[derive(Debug)]
pub struct Client {
    mode: ClientMode,
}

impl Client {
    /// Queries the health of the ClickHouse database
    /// This does nothing in `ClientMode::HTTPGateway`
    pub async fn clickhouse_health(&self) -> Result<(), TensorZeroError> {
        match &self.mode {
            ClientMode::HTTPGateway(_) => Ok(()),
            ClientMode::EmbeddedGateway(client) => client
                .state
                .clickhouse_connection_info
                .health()
                .await
                .map_err(|e| TensorZeroError::Other { source: e.into() }),
        }
    }

    /// Assigns feedback for a TensorZero inference.
    /// See https://www.tensorzero.com/docs/gateway/api-reference#post-feedback
    pub async fn feedback(
        &self,
        params: FeedbackParams,
    ) -> Result<FeedbackResponse, TensorZeroError> {
        match &self.mode {
            ClientMode::HTTPGateway(client) => {
                let url = client
                    .base_url
                    .join("feedback")
                    .map_err(|e| TensorZeroError::Other {
                        source: tensorzero_internal::error::Error::new(
                            ErrorDetails::InvalidBaseUrl {
                                message: format!(
                                    "Failed to join base URL with /feedback endpoint: {}",
                                    e
                                ),
                            },
                        )
                        .into(),
                    })?;
                let builder = client.http_client.post(url).json(&params);
                self.parse_http_response(builder.send().await).await
            }
            ClientMode::EmbeddedGateway(client) => Ok(
                tensorzero_internal::endpoints::feedback::feedback(client.state.clone(), params)
                    .await
                    .map_err(err_to_http)?
                    .0,
            ),
        }
    }

    // Runs a TensorZero inference.
    // See https://www.tensorzero.com/docs/gateway/api-reference#post-inference
    pub async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceOutput, TensorZeroError> {
        match &self.mode {
            ClientMode::HTTPGateway(client) => {
                let url =
                    client
                        .base_url
                        .join("inference")
                        .map_err(|e| TensorZeroError::Other {
                            source: tensorzero_internal::error::Error::new(
                                ErrorDetails::InvalidBaseUrl {
                                    message: format!(
                                        "Failed to join base URL with /inference endpoint: {}",
                                        e
                                    ),
                                },
                            )
                            .into(),
                        })?;
                let builder = client.http_client.post(url).json(&params);
                if params.stream.unwrap_or(false) {
                    let event_source =
                        builder.eventsource().map_err(|e| TensorZeroError::Other {
                            source: tensorzero_internal::error::Error::new(
                                ErrorDetails::JsonRequest {
                                    message: format!("Error constructing event stream: {e:?}"),
                                },
                            )
                            .into(),
                        })?;
                    Ok(InferenceOutput::Streaming(
                        Self::http_inference_stream(event_source).await?,
                    ))
                } else {
                    Ok(InferenceOutput::NonStreaming(
                        self.parse_http_response(builder.send().await).await?,
                    ))
                }
            }
            ClientMode::EmbeddedGateway(client) => {
                Ok(tensorzero_internal::endpoints::inference::inference(
                    client.state.config.clone(),
                    &client.state.http_client,
                    client.state.clickhouse_connection_info.clone(),
                    params.into(),
                )
                .await
                .map_err(err_to_http)?)
            }
        }
    }

    async fn parse_http_response<T: serde::de::DeserializeOwned>(
        &self,
        resp: Result<reqwest::Response, reqwest::Error>,
    ) -> Result<T, TensorZeroError> {
        let resp = resp.map_err(|e| TensorZeroError::Other {
            source: tensorzero_internal::error::Error::new(ErrorDetails::JsonRequest {
                message: format!("Error from server: {e}"),
            })
            .into(),
        })?;

        if let Err(e) = resp.error_for_status_ref() {
            let status_code = resp.status().as_u16();
            let text = resp.text().await.ok();
            return Err(TensorZeroError::Http {
                status_code,
                text,
                source: tensorzero_internal::error::Error::new(ErrorDetails::JsonRequest {
                    message: format!("Request failed: {e}"),
                })
                .into(),
            });
        }

        resp.json().await.map_err(|e| TensorZeroError::Other {
            source: tensorzero_internal::error::Error::new(ErrorDetails::Serialization {
                message: format!("Error deserializing inference response: {e:?}"),
            })
            .into(),
        })
    }

    async fn http_inference_stream(
        event_source: EventSource,
    ) -> Result<InferenceStream, TensorZeroError> {
        let mut event_source = event_source.peekable();
        let first = event_source.peek().await;
        if let Some(Err(_)) = first {
            // Discard the stream if it has an error
            let res = event_source.next().await;
            #[allow(clippy::panic)]
            let Some(Err(e)) = res
            else {
                panic!("Peeked error but got non-err {res:?}");
            };
            let err_str = format!("Error in streaming response: {e:?}");
            let inner_err = tensorzero_internal::error::Error::new(ErrorDetails::StreamError {
                source: Box::new(tensorzero_internal::error::Error::new(
                    ErrorDetails::Serialization { message: err_str },
                )),
            });
            if let reqwest_eventsource::Error::InvalidStatusCode(code, resp) = e {
                return Err(TensorZeroError::Http {
                    status_code: code.as_u16(),
                    text: resp.text().await.ok(),
                    source: inner_err.into(),
                });
            }
            return Err(TensorZeroError::Other {
                source: tensorzero_internal::error::Error::new(ErrorDetails::StreamError {
                    source: Box::new(inner_err),
                })
                .into(),
            });
        }
        Ok(Box::pin(async_stream::stream! {
            while let Some(ev) = event_source.next().await {
                match ev {
                    Err(e) => {
                        if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                            break;
                        }
                        yield Err(tensorzero_internal::error::Error::new(ErrorDetails::StreamError {
                            source: Box::new(tensorzero_internal::error::Error::new(ErrorDetails::Serialization {
                                message: format!("Error in streaming response: {e:?}")
                            }))
                        }))
                    }
                    Ok(e) => match e {
                        Event::Open => continue,
                        Event::Message(message) => {
                            if message.data == "[DONE]" {
                                break;
                            }
                            let data: InferenceResponseChunk =
                                serde_json::from_str(&message.data).map_err(|e| {
                                    tensorzero_internal::error::Error::new(ErrorDetails::Serialization {
                                        message: format!("Error deserializing inference response chunk: {e:?}"),
                                    })
                                })?;

                            yield Ok(data);
                        }
                    }
                }
            }
        }))
    }
}

// This is intentionally not a `From` impl, since we only want to make fake HTTP
// errors for certain embedded gateway errors. For example, a config parsing error
// should be `TensorZeroError::Other`, not `TensorZeroError::Http`.
#[doc(hidden)]
pub fn err_to_http(e: tensorzero_internal::error::Error) -> TensorZeroError {
    TensorZeroError::Http {
        status_code: e.status_code().as_u16(),
        text: Some(serde_json::json!({"error": e.to_string()}).to_string()),
        source: e.into(),
    }
}

#[cfg(feature = "pyo3")]
pub use tensorzero_internal::observability;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // TODO - set an environment variable, or create a new config with dummy credentials
    async fn test_missing_clickhouse() {
        // This config file requires ClickHouse, so it should fail if no ClickHouse URL is provided
        let err = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_path: PathBuf::from(
                "../../examples/haiku-hidden-preferences/config/tensorzero.toml",
            ),
            clickhouse_url: None,
        })
        .build()
        .await
        .expect_err("ClientBuilder should have failed");
        let err_msg = err.to_string();
        assert!(
            err.to_string().contains("Missing ClickHouse URL"),
            "Bad error message: {err_msg}"
        );
    }
}
