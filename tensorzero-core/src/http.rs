use std::{
    pin::Pin,
    sync::{atomic::AtomicU8, Arc},
    task::{Context, Poll},
};

use futures::Stream;
use http::{HeaderName, HeaderValue};
use pin_project::pin_project;
use reqwest::{Body, Response};
use reqwest::{Client, IntoUrl, NoProxy, Proxy, RequestBuilder};
use reqwest_eventsource::{CannotCloneRequestError, Event, EventSource, RequestBuilderExt};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    model_table::CowNoClone,
};

struct LimitedClient {
    // Currently never incremented - we'll use this when we implement
    // connection pooling
    concurrent_requests: Arc<AtomicU8>,
    client: Client,
}

struct LimitedClientTicket<'a> {
    client: CowNoClone<'a, LimitedClient>,
}

impl Drop for LimitedClientTicket<'_> {
    fn drop(&mut self) {
        // We'll implement decrementing the counter here when we implement
        // connection pooling
    }
}

/// A wrapper for `reqwest::Client` that adds extra features:
/// * Improved connection pooling support for HTTP/2
#[derive(Clone)]
pub struct TensorzeroHttpClient {
    fallback_client: Arc<LimitedClient>,
}

impl TensorzeroHttpClient {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            fallback_client: Arc::new(LimitedClient {
                concurrent_requests: Arc::new(AtomicU8::new(0)),
                client: build_client()?,
            }),
        })
    }

    // This will eventually use a connection pool
    fn take_ticket(&self) -> LimitedClientTicket<'_> {
        LimitedClientTicket {
            client: CowNoClone::Borrowed(&self.fallback_client),
        }
    }

    /// Gets the 'fallback' reqwest client. This will bypass things like
    /// connection pooling and outgoing OTEL headers that
    /// we're going to implement on top of `TensorzeroHttpClient`.
    /// Please avoid calling this if at all possible - each callsite
    /// should have an explanation of why it's needed.
    pub fn dangerous_get_fallback_client(&self) -> &Client {
        &self.fallback_client.client
    }

    pub fn get<U: IntoUrl>(&self, url: U) -> TensorzeroRequestBuilder<'_> {
        let ticket = self.take_ticket();
        TensorzeroRequestBuilder {
            builder: ticket.client.client.get(url),
            ticket,
        }
    }

    pub fn post<U: IntoUrl>(&self, url: U) -> TensorzeroRequestBuilder<'_> {
        let ticket = self.take_ticket();
        TensorzeroRequestBuilder {
            builder: ticket.client.client.post(url),
            ticket,
        }
    }

    pub fn request<U: IntoUrl>(
        &self,
        method: reqwest::Method,
        url: U,
    ) -> TensorzeroRequestBuilder<'_> {
        let ticket = self.take_ticket();
        TensorzeroRequestBuilder {
            builder: ticket.client.client.request(method, url),
            ticket,
        }
    }
}

/// A wrapper type around `reqwest::RequestBuilder`.
/// The purpose of this type is to hold on to a `LimitedClientTicket`,
/// so that we can drop it after the request is sent.
///
/// If you need to call a new method on `RequestBuilder`, add a forwarding
/// implementation on `TensorzeroRequestBuilder`, following the pattern of
/// `TensorzeroRequestBuilder.header` and `TensorzeroRequestBuilder.body`.
pub struct TensorzeroRequestBuilder<'a> {
    builder: RequestBuilder,
    ticket: LimitedClientTicket<'a>,
}

/// A wrapper type around `reqwest_eventsource::EventSource`.
/// Like `TensorzeroRequestBuilder`, this type holds on to a `LimitedClientTicket`,
/// so that we can drop it when the stream is dropped (and hold on to it while
/// we're still polling messages from the stream).
#[pin_project]
pub struct TensorZeroEventSource {
    // We forward to this `EventSource` in our `Stream` impl
    #[pin]
    source: EventSource,
    ticket: LimitedClientTicket<'static>,
}

impl TensorZeroEventSource {
    pub fn close(&mut self) {
        self.source.close();
    }
}

impl Stream for TensorZeroEventSource {
    type Item = Result<Event, reqwest_eventsource::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        // Just forward to the underlying `EventSource`, without doing anything else.
        // The `TensorZeroEventSource` type only exists to hold on to a `LimitedClientTicket`
        // until the stream is dropped.
        self.project().source.poll_next(cx)
    }
}

impl<'a> TensorzeroRequestBuilder<'a> {
    pub fn header<K, V>(self, key: K, value: V) -> TensorzeroRequestBuilder<'a>
    where
        HeaderName: TryFrom<K>,
        <HeaderName as TryFrom<K>>::Error: Into<http::Error>,
        HeaderValue: TryFrom<V>,
        <HeaderValue as TryFrom<V>>::Error: Into<http::Error>,
    {
        Self {
            builder: self.builder.header(key, value),
            ticket: self.ticket,
        }
    }

    pub fn headers(self, headers: http::header::HeaderMap) -> TensorzeroRequestBuilder<'a> {
        Self {
            builder: self.builder.headers(headers),
            ticket: self.ticket,
        }
    }

    pub fn bearer_auth<T>(self, token: T) -> TensorzeroRequestBuilder<'a>
    where
        T: std::fmt::Display,
    {
        Self {
            builder: self.builder.bearer_auth(token),
            ticket: self.ticket,
        }
    }
    pub fn multipart(self, multipart: reqwest::multipart::Form) -> TensorzeroRequestBuilder<'a> {
        Self {
            builder: self.builder.multipart(multipart),
            ticket: self.ticket,
        }
    }

    pub fn json<T: Serialize + ?Sized>(self, json: &T) -> TensorzeroRequestBuilder<'a> {
        Self {
            builder: self.builder.json(json),
            ticket: self.ticket,
        }
    }

    pub fn body<T: Into<Body>>(self, body: T) -> TensorzeroRequestBuilder<'a> {
        Self {
            builder: self.builder.body(body),
            ticket: self.ticket,
        }
    }

    pub fn eventsource(self) -> Result<TensorZeroEventSource, CannotCloneRequestError> {
        Ok(TensorZeroEventSource {
            source: self.builder.eventsource()?,
            ticket: LimitedClientTicket {
                client: CowNoClone::Owned(LimitedClient {
                    concurrent_requests: self.ticket.client.concurrent_requests.clone(),
                    client: self.ticket.client.client.clone(),
                }),
            },
        })
    }

    // This method takes an owned `self`, so we'll drop `self.ticket` when this method
    // returns (after we've gotten a response)
    pub async fn send(self) -> Result<Response, reqwest::Error> {
        self.builder.send().await
    }

    pub async fn send_and_parse_json<T: DeserializeOwned>(
        self,
        provider_type: &str,
    ) -> Result<T, Error> {
        let (client, request) = self.builder.build_split();
        let request = request.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: None,
                message: format!("Error building request: {}", DisplayOrDebugGateway::new(e)),
                provider_type: provider_type.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        let url = request.url().clone();
        let raw_body = request
            .body()
            .and_then(|b| b.as_bytes().map(|b| String::from_utf8_lossy(b).to_string()));
        let response = client.execute(request).await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                provider_type: provider_type.to_string(),
                raw_request: raw_body.clone(),
                raw_response: None,
            })
        })?;

        let status_code = response.status();

        let raw_response = response.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                provider_type: provider_type.to_string(),
                raw_request: raw_body.clone(),
                raw_response: None,
            })
        })?;

        if !status_code.is_success() {
            return Err(Error::new(ErrorDetails::InferenceClient {
                status_code: Some(status_code),
                message: format!("Non-successful status code for url `{url}`",),
                provider_type: provider_type.to_string(),
                raw_request: raw_body.clone(),
                raw_response: Some(raw_response.clone()),
            }));
        }

        let res: T = serde_json::from_str(&raw_response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error parsing JSON response: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                raw_request: raw_body.clone(),
                raw_response: Some(raw_response.clone()),
                provider_type: provider_type.to_string(),
            })
        })?;
        Ok(res)
    }
}

// This is set high enough that it should never be hit for a normal model response.
// In the future, we may want to allow overriding this at the model provider level.
const DEFAULT_HTTP_CLIENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5 * 60);

fn build_client() -> Result<Client, Error> {
    let mut http_client_builder = Client::builder().timeout(DEFAULT_HTTP_CLIENT_TIMEOUT);

    if cfg!(feature = "e2e_tests") {
        if let Ok(proxy_url) = std::env::var("TENSORZERO_E2E_PROXY") {
            tracing::info!("Using proxy URL from TENSORZERO_E2E_PROXY: {proxy_url}");
            http_client_builder = http_client_builder
                .proxy(
                    Proxy::all(proxy_url)
                        .map_err(|e| {
                            Error::new(ErrorDetails::AppState {
                                message: format!("Invalid proxy URL: {e}"),
                            })
                        })?
                        .no_proxy(NoProxy::from_string("localhost,127.0.0.1,minio")),
                )
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
