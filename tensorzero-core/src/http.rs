use chrono::Duration;
use once_cell::sync::OnceCell;
use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    task::{Context, Poll},
};
use tracing::Span;
use tracing_futures::Instrument;

use futures::Stream;
use http::{HeaderName, HeaderValue};
use pin_project::pin_project;
use reqwest::{Body, Response};
use reqwest::{Client, IntoUrl, NoProxy, Proxy, RequestBuilder};
use reqwest_eventsource::{CannotCloneRequestError, Event, EventSource, RequestBuilderExt};
use serde::{de::DeserializeOwned, Serialize};

use crate::endpoints::status::TENSORZERO_VERSION;
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::{
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    model_table::CowNoClone,
};

// A wrapper around `reqwest::Client` that tracks the number of outstanding concurrent requests.
// We store an array of these in `TensorzeroHttpClient`, and pick the first one that has
// a `concurrent_requests` count that's below `CONCURRENCY_LIMIT`.
#[derive(Debug)]
struct LimitedClient {
    // This needs to be an `Arc` so that we can share the counter
    // when we create a `TensorzeroEventSource` stream wrapper
    // (we need to decrement it when the stream is dropped)
    // Be careful - this should only ever be cloned from `LimitedClientTicket.into_owned`,
    // where we also set `should_decrement` to `false`
    concurrent_requests: Arc<AtomicU8>,
    client: Client,
}

struct LimitedClientTicket<'a> {
    client: CowNoClone<'a, LimitedClient>,
    // If `true`, decrement `client.concurrent_requests` when we drop `self
    // This will be `false` when using the fallback client (which has no limit),
    // or when we transfer ownership in 'LimitedClientTicket.into_owned'
    should_decrement: bool,
}

impl LimitedClientTicket<'_> {
    fn into_owned(mut self) -> LimitedClientTicket<'static> {
        // Set 'self.should_decrement' to `false`, so that we don't decrement the counter when we drop `self`
        // The previous value of 'self.should_decrement' is used in the new `LimitedClientTicket`,
        // which takes responsibility for decrementing the counter
        let should_decrement = std::mem::replace(&mut self.should_decrement, false);
        LimitedClientTicket {
            client: CowNoClone::Owned(LimitedClient {
                concurrent_requests: self.client.concurrent_requests.clone(),
                client: self.client.client.clone(),
            }),
            should_decrement,
        }
    }
}

impl Drop for LimitedClientTicket<'_> {
    fn drop(&mut self) {
        if self.should_decrement {
            self.client
                .concurrent_requests
                .fetch_sub(1, Ordering::SeqCst);
        }
    }
}

const MAX_NUM_CLIENTS: usize = 1024;
// This is set to a common value for the HTTP `max_concurrent_streams` setting sent by the server
// (OpenAI, Anthropic, and GCP Vertex all use this value).
// This works around a limitation in reqwest/hyper - when a HTTP2 connection is negotiated,
// request will only ever open at most one TCP connection to the remote (host, port) pair,.
// If the total number of concurrency requests exceeds the HTTP2 `max_concurrent_streams` limit,
// hyper will delay submitting new requests until a request completes.
//
// This is a significant limitation in our case, since what we actually care about is
// the higher-level rate limit imposed by the model provider, which can be much higher.
// To avoid artifically limiting our concurrency, we spread requests across multiple `reqwest::Client`
// instances to allow opening multiple HTTP2 TCP connections to the remote server
//
// This unfortunately penalizes some other use cases:
// * Multiple HTTP1 connections
// * Many different requests (whether HTTP1 or HTTP2) to many different hosts
//
// If we have high concurrency in one of the above cases, we'll end up using more than one
// `reqwest::Client` instance, even though we wouldn't have hit a `max_concurrent_streams` limit.
// This will make the internal `reqwest` connection pool less effective, as we'll have more
// duplicate connections than if we had used a single `reqwest::Client` instance.
//
// Users have reported hitting the `max_concurrent_streams`, so we're prioritizing the ability
// to scale to high QPS. Ideally, `reqwest/hyper` would natively support opening multiple TCP
// connections for HTTP2, allowing us to remove our manual connection pooling.
const CONCURRENCY_LIMIT: u8 = 100;

/// A wrapper for `reqwest::Client` that adds extra features:
/// * Improved connection pooling support for HTTP/2
#[derive(Debug, Clone)]
pub struct TensorzeroHttpClient {
    // A 'waterfall' of clients for connecting pooling.
    // When we try to obtain a client with `take_ticket`, we iterate over
    // the list until we find a client with an concurrent request count that's
    // below `concurrency_limit`. This allows requests to share a `reqwest::Client`
    // when possible (giving reqwest the chance to re-use TCP connections), while
    // distributing requests across multiple clients to prevent hanging due to the
    // HTTP2 concurrent stream limit.

    // This implementation is deliberately simplistic:
    // * We don't use the host/port as a key - all requests go through the same array
    // * We don't attempt to detect if we're going to use HTTP2 (reqwest only exposes
    //   this information after a request has completed)
    // * The CONCURRENCY_LIMIT is static, as reqwest doesn't expose the HTTP2
    //   `max_concurrent_streams` limit sent by the server.
    //
    // We're optimizing for the case where a large number of concurrent requests are made
    // to the same model provider.
    //
    // Initializing 1024 `reqwest::Client`s can take several seconds. To avoid making
    // embedding client construction very slow, we lazily construct them the first time
    // our concurrent requests breaches a threshold (i.e. the first time we try
    // to access a slot in the array). Once a `LimitedClient` is constructed,
    // it stays alive for the lifetime of the `TensorzeroHttpClient` instance.
    clients: Arc<[OnceCell<LimitedClient>]>,
    fallback_client: Arc<LimitedClient>,
    global_outbound_http_timeout: Duration,
}

#[cfg(any(test, feature = "e2e_tests", feature = "pyo3"))]
impl Default for TensorzeroHttpClient {
    fn default() -> Self {
        // This is only available in tests and e2e tests, so it's fine to unwrap here
        #[expect(clippy::unwrap_used)]
        Self::new_testing().unwrap()
    }
}

impl TensorzeroHttpClient {
    #[cfg(any(test, feature = "e2e_tests", feature = "pyo3"))]
    pub fn new_testing() -> Result<Self, Error> {
        Self::new(DEFAULT_HTTP_CLIENT_TIMEOUT)
    }
    pub fn new(global_outbound_http_timeout: Duration) -> Result<Self, Error> {
        let clients = (0..MAX_NUM_CLIENTS)
            .map(|_| OnceCell::new())
            .collect::<Vec<_>>();
        let client = Self {
            clients: clients.into(),
            fallback_client: Arc::new(LimitedClient {
                concurrent_requests: Arc::new(AtomicU8::new(0)),
                client: build_client(global_outbound_http_timeout)?,
            }),
            global_outbound_http_timeout,
        };
        // Eagerly initialize the first `OnceCell` in the array
        client.take_ticket();
        Ok(client)
    }

    fn take_ticket(&self) -> LimitedClientTicket<'_> {
        for client_cell in self.clients.iter() {
            let client = match client_cell.get_or_try_init(|| {
                Ok::<_, Error>(LimitedClient {
                    concurrent_requests: Arc::new(AtomicU8::new(0)),
                    client: build_client(self.global_outbound_http_timeout)?,
                })
            }) {
                Ok(client) => client,
                Err(_) => {
                    // The error was already logged - continue on and try to access
                    // the next `OnceCell` in the array. If all of them fail on this
                    // pass through the loop, we'll end up using the fallback client.
                    continue;
                }
            };

            // Attempt to increment the value, failing if we're at `CONCURRENCY_LIMIT`
            let val = client.concurrent_requests.fetch_update(
                Ordering::SeqCst,
                Ordering::SeqCst,
                |val| {
                    if val < CONCURRENCY_LIMIT {
                        Some(val + 1)
                    } else {
                        None
                    }
                },
            );
            // If we successfully incremented the value, then use this client.
            if val.is_ok() {
                return LimitedClientTicket {
                    client: CowNoClone::Borrowed(client),
                    should_decrement: true,
                };
            }
            // Otherwise, continue looping through the array
        }
        // If we somehow have 'CONCURRENCY_LIMIT * NUM_CLIENTS' outstanding requests,
        // then log an error, and use the fallback client.
        // When this happens, we gracefully degrade to our behavior before `TensorzeroHttpClient`
        // was introduced (sharing a single `reqwest::Client` instance, which will limit the
        // concurrency for HTTP2 requests to the same host).
        Error::new(ErrorDetails::InternalError {
            message: format!("No available HTTP clients. {IMPOSSIBLE_ERROR_MESSAGE}"),
        });
        LimitedClientTicket {
            client: CowNoClone::Borrowed(&self.fallback_client),
            // The fallback client has no limit, so don't decrement its counter when we drop it
            should_decrement: false,
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

    pub fn patch<U: IntoUrl>(&self, url: U) -> TensorzeroRequestBuilder<'_> {
        let ticket = self.take_ticket();
        TensorzeroRequestBuilder {
            builder: ticket.client.client.patch(url),
            ticket,
        }
    }

    pub fn delete<U: IntoUrl>(&self, url: U) -> TensorzeroRequestBuilder<'_> {
        let ticket = self.take_ticket();
        TensorzeroRequestBuilder {
            builder: ticket.client.client.delete(url),
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
    span: Span,
}

impl TensorZeroEventSource {
    pub fn close(&mut self) {
        self.source.close();
    }
}

impl Stream for TensorZeroEventSource {
    type Item = Result<Event, reqwest_eventsource::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let _guard = this.span.enter();
        // Just forward to the underlying `EventSource`, without doing anything else.
        // The `TensorZeroEventSource` type only exists to hold on to a `LimitedClientTicket`
        // until the stream is dropped.
        this.source.poll_next(cx)
    }
}

// Workaround for https://github.com/hyperium/h2/issues/763
// The 'h2' crate creates a long-lived span for outgoing HTTP connections.
// Due to connection pooling, these spans can live for a long time -
// in particular, they can live across multiple TensorZero `POST /inference` requests.
//
// A `tracing` span always lives as long as its longest-lived descendant span:
// https://docs.rs/tracing/latest/tracing/span/index.html#span-relationships
// As a result, the h2 connection span can cause our spans to live for an extremely long time,
// delaying the reporting of spans to OpenTelemetry.
//
// The h2 span is a trace-level span, so it would normally get disabled entirely
// by our span filters. Unfortunately, our workaround for a tracing bug
// intentionally blocks this type of logic (`Interest::never()` / `Interest::always()`)
// - see apply_filter_fixing_tracing_bug
//
// As a result, we need to prevent the h2 span from ending up as a descendant of our spans.
// When we call into `reqwest`, we enter this special span, which we override to be a root span
// (no parent). This prevents the h2 span from getting associated with any of our OTEL spans.
//
// If https://github.com/hyperium/h2/issues/713 is ever fixed, we should disable tracing
// within the `h2` crate itself.
fn tensorzero_h2_workaround_span() -> tracing::Span {
    tracing::trace_span!(parent: None, "__tensorzero_h2_span_hack__")
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

    pub fn timeout(self, timeout: std::time::Duration) -> TensorzeroRequestBuilder<'a> {
        Self {
            builder: self.builder.timeout(timeout),
            ticket: self.ticket,
        }
    }

    pub fn query<T: Serialize + ?Sized>(self, query: &T) -> TensorzeroRequestBuilder<'a> {
        Self {
            builder: self.builder.query(query),
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
            ticket: self.ticket.into_owned(),
            span: tensorzero_h2_workaround_span(),
        })
    }

    // This method takes an owned `self`, so we'll drop `self.ticket` when this method
    // returns (after we've gotten a response)
    pub async fn send(self) -> Result<Response, reqwest::Error> {
        self.builder
            .send()
            .instrument(tensorzero_h2_workaround_span())
            .await
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
        let response = client
            .execute(request)
            .instrument(tensorzero_h2_workaround_span())
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error sending request: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: provider_type.to_string(),
                    raw_request: raw_body.clone(),
                    raw_response: None,
                })
            })?;

        let status_code = response.status();

        let raw_response = response
            .text()
            .instrument(tensorzero_h2_workaround_span())
            .await
            .map_err(|e| {
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
// Users can customize it via `gateway.global_outbound_http_timeout_ms` in the config file.
pub const DEFAULT_HTTP_CLIENT_TIMEOUT: Duration = Duration::seconds(5 * 60);

fn build_client(global_outbound_http_timeout: Duration) -> Result<Client, Error> {
    let mut http_client_builder = Client::builder()
        .timeout(global_outbound_http_timeout.to_std().map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to convert Duration to std::time::Duration: {e}"),
            })
        })?)
        .user_agent(format!("TensorZero/{TENSORZERO_VERSION}"));

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
                        .no_proxy(NoProxy::from_string(
                            "localhost,127.0.0.1,minio,mock-inference-provider,gateway,provider-proxy,clickhouse",
                        )),
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

#[cfg(test)]
mod tests {
    use std::{
        future::IntoFuture,
        net::SocketAddr,
        sync::{
            atomic::{AtomicU8, Ordering},
            Arc,
        },
    };

    use axum::{
        extract::Request,
        response::{sse::Event, Sse},
        routing::get,
        Router,
    };
    use futures::StreamExt;
    use reqwest::Proxy;
    use tokio::task::{JoinHandle, JoinSet};

    use crate::http::{LimitedClient, TensorZeroEventSource, CONCURRENCY_LIMIT};

    async fn start_target_server() -> (SocketAddr, JoinHandle<Result<(), std::io::Error>>) {
        let app = Router::new()
            .route(
                "/hello",
                get(|_req: Request| async {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    http::Response::new("Hello".to_string())
                }),
            )
            .route(
                "/hello-stream",
                get(|_req: Request| async {
                    Sse::new(futures::stream::iter(vec![
                        Ok::<_, String>(Event::default().data("Hello")),
                        Ok(Event::default().data("[DONE]")),
                    ]))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
        #[expect(clippy::disallowed_methods)]
        let handle = tokio::spawn(axum::serve(listener, app).into_future());
        (addr, handle)
    }

    async fn process_stream(stream: &mut TensorZeroEventSource) {
        while let Some(event) = stream.next().await {
            match event {
                Ok(_) => {}
                Err(e) => {
                    if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                        break;
                    }
                    panic!("Error in streaming response: {e:?}");
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_requests_helper() {
        let (addr, _handle) = start_target_server().await;
        let mut client = super::TensorzeroHttpClient::new_testing().unwrap();

        // Send one non-stream request, and check that it didn't require using a new client
        let response = client
            .get(format!("http://{addr}/hello"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);

        for (i, client_cell) in client.clients.iter().enumerate() {
            if i == 0 {
                assert_eq!(
                    client_cell
                        .get()
                        .unwrap()
                        .concurrent_requests
                        .load(Ordering::SeqCst),
                    0
                );
            } else {
                assert!(
                    client_cell.get().is_none(),
                    "Client {i} should not be initialized"
                );
            }
        }

        // Send one non-stream request, and check that it didn't require using a new client
        let mut event_source = client
            .get(format!("http://{addr}/hello-stream"))
            .eventsource()
            .unwrap();
        process_stream(&mut event_source).await;
        drop(event_source);

        for (i, client_cell) in client.clients.iter().enumerate() {
            if i == 0 {
                assert_eq!(
                    client_cell
                        .get()
                        .unwrap()
                        .concurrent_requests
                        .load(Ordering::SeqCst),
                    0
                );
            } else {
                assert!(
                    client_cell.get().is_none(),
                    "Client {i} should not be initialized"
                );
            }
        }

        let mut tasks = JoinSet::new();
        let num_tasks: usize = 1000;
        for i in 0..num_tasks {
            let client = client.clone();
            if i % 2 == 0 {
                tasks.spawn(async move {
                    client
                        .get(format!("http://{addr}/hello"))
                        .send()
                        .await
                        .unwrap();
                });
            } else {
                tasks.spawn(async move {
                    let mut stream = client
                        .get(format!("http://{addr}/hello-stream"))
                        .eventsource()
                        .unwrap();
                    process_stream(&mut stream).await;
                });
            }
        }
        tasks.join_all().await;
        // We should have used at least one new client
        assert!(client.clients[1].get().is_some());
        // At most `num_tasks/CONCURRENCY_LIMIT` clients should have been used
        // (the maximum is achieved if all tasks happen to run concurrently)
        assert_eq!(num_tasks % (CONCURRENCY_LIMIT as usize), 0);
        let num_initialized_clients = client.clients.iter().filter(|c| c.get().is_some()).count();
        assert!(num_initialized_clients <= (num_tasks / (CONCURRENCY_LIMIT as usize) ), "Too many initialized clients - found {num_initialized_clients} but expected at most {}", num_tasks / (CONCURRENCY_LIMIT as usize));
        for client_cell in client.clients.iter() {
            if let Some(client) = client_cell.get() {
                assert_eq!(client.concurrent_requests.load(Ordering::SeqCst), 0);
            }
        }

        // Clear out the clients, and verify that making a new request only uses the first one (and does not initialize any new clients)
        let clients_mut = Arc::get_mut(&mut client.clients).unwrap();
        for client_cell in clients_mut.iter_mut() {
            client_cell.take();
        }

        // Send one request, and check that it didn't require using a new client
        let response = client
            .get(format!("http://{addr}/hello"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        for (i, client_cell) in client.clients.iter().enumerate() {
            assert_eq!(
                client_cell.get().is_some(),
                i == 0,
                "Wrong initialization state for client {i}"
            );
            if let Some(client) = client_cell.get() {
                assert_eq!(client.concurrent_requests.load(Ordering::SeqCst), 0);
            }
        }

        // Send one stream request, and check that it didn't require using a new client
        let mut stream = client
            .get(format!("http://{addr}/hello-stream"))
            .eventsource()
            .unwrap();
        process_stream(&mut stream).await;
        drop(stream);
        for (i, client_cell) in client.clients.iter().enumerate() {
            assert_eq!(
                client_cell.get().is_some(),
                i == 0,
                "Wrong initialization state for client {i}"
            );
            if let Some(client) = client_cell.get() {
                assert_eq!(client.concurrent_requests.load(Ordering::SeqCst), 0);
            }
        }

        let clients_mut = Arc::get_mut(&mut client.clients).unwrap();
        // Store invalid clients in the array, to verify that we don't try to use them when our concurrency
        // level is below `CONCURRENCY_LIMIT`
        for client_cell in clients_mut.iter_mut().skip(1) {
            client_cell.take();
            client_cell
                .set(LimitedClient {
                    concurrent_requests: Arc::new(AtomicU8::new(0)),
                    client: reqwest::Client::builder()
                        .proxy(Proxy::all("http://tensorzero.invalid:8080").unwrap())
                        .build()
                        .unwrap(),
                })
                .unwrap();
        }

        // We only spawn `CONCURRENCY_LIMIT` tasks, so we should never need to go beyond the first array entry
        let mut new_tasks = JoinSet::new();
        for i in 0..CONCURRENCY_LIMIT as usize {
            let client = client.clone();
            if i % 2 == 0 {
                new_tasks.spawn(async move {
                    client
                        .get(format!("http://{addr}/hello"))
                        .send()
                        .await
                        .unwrap();
                });
            } else {
                new_tasks.spawn(async move {
                    let mut stream = client
                        .get(format!("http://{addr}/hello-stream"))
                        .eventsource()
                        .unwrap();
                    process_stream(&mut stream).await;
                });
            }
        }

        new_tasks.join_all().await;
        for client_cell in client.clients.iter() {
            if let Some(client) = client_cell.get() {
                assert_eq!(client.concurrent_requests.load(Ordering::SeqCst), 0);
            }
        }

        // Spawn a streaming request, and verify that it holds a ticket until the stream is dropped
        let mut stream = client
            .get(format!("http://{addr}/hello-stream"))
            .eventsource()
            .unwrap();

        process_stream(&mut stream).await;
        assert_eq!(
            client.clients[0]
                .get()
                .unwrap()
                .concurrent_requests
                .load(Ordering::SeqCst),
            1
        );
        drop(stream);
        assert_eq!(
            client.clients[0]
                .get()
                .unwrap()
                .concurrent_requests
                .load(Ordering::SeqCst),
            0
        );

        for client_cell in client.clients.iter() {
            assert!(client_cell.get().is_some(), "Client should be initialized");
        }
    }
}
