use once_cell::sync::OnceCell;
use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

use futures::Stream;
use http::{HeaderName, HeaderValue};
use pin_project::pin_project;
use reqwest::{Body, Response};
use reqwest::{Client, IntoUrl, NoProxy, Proxy, RequestBuilder};
use reqwest_eventsource::{CannotCloneRequestError, Event, EventSource, RequestBuilderExt};
use serde::{de::DeserializeOwned, Serialize};

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
    concurrent_requests: Arc<AtomicU8>,
    client: Client,
}

struct LimitedClientTicket<'a> {
    client: CowNoClone<'a, LimitedClient>,
}

impl Drop for LimitedClientTicket<'_> {
    fn drop(&mut self) {
        self.client
            .concurrent_requests
            .fetch_sub(1, Ordering::SeqCst);
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
#[derive(Clone)]
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
}

impl TensorzeroHttpClient {
    pub fn new() -> Result<Self, Error> {
        let clients = (0..MAX_NUM_CLIENTS)
            .map(|_| OnceCell::new())
            .collect::<Vec<_>>();
        let client = Self {
            clients: clients.into(),
            fallback_client: Arc::new(LimitedClient {
                concurrent_requests: Arc::new(AtomicU8::new(0)),
                client: build_client()?,
            }),
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
                    client: build_client()?,
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

    use axum::{extract::Request, routing::get, Router};
    use reqwest::Proxy;
    use tokio::task::{JoinHandle, JoinSet};

    use crate::http::{LimitedClient, CONCURRENCY_LIMIT};

    async fn start_target_server() -> (SocketAddr, JoinHandle<Result<(), std::io::Error>>) {
        let app = Router::new().route(
            "/hello",
            get(|_req: Request| async {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                http::Response::new("Hello".to_string())
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(axum::serve(listener, app).into_future());
        (addr, handle)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_requests() {
        let (addr, _handle) = start_target_server().await;
        let mut client = super::TensorzeroHttpClient::new().unwrap();
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
        }
        let mut tasks = JoinSet::new();
        let num_tasks: usize = 1000;
        for _ in 0..num_tasks {
            let client = client.clone();
            tasks.spawn(async move {
                client
                    .get(format!("http://{addr}/hello"))
                    .send()
                    .await
                    .unwrap()
            });
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
        for _ in 0..CONCURRENCY_LIMIT as usize {
            let client = client.clone();
            new_tasks.spawn(async move {
                client
                    .get(format!("http://{addr}/hello"))
                    .send()
                    .await
                    .unwrap()
            });
        }

        new_tasks.join_all().await;
        for client_cell in client.clients.iter() {
            if let Some(client) = client_cell.get() {
                assert_eq!(client.concurrent_requests.load(Ordering::SeqCst), 0);
            }
        }
    }
}
