use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    body::Body,
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::Response,
};
use dashmap::DashMap;
use http::Method;
use http_body::{Frame, SizeHint};
use tracing::Level;
use tracing_futures::Instrument;
use uuid::Uuid;

use tensorzero_overhead::{ConnectionDropGuard, TENSORZERO_TRACK_OVERHEAD_ATTRIBUTE_NAME};

pub use tensorzero_overhead::HttpMetricData;

/// A wrapper for an `axum::Body` that holds on to a `ConnectionDropGuard`
/// We explicitly mark the guard as finished if the underlying body returns `Ok(None)`,
/// from `poll_frame`, or `is_end_stream` returns `true`.
///
/// If this is dropped without either of those conditions being reached (which indicates
/// that the client closed the connection before the server finished sending the response),
/// then `ConnectionDropGuard` will log a warning.
///
/// This is used to log a warning when the client closes a streaming response early.
/// It's insufficient to just use a `ConnectionDropGuard` in a middleware function, because,
/// because the middleware itself will return a (streaming) response body, which Axum then
/// streams to the client outside of our middleware/handler code.
/// Instead, we must instrument the body itself, to detect whether the body is polled to completion
/// or dropped early.
#[pin_project::pin_project]
pub struct GuardBodyWrapper {
    #[pin]
    inner: Body,
    guard: ConnectionDropGuard,
}

#[warn(clippy::missing_trait_methods)]
impl http_body::Body for GuardBodyWrapper {
    type Data = <Body as http_body::Body>::Data;
    type Error = <Body as http_body::Body>::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.project();
        let res = this.inner.poll_frame(cx);
        match res {
            Poll::Ready(None) => {
                this.guard.mark_finished();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(_))) | Poll::Ready(Some(Err(_))) | Poll::Pending => res,
        }
    }

    fn is_end_stream(&self) -> bool {
        // 'poll_frame' might not be called again if this returns 'true', so mark the guard as finished
        let ended = self.inner.is_end_stream();
        if ended {
            self.guard.mark_finished();
        }
        ended
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// A clonable handle that tracks the number of in-flight requests for each route.
#[derive(Clone)]
pub struct InFlightRequestsData {
    count_per_route: Arc<DashMap<String, Arc<AtomicU32>>>,
}

impl InFlightRequestsData {
    #[expect(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            count_per_route: Arc::new(Default::default()),
        }
    }

    /// Gets the current in-flight requests counts for each route
    /// Note that routes that have never been requested will not be included in the iterator.
    pub fn current_counts_by_route(&self) -> impl Iterator<Item = (String, u32)> {
        self.count_per_route
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
    }
}

// An Axum middleware that logs request processing events
// * 'started processing request' when we begin processing a request
// * 'finished processing request' when we we *completely* finish a request.
//    For SSE streams, this is logged when we finish sending the entire stream,
//    not when the response status code is initially sent
// * 'Client closed the connection before the response was sent' if the connection is closed early.
pub async fn request_logging_middleware(
    state: State<InFlightRequestsData>,
    request: Request,
    next: Next,
) -> Response<GuardBodyWrapper> {
    let start_time = Instant::now();

    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| format!("{} {}", request.method(), p.as_str()).to_string())
        .unwrap_or_else(|| "<unknown_route>".to_string());
    let count_per_route = state
        .count_per_route
        .entry(route)
        .or_insert_with(|| Arc::new(Default::default()))
        .clone();

    count_per_route.fetch_add(1, Ordering::Relaxed);

    // Create a separate span for latency tracing, using a custom 'target' that will
    // get filtered out when we log to console/otel
    // This prevents the `tensorzero.overhead.*` span attributes from being visible to users
    // in the logs/OTEL
    let latency_span = if request.method() == Method::POST && request.uri() == "/inference" {
        Some(tracing::span!(
            target: "tensorzero.overhead",
            Level::DEBUG,
            "request_latency_tracking",
            { TENSORZERO_TRACK_OVERHEAD_ATTRIBUTE_NAME } = true,
        ))
    } else {
        None
    };

    // Generate a random ID so that we can associate log lines with this request
    let request_id = Uuid::now_v7();

    let span = if let Some(latency_span) = &latency_span {
        tracing::info_span!(
            target: "gateway",
            parent: latency_span,
            "request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = %request_id,
            x_amzn_trace_id = tracing::field::Empty,
        )
    } else {
        tracing::info_span!(
            target: "gateway",
            "request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = %request_id,
            x_amzn_trace_id = tracing::field::Empty,
        )
    };
    if let Some(x_amzn_trace_id) = request
        .headers()
        .get("x-amzn-trace-id")
        .and_then(|h| h.to_str().ok())
    {
        span.record("x_amzn_trace_id", x_amzn_trace_id);
    }
    span.in_scope(|| {
        tracing::debug!("started processing request");
    });

    // Axum runs GET handlers when a HEAD requests is made, but drops the body.
    // To avoid false positives, we never log a warning for HEAD requests.
    let is_finished = matches!(request.method(), &Method::HEAD);
    let mut guard =
        ConnectionDropGuard::new(count_per_route, latency_span, span.clone(), start_time);
    let mut response = next.run(request).instrument(span).await;
    guard.set_request_logging_data(response.extensions_mut().remove::<HttpMetricData>());
    guard.set_status(response.status());
    if is_finished {
        guard.mark_finished();
    }
    response.map(|body| GuardBodyWrapper { inner: body, guard })
}
