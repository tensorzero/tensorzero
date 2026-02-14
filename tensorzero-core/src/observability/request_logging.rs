use std::{
    cell::Cell,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::Response,
};
use dashmap::DashMap;
use http::{Method, StatusCode};
use http_body::{Frame, SizeHint};
use metrics::Label;
use tracing::{Level, Span};
use tracing_futures::Instrument;
use uuid::Uuid;

use crate::observability::overhead_timing::{
    OverheadSpanExt, TENSORZERO_TRACK_OVERHEAD_ATTRIBUTE_NAME,
};

/// A drop guard that logs a message on drop if `start_time` is set.
struct ConnectionDropGuard {
    // The counter to decrement when the request is finished
    count_per_route: Arc<AtomicU32>,
    latency_span: Option<Span>,
    request_logging_data: Option<HttpMetricData>,
    span: Span,
    start_time: Instant,
    finished_with_latency: Cell<Option<Duration>>,
    status: Option<StatusCode>,
}

impl ConnectionDropGuard {
    // Mark the guard as explicitly finished by the server.
    // This suppresses the warning that we would otherwise log in the `Drop` impl.
    // Note that we call this method even if the server produces an error - the purpose
    // of this method is to detect early drops, when the server didn't produce a response of any kind.
    fn mark_finished(&self) {
        // Calculate the elapsed time when we've finished sending the response to
        // the client - this is the latency that we want to log to users,
        // and use for computing the `tensorzero_inference_latency_overhead_seconds` metric
        self.finished_with_latency
            .set(Some(self.start_time.elapsed()));
    }
}

impl Drop for ConnectionDropGuard {
    fn drop(&mut self) {
        let _guard = self.span.enter();
        self.count_per_route.fetch_sub(1, Ordering::Relaxed);
        let latency_duration = if let Some(finished_with_latency) = self.finished_with_latency.get()
        {
            if let Some(latency_span) = &self.latency_span {
                // Only update the 'overhead' metric when the request finished, so that we
                // can accurately subtract off the time taken for 'external' spans.
                latency_span.set_inference_latency_and_record(
                    finished_with_latency,
                    self.request_logging_data
                        .take()
                        .map(|data| data.extra_overhead_labels),
                );
            }
            finished_with_latency
        } else {
            // If we didn't explicitly mark the request as 'finished' (due to the connection
            // getting dropped early), then use the current time to compute the latency.
            self.start_time.elapsed()
        };

        let latency = format!("{} ms", latency_duration.as_millis());

        // If we did not explicitly set 'finished', then `ConnectionDropGuard` was dropped before the response was sent
        // We log a warning and the latency of the request.
        if self.finished_with_latency.get().is_none() {
            tracing::warn!(
                %latency,
                "Client closed the connection before the response was sent",
            );
        }

        // We might have a status code even if the client closed the connection early
        // (e.g. if we were sending an SSE stream)
        if let Some(status) = self.status {
            tracing::debug!(
                %latency,
                status = i32::from(status.as_u16()),
                success = status.is_success(),
                "finished processing request"
            );
        }
    }
}

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

/// A *response* extension used to pass data from a route handler to `request_logging_middleware`
/// See the `inference` route handler for an example of how to use this.
#[derive(Clone)]
pub struct HttpMetricData {
    /// Extra labels to add to the `tensorzero_inference_latency_overhead_seconds` metric.
    /// We currently use this to attach `function_name`, `variant_name`, and `model_name` labels
    /// when recording the overhead of `/inference` requests
    pub extra_overhead_labels: Vec<Label>,
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
    let mut guard = ConnectionDropGuard {
        latency_span,
        request_logging_data: None,
        span: span.clone(),
        start_time,
        finished_with_latency: Cell::new(None),
        count_per_route,
        status: None,
    };
    let mut response = next.run(request).instrument(span).await;
    guard.request_logging_data = response.extensions_mut().remove::<HttpMetricData>();
    guard.status = Some(response.status());
    if is_finished {
        guard.mark_finished();
    }
    response.map(|body| GuardBodyWrapper { inner: body, guard })
}
