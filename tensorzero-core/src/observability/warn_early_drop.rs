use std::{
    cell::Cell,
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use http::Method;
use http_body::{Frame, SizeHint};
use tracing::Span;

/// A drop guard that logs a message on drop if `start_time` is set.
struct ConnectionDropGuard {
    span: Span,
    start_time: Instant,
    finished: Cell<bool>,
}

impl ConnectionDropGuard {
    // Mark the guard as explicitly finished by the server.
    // This suppresses the warning that we would otherwise log in the `Drop` impl.
    // Note that we call this method even if the server produces an error - the purpose
    // of this method is to detect early drops, when the server didn't produce a response of any kind.
    fn mark_finished(&self) {
        self.finished.set(true);
    }
}

impl Drop for ConnectionDropGuard {
    fn drop(&mut self) {
        // If `start_time` is set, then `ConnectionDropGuard` was dropped before the response was sent
        // We log a warning and the latency of the request.
        if !self.finished.get() {
            let latency = format!("{} ms", self.start_time.elapsed().as_millis());
            let _guard = self.span.enter();
            tracing::warn!(
                %latency,
                "Client closed the connection before the response was sent",
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

/// An Axum middleware that logs a warning if it's dropped early (which will occur if the client closes the connection early)
pub async fn warn_on_early_connection_drop(
    request: Request,
    next: Next,
) -> Response<GuardBodyWrapper> {
    let start_time = Instant::now();
    // Axum runs GET handlers when a HEAD requests is made, but drops the body.
    // To avoid false positives, we never log a warning for HEAD requests.
    let is_finished = matches!(request.method(), &Method::HEAD);
    let guard = ConnectionDropGuard {
        // Save the current span, since we have no idea what span we'll be in when `Drop` runs.
        span: Span::current(),
        start_time,
        finished: Cell::new(is_finished),
    };
    let response = next.run(request).await;
    response.map(|body| GuardBodyWrapper { inner: body, guard })
}
