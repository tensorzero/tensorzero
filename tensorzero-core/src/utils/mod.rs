use std::future::Future;

use tracing::Span;
use tracing_futures::Instrument;

use crate::error::Error;
use crate::error::ErrorDetails;
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;

pub mod gateway;
pub mod retries;
#[cfg(any(test, feature = "e2e_tests"))]
pub mod testing;
pub mod uuid;

/// A helper function that wraps a future that might have unbounded recursion.
/// Note that this is *not* the same thing as boxing a future - boxing does not
/// prevent us from having an arbitrarily deep call stack.
/// Consider the function:
/// ```rust
/// fn call_self() -> impl Future<Output = ()> + Send  {
///     async move {
///         call_self().boxed().await
///     }
/// }
/// ```
///
/// Each recursive call creates a new stack frame, despite the `boxed()` call.
///
/// The `unbounded_recursion_wrapper` creates a new tokio task, which will
/// give us a new call stack (via tokio)
pub async fn unbounded_recursion_wrapper<R: Send + 'static>(
    fut: impl Future<Output = Result<R, Error>> + Send + 'static,
) -> Result<R, Error> {
    // We await this immediately
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(fut.instrument(Span::current()))
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to join task: {e:?}. {IMPOSSIBLE_ERROR_MESSAGE}"),
            })
        })?
}

/// A helper function that emits a deprecation warning.
/// All deprecation warnings should be emitted using this function,
/// so that we can detect unintentional use of deprecated behavior in our e2e tests.
pub fn deprecation_warning(message: &str) {
    tracing::warn!("Deprecation warning: {message}");
}

/// Spawns a background task that does not interact with gateway shutdown -
/// it will be cancelled at some arbitrary `.await` point when the gateway shuts down.
/// This method should only be used for 'best-effort' background tasks (e.g. cache writes,
/// Howdy, etc) where we don't care if they finish or not (so they don't need to block
/// the gateway shutdown).
pub fn spawn_ignoring_shutdown(fut: impl Future<Output = ()> + Send + 'static) {
    // The `spawn_ignoring_shutdown` function is an explicit way of invoking `tokio::spawn`
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(fut);
}
