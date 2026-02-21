use std::fmt::Debug;
use std::future::Future;
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_millis(500);
const POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// Polls a query until the result matches a predicate.
///
/// Returns the first `Ok` result that satisfies `predicate`. Retries on both errors and
/// non-matching results. Panics if the timeout (default 5s) is reached.
pub async fn poll_for_result<T, E, F, Fut>(
    query_fn: F,
    predicate: impl Fn(&T) -> bool,
    timeout_msg: &str,
) -> T
where
    T: Debug,
    E: Debug,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    poll_for_result_with_interval_and_timeout(
        query_fn,
        predicate,
        POLL_INTERVAL,
        POLL_TIMEOUT,
        timeout_msg,
    )
    .await
}

/// Polls a query every `poll_interval` until the result matches a predicate.
///
/// Returns the first `Ok` result that satisfies `predicate`. Retries on both `Err` results and
/// `Ok` results that don't satisfy the predicate. Panics only when the `timeout` is reached,
/// reporting the last non-matching outcome.
pub async fn poll_for_result_with_interval_and_timeout<T, E, F, Fut>(
    mut query_fn: F,
    predicate: impl Fn(&T) -> bool,
    poll_interval: Duration,
    timeout: Duration,
    timeout_msg: &str,
) -> T
where
    T: Debug,
    E: Debug,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        match query_fn().await {
            Ok(result) if predicate(&result) => return result,
            other => {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "{timeout_msg}: {other:?}",
                );
            }
        }
        tokio::time::sleep(poll_interval).await;
    }
}
