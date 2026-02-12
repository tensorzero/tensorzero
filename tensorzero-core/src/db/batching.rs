use std::future::Future;
use std::time::Duration;

use tokio::sync::mpsc::UnboundedReceiver;

/// Shared channel flushing logic for batch writers (used by both ClickHouse and Postgres).
///
/// Buffers items from `channel` and calls `flush` when the buffer reaches
/// `max_rows` or `flush_interval` elapses, whichever comes first.
/// Drains all remaining items when the channel closes.
///
/// The `flush` callback takes ownership of the buffer and returns it, so we can
/// avoid reallocating between batches while keeping the callback `Send`.
pub(crate) async fn process_channel_with_capacity_and_timeout<T, F, Fut>(
    mut channel: UnboundedReceiver<T>,
    max_rows: usize,
    flush_interval: Duration,
    mut flush: F,
) where
    T: Send + 'static,
    // TODO: try making this F: FnMut(&mut Vec<T>)
    F: FnMut(Vec<T>) -> Fut + Send + 'static,
    Fut: Future<Output = Vec<T>> + Send + 'static,
{
    let mut buffer = Vec::with_capacity(max_rows);
    // The channel can be closed but still contain messages.
    // We continue looping until the channel is closed and empty, at which point we're guaranteed
    // to never see any new messages from `channel.recv/recv_many`.
    while !channel.is_closed() || !channel.is_empty() {
        let deadline = tokio::time::Instant::now() + flush_interval;
        // Repeatedly fetch entries from the channel until we have a full batch.
        // We exit early from the loop if our deadline is reached, and submit
        // however many rows we have.
        while buffer.len() < max_rows {
            let remaining = max_rows - buffer.len();
            // `recv_many` is explicitly documented to be cancellation-safe,
            // so we can safely wrap it in a timeout without losing messages
            match tokio::time::timeout_at(deadline, channel.recv_many(&mut buffer, remaining)).await
            {
                // The channel has closed, so we're done with this batch
                Ok(0) => break,
                // We added some rows to the buffer, so keep receiving more rows
                Ok(_) => {}
                Err(elapsed) => {
                    // Compile-time assertion that this is actually an Elapsed error.
                    // We hit our deadline, so we should submit our current batch
                    let _: tokio::time::error::Elapsed = elapsed;
                    break;
                }
            }
        }
        if !buffer.is_empty() {
            buffer = flush(buffer).await;
            buffer.clear();
        }
    }
}
