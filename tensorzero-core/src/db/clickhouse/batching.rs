use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use crate::config::BatchWritesConfig;
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use enum_map::EnumMap;
use futures::future::Shared;
use futures::{FutureExt, TryFutureExt};
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinSet;
use tokio::time::error::Elapsed;
use tokio::time::Instant;

use crate::db::clickhouse::{ClickHouseConnectionInfo, Rows, TableName};
use crate::error::{Error, ErrorDetails};

pub type BatchWriterHandle = Shared<Pin<Box<dyn Future<Output = Result<(), String>> + Send>>>;

/// A `BatchSender` is used to submit entries to the batch writer, which aggregates
/// and submits them to ClickHouse on a schedule defined by a `BatchWritesConfig`.
/// When a `BatchSender` is dropped, it blocks until the batch writer finishes
/// processing all outstanding batches.
#[derive(Debug)]
pub struct BatchSender {
    // This needs to be an `Option`, so that we can drop it
    // (in particular, the `UnboundedSender`) from our `Drop` impl.
    // This signals to the writer tasks that the channel is closed,
    // and that they should exit after they finish processing all messages
    // currently in the channell.
    channels: Option<EnumMap<TableName, UnboundedSender<String>>>,
    pub writer_handle: BatchWriterHandle,
}

impl BatchSender {
    pub fn new(
        clickhouse: ClickHouseConnectionInfo,
        config: BatchWritesConfig,
    ) -> Result<Self, Error> {
        // We call `tokio::task::block_in_place` in our `Drop` impl to wait for outstanding
        // batch writes to finish. This does not work on the CurrentThread runtime,
        // so we fail here rather than panicking at shutdown.
        if Handle::current().runtime_flavor() == RuntimeFlavor::CurrentThread {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cannot use ClickHouse batching with the CurrentThread Tokio runtime"
                    .to_string(),
            }));
        }
        let mut channels: EnumMap<TableName, _> = enum_map::enum_map! {
            _ => {
                let (tx, rx) = mpsc::unbounded_channel();
                (Some(tx), Some(rx))
            }
        };
        let reader_channels = enum_map::enum_map! {
            table_name => { channels[table_name].0.take().ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to take reader channel for table {table_name:?}. {IMPOSSIBLE_ERROR_MESSAGE}"),
                })
            })? }
        };
        let writer_channels = enum_map::enum_map! {
            table_name => { channels[table_name].1.take().ok_or_else(|| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to take writer channel for table {table_name:?}. {IMPOSSIBLE_ERROR_MESSAGE}"),
                })
            })? }
        };
        let writer: BatchWriter = BatchWriter {
            channels: writer_channels,
        };
        let handle = tokio::runtime::Handle::current();
        // We intentionally don't use a `CancellationToken` here - we want the batch writer
        // to keep running as long a `Sender` is still active (from inside a
        // `ClickHouseConnectionInfo`). We only exit once all of the `Sender`s are dropped,
        // (and we've finished writing our current batch)
        // We use `spawn_blocking` to ensure that when the runtime shuts down, it waits for this task to complete.
        let writer_handle = tokio::task::spawn_blocking(move || {
            handle.block_on(async move {
                tracing::debug!("ClickHouse batch write handler started");
                writer.process(clickhouse, config).await;
                tracing::info!("ClickHouse batch write handler finished");
            });
        });
        Ok(Self {
            channels: Some(reader_channels),
            writer_handle: writer_handle.map_err(|e| format!("{e:?}")).boxed().shared(),
        })
    }

    pub async fn add_to_batch(
        &self,
        table_name: TableName,
        rows: Vec<String>,
    ) -> Result<(), Error> {
        let Some(channels) = &self.channels else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Batch sender dropped. {IMPOSSIBLE_ERROR_MESSAGE}"),
            }));
        };
        let channel = &channels[table_name];
        for row in rows {
            if let Err(e) = channel.send(row) {
                tracing::error!(
                    "Error sending row to batch channel: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"
                );
            }
        }
        Ok(())
    }
}

pub struct BatchWriter {
    channels: EnumMap<TableName, UnboundedReceiver<String>>,
}

impl BatchWriter {
    pub async fn process(self, clickhouse: ClickHouseConnectionInfo, config: BatchWritesConfig) {
        let mut join_set = JoinSet::new();
        for (table_name, mut channel) in self.channels {
            let clickhouse = clickhouse.clone();
            let batch_timeout = Duration::from_millis(config.flush_interval_ms);
            join_set.spawn(async move {
                let mut buffer = Vec::with_capacity(config.max_rows);
                // The channel can be closed but still contain messages.
                // We continue looping until the channel is closed and empty, at which point we're guaranteed
                // to never see any new messages from `channel.recv/recv_many`.
                while !channel.is_closed() || !channel.is_empty() {
                    let deadline = Instant::now() + batch_timeout;
                    // Repeatedly fetch entries from the channel until we have a full batch.
                    // We exit early from the loop if our deadline is reached, and submit
                    // however many rows we have.
                    while buffer.len() < config.max_rows {
                        let remaining = config.max_rows - buffer.len();
                        // `recv_many` is explicitly documented to be cancellation-safe,
                        // so we can safely wrap it in a timeout without losing messages
                        match tokio::time::timeout_at(
                            deadline,
                            channel.recv_many(&mut buffer, remaining),
                        )
                        .await
                        {
                            // The channel has closed, so we're done with this batch
                            Ok(0) => break,
                            // We added some rows to the buffer, so keep receiving more rows
                            Ok(_) => {}
                            Err(e) => {
                                // Compile-time assertion that this is actually an Elapsed error.
                                // We hit our deadline, so we should submit our current batch
                                let _: Elapsed = e;
                                break;
                            }
                        }
                    }
                    if !buffer.is_empty() {
                        if let Err(e) = clickhouse
                            .write_non_batched::<()>(Rows::Serialized(&buffer), table_name)
                            .await
                        {
                            // TODO - should we retry?
                            tracing::error!("Error writing to ClickHouse: {e}");
                        }
                        buffer.clear();
                    }
                }
            });
        }
        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                tracing::error!("Error in batch writer: {e}");
            }
        }
    }
}
