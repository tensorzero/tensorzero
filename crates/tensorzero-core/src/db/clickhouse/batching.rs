use std::time::Duration;

use crate::config::BatchWritesConfig;
use crate::db::BatchWriterHandle;
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use enum_map::EnumMap;
use futures::{FutureExt, TryFutureExt};
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use crate::db::batching::{ChannelReceiver, process_channel_with_capacity_and_timeout};
use crate::db::clickhouse::{ClickHouseConnectionInfo, Rows, TableName};
use crate::error::{Error, ErrorDetails};

/// Wraps either a bounded or unbounded mpsc sender.
#[derive(Debug)]
enum ChannelSender {
    Bounded(mpsc::Sender<String>),
    Unbounded(mpsc::UnboundedSender<String>),
}

fn create_channel_pair(capacity: Option<usize>) -> (ChannelSender, ChannelReceiver<String>) {
    match capacity {
        Some(cap) => {
            let (tx, rx) = mpsc::channel(cap);
            (ChannelSender::Bounded(tx), ChannelReceiver::Bounded(rx))
        }
        None => {
            let (tx, rx) = mpsc::unbounded_channel();
            (ChannelSender::Unbounded(tx), ChannelReceiver::Unbounded(rx))
        }
    }
}

/// A `BatchSender` is used to submit entries to the batch writer, which aggregates
/// and submits them to ClickHouse on a schedule defined by a `BatchWritesConfig`.
///
/// By default, channels are unbounded (no data is dropped). If `write_queue_capacity` is set,
/// channels are bounded: when full, new rows are dropped and logged rather than buffering
/// without limit.
///
/// When a `BatchSender` is dropped, it blocks until the batch writer finishes
/// processing all outstanding batches.
#[derive(Debug)]
pub struct BatchSender {
    // This needs to be an `Option`, so that we can drop it
    // (in particular, the sender) from our `Drop` impl.
    // This signals to the writer tasks that the channel is closed,
    // and that they should exit after they finish processing all messages
    // currently in the channel.
    channels: Option<EnumMap<TableName, ChannelSender>>,
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
        let capacity = config.write_queue_capacity;
        let mut channels: EnumMap<TableName, _> = enum_map::enum_map! {
            _ => {
                let (tx, rx) = create_channel_pair(capacity);
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

    pub fn add_to_batch(&self, table_name: TableName, rows: Vec<String>) -> Result<(), Error> {
        let Some(channels) = &self.channels else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Batch sender dropped. {IMPOSSIBLE_ERROR_MESSAGE}"),
            }));
        };
        let channel = &channels[table_name];
        for row in rows {
            match channel {
                ChannelSender::Bounded(tx) => match tx.try_send(row) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        tracing::error!(
                            table = ?table_name,
                            "ClickHouse batch channel full — dropping row. \
                             Increase `write_queue_capacity` or check ClickHouse performance."
                        );
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        tracing::error!(
                            table = ?table_name,
                            "ClickHouse batch writer has shut down — dropping row."
                        );
                    }
                },
                ChannelSender::Unbounded(tx) => {
                    if let Err(e) = tx.send(row) {
                        tracing::error!(
                            "Error sending row to batch channel: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

pub struct BatchWriter {
    channels: EnumMap<TableName, ChannelReceiver<String>>,
}

impl BatchWriter {
    pub async fn process(self, clickhouse: ClickHouseConnectionInfo, config: BatchWritesConfig) {
        let mut join_set = JoinSet::new();
        let batch_timeout = Duration::from_millis(
            config
                .flush_interval_ms
                .unwrap_or_else(crate::config::default_flush_interval_ms),
        );
        let max_rows = config
            .max_rows
            .unwrap_or_else(crate::config::default_max_rows);

        for (table_name, channel) in self.channels {
            let clickhouse = clickhouse.clone();
            let flush = move |buffer: Vec<String>| {
                let clickhouse = clickhouse.clone();
                async move {
                    if let Err(e) = clickhouse
                        .write_non_batched::<()>(Rows::Serialized(&buffer), table_name)
                        .await
                    {
                        // TODO: if this errors, should we retry?
                        tracing::error!("Error writing to ClickHouse: {e}");
                    }
                    buffer
                }
            };
            join_set.spawn(async move {
                process_channel_with_capacity_and_timeout(channel, max_rows, batch_timeout, flush)
                    .await;
            });
        }
        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                tracing::error!("Error in batch writer: {e}");
            }
        }
    }
}
