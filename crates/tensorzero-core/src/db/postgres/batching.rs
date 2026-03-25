use std::time::Duration;

use futures::{FutureExt, TryFutureExt};
use sqlx::PgPool;
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use sqlx::QueryBuilder;

use crate::config::BatchWritesConfig;
use crate::db::BatchWriterHandle;
use crate::db::batching::{ChannelReceiver, process_channel_with_capacity_and_timeout};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    ChatInferenceDatabaseInsert, JsonInferenceDatabaseInsert, StoredModelInference,
};

use super::inference_queries::{
    build_insert_chat_inference_data_query, build_insert_chat_inferences_query,
    build_insert_json_inference_data_query, build_insert_json_inferences_query,
};
use super::model_inferences::{
    build_insert_model_inference_data_query, build_insert_model_inferences_query,
};

/// A `PostgresBatchSender` is used to submit entries to the batch writer, which aggregates
/// and submits them to Postgres on a schedule defined by a `BatchWritesConfig`.
///
/// By default, channels are unbounded (no data is dropped). If `write_queue_capacity` is set,
/// channels are bounded: when full, new rows are dropped and logged rather than buffering
/// without limit. This protects against out-of-memory crashes at the cost of data loss
/// under sustained backpressure.
///
/// When a `PostgresBatchSender` is dropped, the batch writer will finish
/// processing all outstanding batches once all senders are dropped.
#[derive(Debug)]
pub struct PostgresBatchSender {
    chat_inferences: ChannelSender<ChatInferenceDatabaseInsert>,
    json_inferences: ChannelSender<JsonInferenceDatabaseInsert>,
    model_inferences: ChannelSender<StoredModelInference>,
    pub writer_handle: BatchWriterHandle,
}

/// Wraps either a bounded or unbounded mpsc sender.
#[derive(Debug)]
enum ChannelSender<T> {
    Bounded(mpsc::Sender<T>),
    Unbounded(mpsc::UnboundedSender<T>),
}

impl<T> ChannelSender<T> {
    /// Send a value, dropping it with an error log if the bounded channel is full.
    fn send(&self, value: T, type_name: &str) {
        match self {
            ChannelSender::Bounded(tx) => match tx.try_send(value) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::error!(
                        "Postgres batch channel full — dropping {type_name} record. \
                         Increase `write_queue_capacity` or check Postgres performance."
                    );
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    tracing::error!(
                        "Postgres batch writer has shut down — dropping {type_name} record."
                    );
                }
            },
            ChannelSender::Unbounded(tx) => {
                if let Err(e) = tx.send(value) {
                    tracing::error!(
                        "Postgres batch writer has shut down — dropping {type_name} record. \
                         Error: {e}"
                    );
                }
            }
        }
    }
}

fn create_channel_pair<T>(capacity: Option<usize>) -> (ChannelSender<T>, ChannelReceiver<T>) {
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

impl PostgresBatchSender {
    pub fn new(pool: PgPool, config: BatchWritesConfig) -> Result<Self, Error> {
        // We call `tokio::task::block_in_place` during shutdown to wait for outstanding
        // batch writes to finish. This does not work on the CurrentThread runtime,
        // so we fail here rather than panicking at shutdown.
        if Handle::current().runtime_flavor() == RuntimeFlavor::CurrentThread
            && !config.__force_allow_embedded_batch_writes
        {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cannot use Postgres batching with the CurrentThread Tokio runtime"
                    .to_string(),
            }));
        }

        let (chat_tx, chat_rx) = create_channel_pair(config.write_queue_capacity);
        let (json_tx, json_rx) = create_channel_pair(config.write_queue_capacity);
        let (model_tx, model_rx) = create_channel_pair(config.write_queue_capacity);

        let writer = PostgresBatchWriter {
            chat_inferences_rx: chat_rx,
            json_inferences_rx: json_rx,
            model_inferences_rx: model_rx,
        };

        let handle = tokio::runtime::Handle::current();
        // We use `spawn_blocking` to ensure that when the runtime shuts down, it waits for this task to complete.
        let writer_handle = tokio::task::spawn_blocking(move || {
            handle.block_on(async move {
                tracing::debug!("Postgres batch write handler started");
                writer.process(pool, config).await;
                tracing::info!("Postgres batch write handler finished");
            });
        });

        Ok(Self {
            chat_inferences: chat_tx,
            json_inferences: json_tx,
            model_inferences: model_tx,
            writer_handle: writer_handle.map_err(|e| format!("{e:?}")).boxed().shared(),
        })
    }

    pub fn send_chat_inferences(&self, rows: &[ChatInferenceDatabaseInsert]) {
        for row in rows {
            self.chat_inferences.send(row.clone(), "chat inference");
        }
    }

    pub fn send_json_inferences(&self, rows: &[JsonInferenceDatabaseInsert]) {
        for row in rows {
            self.json_inferences.send(row.clone(), "json inference");
        }
    }

    pub fn send_model_inferences(&self, rows: &[StoredModelInference]) {
        for row in rows {
            self.model_inferences.send(row.clone(), "model inference");
        }
    }
}

struct PostgresBatchWriter {
    chat_inferences_rx: ChannelReceiver<ChatInferenceDatabaseInsert>,
    json_inferences_rx: ChannelReceiver<JsonInferenceDatabaseInsert>,
    model_inferences_rx: ChannelReceiver<StoredModelInference>,
}

/// Spawn a flush task that periodically drains `channel` and executes two insert queries
/// (one for metadata, one for data) built from each batch.
#[expect(clippy::too_many_arguments)]
fn spawn_flush_task<T: Send + 'static>(
    join_set: &mut JoinSet<()>,
    channel: ChannelReceiver<T>,
    pool: PgPool,
    max_rows: usize,
    batch_timeout: Duration,
    build_meta: fn(&[T]) -> Result<QueryBuilder<sqlx::Postgres>, Error>,
    build_data: fn(&[T]) -> Result<QueryBuilder<sqlx::Postgres>, Error>,
    meta_err: &'static str,
    data_err: &'static str,
    meta_build_err: &'static str,
    data_build_err: &'static str,
) {
    join_set.spawn(async move {
        process_channel_with_capacity_and_timeout(
            channel,
            max_rows,
            batch_timeout,
            move |buffer| {
                let pool = pool.clone();
                async move {
                    // TODO: if this errors, should we retry?
                    match build_meta(&buffer) {
                        Ok(mut qb) => {
                            if let Err(e) = qb.build().execute(&pool).await {
                                tracing::error!("{meta_err}: {e}");
                            }
                        }
                        Err(e) => {
                            tracing::error!("{meta_build_err}: {e}");
                        }
                    }
                    match build_data(&buffer) {
                        Ok(mut qb) => {
                            if let Err(e) = qb.build().execute(&pool).await {
                                tracing::error!("{data_err}: {e}");
                            }
                        }
                        Err(e) => {
                            tracing::error!("{data_build_err}: {e}");
                        }
                    }
                    buffer
                }
            },
        )
        .await;
    });
}

impl PostgresBatchWriter {
    async fn process(self, pool: PgPool, config: BatchWritesConfig) {
        let mut join_set = JoinSet::new();
        let batch_timeout = Duration::from_millis(config.flush_interval_ms);
        let max_rows = config.max_rows_postgres.unwrap_or(config.max_rows);

        // Chat inferences flush task
        spawn_flush_task(
            &mut join_set,
            self.chat_inferences_rx,
            pool.clone(),
            max_rows,
            batch_timeout,
            build_insert_chat_inferences_query,
            build_insert_chat_inference_data_query,
            "Error writing chat inferences to Postgres",
            "Error writing chat inference IO to Postgres",
            "Error building chat inferences query",
            "Error building chat inference IO query",
        );

        // JSON inferences flush task
        spawn_flush_task(
            &mut join_set,
            self.json_inferences_rx,
            pool.clone(),
            max_rows,
            batch_timeout,
            build_insert_json_inferences_query,
            build_insert_json_inference_data_query,
            "Error writing json inferences to Postgres",
            "Error writing json inference IO to Postgres",
            "Error building json inferences query",
            "Error building json inference IO query",
        );

        // Model inferences flush task
        spawn_flush_task(
            &mut join_set,
            self.model_inferences_rx,
            pool,
            max_rows,
            batch_timeout,
            build_insert_model_inferences_query,
            build_insert_model_inference_data_query,
            "Error writing model inferences to Postgres",
            "Error writing model inference IO to Postgres",
            "Error building model inferences query",
            "Error building model inference IO query",
        );

        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                tracing::error!("Error in Postgres batch writer: {e}");
            }
        }
    }
}
