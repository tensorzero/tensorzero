use std::time::Duration;

use futures::{FutureExt, TryFutureExt};
use sqlx::PgPool;
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::mpsc::{self, Receiver, Sender, error::TrySendError};
use tokio::task::JoinSet;

use crate::config::BatchWritesConfig;
use crate::db::BatchWriterHandle;
use crate::db::batching::process_bounded_channel_with_capacity_and_timeout;
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
/// Uses bounded channels to provide backpressure: when the write queue is full,
/// new rows are dropped and logged rather than buffering without limit.
///
/// Each table type (chat, json, model inferences) has its own accumulator task that
/// collects rows and flushes them when `max_rows` or `flush_interval_ms` is reached.
/// Within each flush, the metadata and data INSERTs run concurrently via `tokio::join!`.
///
/// ## Detecting backpressure
///
/// When Postgres INSERT latency exceeds the flush interval, the accumulator blocks
/// (it can't accept new rows while flushing). This causes the bounded input channel
/// to fill up, at which point `try_send` fails with `TrySendError::Full` and the
/// error is logged. Monitor for these log messages:
///
///   `"Postgres batch channel full — dropping ... inference record"`
///
/// If you see sustained drops, consider:
/// 1. Increasing `write_queue_capacity` to absorb temporary spikes
/// 2. Tuning `max_rows` / `flush_interval_ms` to reduce per-flush latency
/// 3. Scaling up Postgres (connections, IOPS, etc.)
///
/// ## Future: concurrent flush worker pool
///
/// The current design flushes serially within each accumulator — while an INSERT is
/// in flight, the accumulator cannot collect new rows. If INSERT latency regularly
/// exceeds `flush_interval_ms` and the above tuning is insufficient, the next step
/// would be to decouple accumulation from flushing by introducing a shared pool of
/// N flush workers. The accumulator would hand off ready batches to the pool and
/// immediately start collecting the next batch. This adds complexity (shared
/// `Arc<Mutex<Receiver>>`, boxed futures, two-stage shutdown) but eliminates
/// accumulator stalls under sustained Postgres latency.
///
/// When a `PostgresBatchSender` is dropped, the batch writer will finish
/// processing all outstanding batches once all senders are dropped.
#[derive(Debug)]
pub struct PostgresBatchSender {
    chat_inferences: Sender<ChatInferenceDatabaseInsert>,
    json_inferences: Sender<JsonInferenceDatabaseInsert>,
    model_inferences: Sender<StoredModelInference>,
    pub writer_handle: BatchWriterHandle,
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

        let channel_capacity = config.write_queue_capacity;
        let (chat_tx, chat_rx) = mpsc::channel(channel_capacity);
        let (json_tx, json_rx) = mpsc::channel(channel_capacity);
        let (model_tx, model_rx) = mpsc::channel(channel_capacity);

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
            match self.chat_inferences.try_send(row.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    tracing::error!(
                        "Postgres batch channel full — dropping chat inference record. \
                         Increase `write_queue_capacity` or check Postgres performance."
                    );
                }
                Err(TrySendError::Closed(_)) => {
                    tracing::error!(
                        "Postgres batch writer has shut down — dropping chat inference record."
                    );
                }
            }
        }
    }

    pub fn send_json_inferences(&self, rows: &[JsonInferenceDatabaseInsert]) {
        for row in rows {
            match self.json_inferences.try_send(row.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    tracing::error!(
                        "Postgres batch channel full — dropping json inference record. \
                         Increase `write_queue_capacity` or check Postgres performance."
                    );
                }
                Err(TrySendError::Closed(_)) => {
                    tracing::error!(
                        "Postgres batch writer has shut down — dropping json inference record."
                    );
                }
            }
        }
    }

    pub fn send_model_inferences(&self, rows: &[StoredModelInference]) {
        for row in rows {
            match self.model_inferences.try_send(row.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    tracing::error!(
                        "Postgres batch channel full — dropping model inference record. \
                         Increase `write_queue_capacity` or check Postgres performance."
                    );
                }
                Err(TrySendError::Closed(_)) => {
                    tracing::error!(
                        "Postgres batch writer has shut down — dropping model inference record."
                    );
                }
            }
        }
    }
}

struct PostgresBatchWriter {
    chat_inferences_rx: Receiver<ChatInferenceDatabaseInsert>,
    json_inferences_rx: Receiver<JsonInferenceDatabaseInsert>,
    model_inferences_rx: Receiver<StoredModelInference>,
}

impl PostgresBatchWriter {
    async fn process(self, pool: PgPool, config: BatchWritesConfig) {
        let mut join_set = JoinSet::new();
        let batch_timeout = Duration::from_millis(config.flush_interval_ms);
        let max_rows = config.max_rows_postgres.unwrap_or(config.max_rows);

        // Chat inferences accumulator
        {
            let pool = pool.clone();
            let channel = self.chat_inferences_rx;
            join_set.spawn(async move {
                process_bounded_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        async move {
                            flush_chat_inferences(&pool, &buffer).await;
                            buffer
                        }
                    },
                )
                .await;
            });
        }

        // JSON inferences accumulator
        {
            let pool = pool.clone();
            let channel = self.json_inferences_rx;
            join_set.spawn(async move {
                process_bounded_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        async move {
                            flush_json_inferences(&pool, &buffer).await;
                            buffer
                        }
                    },
                )
                .await;
            });
        }

        // Model inferences accumulator
        {
            let channel = self.model_inferences_rx;
            join_set.spawn(async move {
                process_bounded_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        async move {
                            flush_model_inferences(&pool, &buffer).await;
                            buffer
                        }
                    },
                )
                .await;
            });
        }

        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                tracing::error!("Error in Postgres batch writer: {e}");
            }
        }
    }
}

/// Execute both chat inference INSERTs (metadata + data) concurrently.
async fn flush_chat_inferences(pool: &PgPool, buffer: &[ChatInferenceDatabaseInsert]) {
    let row_count = buffer.len();
    let metadata_future = async {
        match build_insert_chat_inferences_query(buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), pool, "chat_inferences", row_count).await
                {
                    tracing::error!("Error writing chat inferences to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building chat inferences query: {e}");
            }
        }
    };
    let data_future = async {
        match build_insert_chat_inference_data_query(buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), pool, "chat_inference_data", row_count)
                        .await
                {
                    tracing::error!("Error writing chat inference data to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building chat inference data query: {e}");
            }
        }
    };
    tokio::join!(metadata_future, data_future);
}

/// Execute both JSON inference INSERTs (metadata + data) concurrently.
async fn flush_json_inferences(pool: &PgPool, buffer: &[JsonInferenceDatabaseInsert]) {
    let row_count = buffer.len();
    let metadata_future = async {
        match build_insert_json_inferences_query(buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), pool, "json_inferences", row_count).await
                {
                    tracing::error!("Error writing json inferences to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building json inferences query: {e}");
            }
        }
    };
    let data_future = async {
        match build_insert_json_inference_data_query(buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), pool, "json_inference_data", row_count)
                        .await
                {
                    tracing::error!("Error writing json inference data to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building json inference data query: {e}");
            }
        }
    };
    tokio::join!(metadata_future, data_future);
}

/// Execute both model inference INSERTs (metadata + data) concurrently.
async fn flush_model_inferences(pool: &PgPool, buffer: &[StoredModelInference]) {
    let row_count = buffer.len();
    let metadata_future = async {
        match build_insert_model_inferences_query(buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), pool, "model_inferences", row_count)
                        .await
                {
                    tracing::error!("Error writing model inferences to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building model inferences query: {e}");
            }
        }
    };
    let data_future = async {
        match build_insert_model_inference_data_query(buffer) {
            Ok(mut qb) => {
                if let Err(e) =
                    super::execute_with_timing(qb.build(), pool, "model_inference_data", row_count)
                        .await
                {
                    tracing::error!("Error writing model inference data to Postgres: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Error building model inference data query: {e}");
            }
        }
    };
    tokio::join!(metadata_future, data_future);
}
