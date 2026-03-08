use std::time::Duration;

use futures::{FutureExt, TryFutureExt};
use sqlx::PgPool;
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinSet;

use crate::config::BatchWritesConfig;
use crate::db::BatchWriterHandle;
use crate::db::batching::process_channel_with_capacity_and_timeout;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
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
/// When a `PostgresBatchSender` is dropped, the batch writer will finish
/// processing all outstanding batches once all senders are dropped.
#[derive(Debug)]
pub struct PostgresBatchSender {
    channels: Option<PostgresBatchChannels>,
    pub writer_handle: BatchWriterHandle,
}

#[derive(Debug)]
struct PostgresBatchChannels {
    chat_inferences: UnboundedSender<ChatInferenceDatabaseInsert>,
    json_inferences: UnboundedSender<JsonInferenceDatabaseInsert>,
    model_inferences: UnboundedSender<StoredModelInference>,
}

impl PostgresBatchSender {
    pub fn new(pool: PgPool, config: BatchWritesConfig) -> Result<Self, Error> {
        // We call `tokio::task::block_in_place` during shutdown to wait for outstanding
        // batch writes to finish. This does not work on the CurrentThread runtime,
        // so we fail here rather than panicking at shutdown.
        if Handle::current().runtime_flavor() == RuntimeFlavor::CurrentThread {
            return Err(Error::new(ErrorDetails::InternalError {
                message: "Cannot use Postgres batching with the CurrentThread Tokio runtime"
                    .to_string(),
            }));
        }

        let (chat_tx, chat_rx) = mpsc::unbounded_channel();
        let (json_tx, json_rx) = mpsc::unbounded_channel();
        let (model_tx, model_rx) = mpsc::unbounded_channel();

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
            channels: Some(PostgresBatchChannels {
                chat_inferences: chat_tx,
                json_inferences: json_tx,
                model_inferences: model_tx,
            }),
            writer_handle: writer_handle.map_err(|e| format!("{e:?}")).boxed().shared(),
        })
    }

    pub fn send_chat_inferences(&self, rows: &[ChatInferenceDatabaseInsert]) -> Result<(), Error> {
        let Some(channels) = &self.channels else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Postgres batch sender dropped. {IMPOSSIBLE_ERROR_MESSAGE}"),
            }));
        };
        for row in rows {
            if let Err(e) = channels.chat_inferences.send(row.clone()) {
                tracing::error!(
                    "Error sending chat inference to batch channel: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"
                );
            }
        }
        Ok(())
    }

    pub fn send_json_inferences(&self, rows: &[JsonInferenceDatabaseInsert]) -> Result<(), Error> {
        let Some(channels) = &self.channels else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Postgres batch sender dropped. {IMPOSSIBLE_ERROR_MESSAGE}"),
            }));
        };
        for row in rows {
            if let Err(e) = channels.json_inferences.send(row.clone()) {
                tracing::error!(
                    "Error sending json inference to batch channel: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"
                );
            }
        }
        Ok(())
    }

    pub fn send_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error> {
        let Some(channels) = &self.channels else {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!("Postgres batch sender dropped. {IMPOSSIBLE_ERROR_MESSAGE}"),
            }));
        };
        for row in rows {
            if let Err(e) = channels.model_inferences.send(row.clone()) {
                tracing::error!(
                    "Error sending model inference to batch channel: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"
                );
            }
        }
        Ok(())
    }
}

struct PostgresBatchWriter {
    chat_inferences_rx: UnboundedReceiver<ChatInferenceDatabaseInsert>,
    json_inferences_rx: UnboundedReceiver<JsonInferenceDatabaseInsert>,
    model_inferences_rx: UnboundedReceiver<StoredModelInference>,
}

impl PostgresBatchWriter {
    async fn process(self, pool: PgPool, config: BatchWritesConfig) {
        let mut join_set = JoinSet::new();
        let batch_timeout = Duration::from_millis(config.flush_interval_ms);
        let max_rows = config.max_rows_postgres.unwrap_or(config.max_rows);

        // Chat inferences flush task
        {
            let pool = pool.clone();
            let channel = self.chat_inferences_rx;
            join_set.spawn(async move {
                process_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        async move {
                            // TODO: if this errors, should we retry?
                            match build_insert_chat_inferences_query(&buffer) {
                                Ok(mut qb) => {
                                    if let Err(e) = qb.build().execute(&pool).await {
                                        tracing::error!(
                                            "Error writing chat inferences to Postgres: {e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Error building chat inferences query: {e}");
                                }
                            }
                            match build_insert_chat_inference_data_query(&buffer) {
                                Ok(mut qb) => {
                                    if let Err(e) = qb.build().execute(&pool).await {
                                        tracing::error!(
                                            "Error writing chat inference IO to Postgres: {e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Error building chat inference IO query: {e}");
                                }
                            }
                            buffer
                        }
                    },
                )
                .await;
            });
        }

        // JSON inferences flush task
        {
            let pool = pool.clone();
            let channel = self.json_inferences_rx;
            join_set.spawn(async move {
                process_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        async move {
                            // TODO: if this errors, should we retry?
                            match build_insert_json_inferences_query(&buffer) {
                                Ok(mut qb) => {
                                    if let Err(e) = qb.build().execute(&pool).await {
                                        tracing::error!(
                                            "Error writing json inferences to Postgres: {e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Error building json inferences query: {e}");
                                }
                            }
                            match build_insert_json_inference_data_query(&buffer) {
                                Ok(mut qb) => {
                                    if let Err(e) = qb.build().execute(&pool).await {
                                        tracing::error!(
                                            "Error writing json inference IO to Postgres: {e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Error building json inference IO query: {e}");
                                }
                            }
                            buffer
                        }
                    },
                )
                .await;
            });
        }

        // Model inferences flush task
        {
            let channel = self.model_inferences_rx;
            join_set.spawn(async move {
                process_channel_with_capacity_and_timeout(
                    channel,
                    max_rows,
                    batch_timeout,
                    move |buffer| {
                        let pool = pool.clone();
                        async move {
                            // TODO: if this errors, should we retry?
                            match build_insert_model_inferences_query(&buffer) {
                                Ok(mut qb) => {
                                    if let Err(e) = qb.build().execute(&pool).await {
                                        tracing::error!(
                                            "Error writing model inferences to Postgres: {e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Error building model inferences query: {e}");
                                }
                            }
                            match build_insert_model_inference_data_query(&buffer) {
                                Ok(mut qb) => {
                                    if let Err(e) = qb.build().execute(&pool).await {
                                        tracing::error!(
                                            "Error writing model inference IO to Postgres: {e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Error building model inference IO query: {e}");
                                }
                            }
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
