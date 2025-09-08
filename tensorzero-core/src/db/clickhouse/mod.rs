use async_trait::async_trait;
use enum_map::Enum;
use migration_manager::migrations::check_table_exists;
use reqwest::multipart::Form;
use reqwest::multipart::Part;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::sync::RwLockWriteGuard;
use url::Url;

mod batching;
pub mod migration_manager;
pub mod query_builder;
mod select_queries;
#[cfg(any(test, feature = "e2e_tests"))]
pub mod test_helpers;

use crate::config::BatchWritesConfig;
use crate::config::Config;
use crate::db::clickhouse::batching::BatchSender;
use crate::db::clickhouse::batching::BatchWriterHandle;
use crate::error::DisplayOrDebugGateway;
use crate::error::{Error, ErrorDetails};
use crate::stored_inference::StoredInference;
use query_builder::generate_list_inferences_sql;
use query_builder::ListInferencesParams;

use super::HealthCheckable;

#[derive(Debug, Clone)]
pub enum ClickHouseConnectionInfo {
    Disabled,
    Mock {
        mock_data: Arc<RwLock<HashMap<String, Vec<serde_json::Value>>>>,
        healthy: bool,
    },
    Production {
        database_url: SecretString,
        cluster_name: Option<String>,
        database: String,
        client: Client,
        batch_sender: Option<Arc<BatchSender>>,
    },
}

pub fn make_clickhouse_http_client() -> Result<Client, Error> {
    Client::builder()
        // https://github.com/ClickHouse/clickhouse-rs/blob/56c5dd3fc95693acc5aa3d02db1f910a26fe5b1c/src/http_client.rs#L45
        .pool_idle_timeout(Duration::from_secs(2))
        // https://github.com/ClickHouse/clickhouse-rs/blob/56c5dd3fc95693acc5aa3d02db1f910a26fe5b1c/src/http_client.rs#L41
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .build()
        .map_err(|e| {
            Error::new(ErrorDetails::ClickHouseConnection {
                message: format!("Failed to build ClickHouse HTTP client: {e}"),
            })
        })
}

/// Defines all of the ClickHouse tables that we write to from Rust
/// This will be used to implement per-table ClickHouse write batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Enum)]
pub enum TableName {
    BatchModelInference,
    BatchRequest,
    ChatInference,
    ChatInferenceDatapoint,
    JsonInference,
    JsonInferenceDatapoint,
    ModelInference,
    ModelInferenceCache,
    DeploymentID,
    TensorZeroMigration,
    BooleanMetricFeedback,
    FloatMetricFeedback,
    DemonstrationFeedback,
    CommentFeedback,
    StaticEvaluationHumanFeedback,
}

impl TableName {
    pub fn as_str(self) -> &'static str {
        match self {
            TableName::BatchModelInference => "BatchModelInference",
            TableName::BatchRequest => "BatchRequest",
            TableName::ChatInference => "ChatInference",
            TableName::ChatInferenceDatapoint => "ChatInferenceDatapoint",
            TableName::JsonInference => "JsonInference",
            TableName::JsonInferenceDatapoint => "JsonInferenceDatapoint",
            TableName::ModelInference => "ModelInference",
            TableName::ModelInferenceCache => "ModelInferenceCache",
            TableName::DeploymentID => "DeploymentID",
            TableName::TensorZeroMigration => "TensorZeroMigration",
            TableName::BooleanMetricFeedback => "BooleanMetricFeedback",
            TableName::FloatMetricFeedback => "FloatMetricFeedback",
            TableName::DemonstrationFeedback => "DemonstrationFeedback",
            TableName::CommentFeedback => "CommentFeedback",
            TableName::StaticEvaluationHumanFeedback => "StaticEvaluationHumanFeedback",
        }
    }
}

impl ClickHouseConnectionInfo {
    /// Create a new ClickHouse connection info from a database URL.
    /// You should always use this function in production code or generic integration tests that
    /// don't test specific ClickHouse behavior.
    /// For e2e tests, you should use the `get_clickhouse` function.
    ///
    /// This function returns an error if anything is malformed but if the connection is unhealthy it logs that and
    /// returns Ok(Production{ ... })
    ///
    /// However, for tests that directly test ClickHouse behavior, you can directly create the struct.
    pub async fn new(database_url: &str, batch_config: BatchWritesConfig) -> Result<Self, Error> {
        // Add a query string for the database using the URL crate
        let mut database_url = Url::parse(database_url).map_err(|_| {
            Error::new(ErrorDetails::Config {
                message: "Invalid ClickHouse database URL".to_string(),
            })
        })?;

        #[cfg(not(feature = "e2e_tests"))]
        let database = validate_clickhouse_url_get_db_name(&database_url)?
            .unwrap_or_else(|| "default".to_string());

        #[cfg(feature = "e2e_tests")]
        let database = std::env::var("TENSORZERO_E2E_TESTS_DATABASE")
            .unwrap_or_else(|_| "tensorzero_e2e_tests".to_string());

        // Although we take the database name from the URL path,
        // we need to set the query string for the database name for the ClickHouse TCP protocol
        database_url.set_path("");
        database_url
            .query_pairs_mut()
            .append_pair("database", &database);

        // Set ClickHouse format settings for some error checking on writes
        set_clickhouse_format_settings(&mut database_url);

        // Store the original URL as a SecretString
        let database_url = SecretString::from(database_url.to_string());

        // Get the cluster name from the `TENSORZERO_CLICKHOUSE_CLUSTER_NAME` environment variable
        let cluster_name = match std::env::var("TENSORZERO_CLICKHOUSE_CLUSTER_NAME") {
            Ok(cluster_name) => {
                tracing::info!("The gateway is expecting a self-hosted replicated ClickHouse deployment with cluster name `{cluster_name}`. Note: The environment variable `TENSORZERO_CLICKHOUSE_CLUSTER_NAME` doesn't apply to ClickHouse Cloud or self-managed single-node deployments.");
                Some(cluster_name)
            }
            Err(_) => {
                tracing::debug!("The environment variable `TENSORZERO_CLICKHOUSE_CLUSTER_NAME` wasn't provided, so the gateway will assume that ClickHouse is not running a self-hosted replicated cluster. Note: This variable doesn't apply to ClickHouse Cloud or self-managed single-node deployments.");
                None
            }
        };

        let mut connection_info = Self::Production {
            database_url,
            cluster_name,
            database,
            client: make_clickhouse_http_client()?,
            batch_sender: None,
        };

        let orig_connection_info = connection_info.clone();
        match &mut connection_info {
            Self::Production { batch_sender, .. } => {
                if batch_config.enabled {
                    // Create the batch sender using the `ClickHouseConnectionInfo` without the `batch_sender` set
                    // (since the batcher itself always performs direct writes)
                    let batcher = BatchSender::new(orig_connection_info, batch_config)?;
                    *batch_sender = Some(Arc::new(batcher));
                }
            }
            Self::Mock { .. } | Self::Disabled { .. } => {}
        }
        // If the connection is unhealthy, we won't be able to run / check migrations. So we just fail here.
        connection_info.health().await?;
        Ok(connection_info)
    }

    pub fn new_mock(healthy: bool) -> Self {
        Self::Mock {
            mock_data: Arc::new(RwLock::new(HashMap::new())),
            healthy,
        }
    }

    pub fn batcher_join_handle(&self) -> Option<BatchWriterHandle> {
        match self {
            Self::Production { batch_sender, .. } => batch_sender
                .as_ref()
                .map(|sender| sender.writer_handle.clone()),
            _ => None,
        }
    }

    pub fn new_disabled() -> Self {
        Self::Disabled
    }

    pub fn database(&self) -> &str {
        match self {
            Self::Disabled => "",
            Self::Mock { .. } => "mock-database",
            Self::Production { database, .. } => database,
        }
    }

    /// Writes rows to ClickHouse using our batched write implementation
    /// (if enabled in the config)
    /// The provided rows might not yet be sent to ClickHouse when this function completes.
    pub async fn write_batched(
        &self,
        rows: &[impl Serialize + Send + Sync],
        table: TableName,
    ) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Mock { mock_data, .. } => {
                write_mock(
                    Rows::Unserialized(rows),
                    table,
                    &mut mock_data.write().await,
                )
                .await
            }
            Self::Production {
                database_url,
                client,
                batch_sender,
                ..
            } => {
                write_production(
                    database_url,
                    client,
                    Rows::Unserialized(rows),
                    table,
                    batch_sender.as_deref(),
                )
                .await
            }
        }
    }

    /// Write rows to ClickHouse without, without using our batched write implementation.
    /// The provided rows will have been sent to ClickHouse when this function completes.
    pub async fn write_non_batched<T: Serialize + Send + Sync>(
        &self,
        rows: Rows<'_, T>,
        table: TableName,
    ) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Mock { mock_data, .. } => {
                write_mock(rows, table, &mut mock_data.write().await).await
            }
            Self::Production {
                database_url,
                client,
                ..
            } => write_production(database_url, client, rows, table, None).await,
        }
    }

    /// Test helper: reads from the table `table` in our mock DB and returns an element that has (serialized) `column` equal to `value`.
    /// Returns None if no such element is found.
    #[cfg(test)]
    #[expect(clippy::missing_panics_doc)]
    pub async fn read(&self, table: &str, column: &str, value: &str) -> Option<serde_json::Value> {
        match self {
            Self::Disabled => None,
            Self::Mock { mock_data, .. } => {
                let mock_data = mock_data.read().await;
                let table = mock_data.get(table).unwrap();
                for row in table {
                    if let Some(value_in_row) = row.get(column) {
                        if value_in_row.as_str() == Some(value) {
                            return Some(row.clone());
                        }
                    }
                }
                None
            }
            Self::Production { .. } => {
                panic!("Production ClickHouse client can't be used for reading data in tests")
            }
        }
    }

    pub fn is_cluster_configured(&self) -> bool {
        match self {
            Self::Disabled => false,
            Self::Mock { .. } => false,
            Self::Production { cluster_name, .. } => cluster_name.is_some(),
        }
    }

    /// Runs a query with the given parameters, waiting for mutations to complete
    /// using `mutations_sync=2` and `alter_sync=2`.
    /// This ensures that we can run `ALTER TABLE ADD COLUMN` in a migration
    /// and have the column available once the query completes.
    pub async fn run_query_synchronous(
        &self,
        query: String,
        parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, Error> {
        self.run_query_synchronous_with_err_logging(query, parameters, true)
            .await
    }

    pub async fn run_query_synchronous_with_err_logging(
        &self,
        query: String,
        parameters: &HashMap<&str, &str>,
        err_logging: bool,
    ) -> Result<ClickHouseResponse, Error> {
        match self {
            Self::Disabled => Ok(ClickHouseResponse {
                response: String::new(),
                metadata: ClickHouseResponseMetadata {
                    read_rows: 0,
                    written_rows: 0,
                },
            }),
            Self::Mock { .. } => Ok(ClickHouseResponse {
                response: String::new(),
                metadata: ClickHouseResponseMetadata {
                    read_rows: 0,
                    written_rows: 0,
                },
            }),
            Self::Production {
                database_url,
                client,
                ..
            } => {
                let mut database_url = Url::parse(database_url.expose_secret()).map_err(|e| Error::new(ErrorDetails::ClickHouseQuery { message: format!("Error parsing ClickHouse URL: {e}. This should never happen. Please submit a bug report at https://github.com/tensorzero/tensorzero/issues/new") }))?;
                // Add query parameters if provided
                for (key, value) in parameters {
                    let param_key = format!("param_{key}");
                    database_url
                        .query_pairs_mut()
                        .append_pair(&param_key, value);
                }
                database_url
                    .query_pairs_mut()
                    .append_pair("mutations_sync", "2");
                database_url
                    .query_pairs_mut()
                    .append_pair("alter_sync", "2");
                let res = client
                    .post(database_url)
                    .body(query)
                    .send()
                    .await
                    .map_err(|e| {
                        Error::new_with_err_logging(
                            ErrorDetails::ClickHouseQuery {
                                message: DisplayOrDebugGateway::new(e).to_string(),
                            },
                            err_logging,
                        )
                    })?;
                let status = res.status();

                // Get the ClickHouse summary info from the headers
                let metadata = if let Some(summary) = res.headers().get("x-clickhouse-summary") {
                    // NOTE: X-Clickhouse-Summary is a ClickHouse-specific header that contains information about the query execution.
                    // It is not formally specified in the ClickHouse documentation so we only warn if it isn't working but won't error here.
                    let summary_str = summary.to_str().map_err(|e| {
                        Error::new_with_err_logging(
                            ErrorDetails::ClickHouseQuery {
                                message: format!(
                                    "Failed to parse x-clickhouse-summary header: {e}"
                                ),
                            },
                            err_logging,
                        )
                    })?;

                    serde_json::from_str::<ClickHouseResponseMetadata>(summary_str).map_err(
                        |e| {
                            Error::new_with_err_logging(
                                ErrorDetails::ClickHouseQuery {
                                    message: format!(
                                        "Failed to deserialize x-clickhouse-summary: {e}"
                                    ),
                                },
                                err_logging,
                            )
                        },
                    )?
                } else {
                    tracing::warn!("No x-clickhouse-summary header found in ClickHouse response");
                    ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    }
                };

                let response_body = res.text().await.map_err(|e| {
                    Error::new_with_err_logging(
                        ErrorDetails::ClickHouseQuery {
                            message: DisplayOrDebugGateway::new(e).to_string(),
                        },
                        err_logging,
                    )
                })?;

                match status {
                    reqwest::StatusCode::OK => Ok(ClickHouseResponse {
                        response: response_body,
                        metadata,
                    }),
                    _ => Err(Error::new_with_err_logging(
                        ErrorDetails::ClickHouseQuery {
                            message: response_body,
                        },
                        err_logging,
                    )),
                }
            }
        }
    }

    // TODO: deprecate this
    pub async fn run_query_synchronous_no_params(
        &self,
        query: String,
    ) -> Result<ClickHouseResponse, Error> {
        self.run_query_synchronous(query, &HashMap::default()).await
    }

    pub async fn run_query_synchronous_no_params_de<T>(&self, query: String) -> Result<T, Error>
    where
        T: DeserializeOwned,
    {
        let result = self
            .run_query_synchronous(query, &HashMap::default())
            .await?;
        serde_json::from_str(&result.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })
    }

    /// Sometimes you might want to treat the data you're sending as a table if you're going
    /// to do some analysis or filtering prior to inserting it into ClickHouse.
    /// This function allows you to do this with ClickHouse's external data feature.
    /// https://clickhouse.com/docs/engines/table-engines/special/external-data
    pub async fn run_query_with_external_data(
        &self,
        external_data: ExternalDataInfo,
        query: String,
    ) -> Result<ClickHouseResponse, Error> {
        match self {
            Self::Disabled | Self::Mock { .. } => Ok(ClickHouseResponse {
                response: String::new(),
                metadata: ClickHouseResponseMetadata {
                    read_rows: 0,
                    written_rows: 0,
                },
            }),
            Self::Production {
                database_url,
                client,
                ..
            } => {
                let database_url = Url::parse(database_url.expose_secret()).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: "Invalid ClickHouse database URL".to_string(),
                    })
                })?;
                // Create the multipart form
                let form = Form::new()
                    .text("new_data_structure", external_data.structure)
                    .text("new_data_format", external_data.format)
                    .part(
                        "new_data",
                        Part::bytes(external_data.data.into_bytes()).file_name("file.data"),
                    )
                    .text("query", query);

                let res = client
                    .post(database_url)
                    .multipart(form)
                    .send()
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseQuery {
                            message: e.to_string(),
                        })
                    })?;

                let status = res.status();
                // Get the ClickHouse summary info from the headers
                let metadata = if let Some(summary) = res.headers().get("x-clickhouse-summary") {
                    let summary_str = summary.to_str().map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseQuery {
                            message: format!("Failed to parse x-clickhouse-summary header: {e}"),
                        })
                    })?;

                    serde_json::from_str::<ClickHouseResponseMetadata>(summary_str).map_err(
                        |e| {
                            Error::new(ErrorDetails::ClickHouseQuery {
                                message: format!("Failed to deserialize x-clickhouse-summary: {e}"),
                            })
                        },
                    )?
                } else {
                    tracing::warn!("No x-clickhouse-summary header found in ClickHouse response");
                    ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    }
                };

                let response_body = res.text().await.map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: e.to_string(),
                    })
                })?;

                match status {
                    reqwest::StatusCode::OK => Ok(ClickHouseResponse {
                        response: response_body,
                        metadata,
                    }),
                    _ => Err(Error::new(ErrorDetails::ClickHouseQuery {
                        message: response_body,
                    })),
                }
            }
        }
    }

    pub async fn check_database_and_migrations_table_exists(&self) -> Result<bool, Error> {
        match self {
            Self::Disabled => Ok(true),
            Self::Mock { .. } => Ok(true),
            Self::Production {
                client,
                database_url,
                ..
            } => {
                let database_url = Url::parse(database_url.expose_secret()).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: "Invalid ClickHouse database URL".to_string(),
                    })
                })?;
                let mut base_url = database_url.clone();
                let query_pairs = database_url
                    .query_pairs()
                    .filter(|(key, _)| key != "database");
                base_url
                    .query_pairs_mut()
                    .clear()
                    .extend_pairs(query_pairs)
                    .append_pair("param_name", self.database())
                    .finish();
                let query =
                    "SELECT COUNT() FROM system.databases WHERE name={name:String}".to_string();
                let response = client
                    .post(base_url)
                    .body(query)
                    .send()
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseQuery {
                            message: e.to_string(),
                        })
                    })?;
                let text = response.text().await.map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: format!("Failed to fetch response text: {e}"),
                    })
                })?;
                let count: u8 = text.trim().parse().map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: format!("Failed to parse count response as u8: {e}"),
                    })
                })?;
                if count == 0 {
                    // The database doesn't exist
                    return Ok(false);
                }
                let migrations_table_exists =
                    check_table_exists(self, "TensorZeroMigration", "0000").await?;

                Ok(migrations_table_exists)
            }
        }
    }

    pub async fn create_database_and_migrations_table(&self) -> Result<(), Error> {
        match self {
            Self::Disabled => {}
            Self::Mock { .. } => {}
            Self::Production {
                database_url,
                database,
                client,
                ..
            } => {
                let database_url = Url::parse(database_url.expose_secret()).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: "Invalid ClickHouse database URL".to_string(),
                    })
                })?;
                let on_cluster_name = self.get_on_cluster_name();
                let query = format!("CREATE DATABASE IF NOT EXISTS {database}{on_cluster_name}");
                // In order to create the database, we need to remove the database query parameter from the URL
                // Otherwise, ClickHouse will throw an error
                let mut base_url = database_url.clone();
                let query_pairs = database_url
                    .query_pairs()
                    .filter(|(key, _)| key != "database");
                base_url
                    .query_pairs_mut()
                    .clear()
                    .extend_pairs(query_pairs)
                    .finish();

                let response = client
                    .post(base_url)
                    .body(query)
                    .send()
                    .await
                    .map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseQuery {
                            message: e.to_string(),
                        })
                    })?;

                let status = response.status();

                let response_body = response.text().await.map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: e.to_string(),
                    })
                })?;

                match status {
                    reqwest::StatusCode::OK => {}
                    _ => {
                        return Err(Error::new(ErrorDetails::ClickHouseQuery {
                            message: response_body,
                        }))
                    }
                }
            }
        }
        // Note - we do *not* run this as a normal migration
        // We decided to add this table after we had already created lots of migrations.
        // We create this table immediately after creating the database, so that
        // we can insert rows into it when running migrations
        let on_cluster_name = self.get_on_cluster_name();
        let table_engine_name =
            self.get_maybe_replicated_table_engine_name(GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "TensorZeroMigration",
                engine_args: &[],
            });
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS TensorZeroMigration{on_cluster_name} (
                migration_id UInt32,
                migration_name String,
                gateway_version String,
                gateway_git_sha String,
                applied_at DateTime64(6, 'UTC') DEFAULT now(),
                execution_time_ms UInt64,
                extra_data Nullable(String)
            )
            ENGINE = {table_engine_name}
            PRIMARY KEY (migration_id)"
        );
        self.run_query_synchronous_no_params(query)
            .await
            .map(|_| ())?;
        Ok(())
    }

    pub async fn list_inferences(
        &self,
        config: &Config,
        opts: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInference>, Error> {
        let (sql, params) = generate_list_inferences_sql(config, opts)?;
        let params_map = params
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();
        let response = self.run_query_synchronous(sql, &params_map).await?;
        let inferences = response
            .response
            .trim()
            .lines()
            .map(|line| {
                serde_json::from_str::<query_builder::ClickHouseStoredInference>(line)
                    .map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseQuery {
                            message: format!("Failed to deserialize response: {e:?}"),
                        })
                    })
                    .and_then(query_builder::ClickHouseStoredInference::try_into)
            })
            .collect::<Result<Vec<StoredInference>, Error>>()?;
        Ok(inferences)
    }

    pub fn get_on_cluster_name(&self) -> String {
        match self {
            Self::Disabled => String::new(),
            Self::Mock { .. } => String::new(),
            Self::Production { cluster_name, .. } => match cluster_name {
                Some(cluster_name) => format!(" ON CLUSTER {cluster_name} "),
                None => String::new(),
            },
        }
    }

    pub fn get_maybe_replicated_table_engine_name(
        &self,
        args: GetMaybeReplicatedTableEngineNameArgs<'_>,
    ) -> String {
        let GetMaybeReplicatedTableEngineNameArgs {
            table_engine_name,
            table_name,
            engine_args,
        } = args;
        match self {
            Self::Disabled => table_engine_name.to_string(),
            Self::Mock { .. } => table_engine_name.to_string(),
            Self::Production {
                cluster_name,
                database,
                ..
            } => match cluster_name {
                Some(_) => get_replicated_table_engine_name(
                    table_engine_name,
                    table_name,
                    database,
                    engine_args,
                ),
                None => {
                    let engine_args_str = engine_args.join(", ");
                    format!("{table_engine_name}({engine_args_str})")
                }
            },
        }
    }
}

#[async_trait]
impl HealthCheckable for ClickHouseConnectionInfo {
    async fn health(&self) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Mock { healthy, .. } => {
                if *healthy {
                    Ok(())
                } else {
                    Err(ErrorDetails::ClickHouseConnection {
                        message: "Mock ClickHouse is not healthy".to_string(),
                    }
                    .into())
                }
            }
            Self::Production {
                database_url,
                client,
                ..
            } => {
                // We need to ping the /ping endpoint to check if ClickHouse is healthy
                let mut ping_url = Url::parse(database_url.expose_secret()).map_err(|_| {
                    Error::new(ErrorDetails::Config {
                        message: "Invalid ClickHouse database URL".to_string(),
                    })
                })?;
                ping_url.set_path("/ping");
                ping_url.set_query(None);

                let timeout = Duration::from_secs(180);

                match client.get(ping_url).timeout(timeout).send().await {
                    Ok(response) if response.status().is_success() => Ok(()),
                    Ok(response) => Err(ErrorDetails::ClickHouseConnection {
                        message: format!(
                            "ClickHouse is not healthy (status code {}): {}",
                            response.status(),
                            response.text().await.unwrap_or_default()
                        ),
                    }
                    .into()),
                    Err(e) => Err(ErrorDetails::ClickHouseConnection {
                        message: format!("ClickHouse is not healthy: {e:?}"),
                    }
                    .into()),
                }
            }
        }
    }
}

pub struct GetMaybeReplicatedTableEngineNameArgs<'a> {
    pub table_engine_name: &'a str,
    pub table_name: &'a str,
    pub engine_args: &'a [&'a str],
}

/// The ClickHouse documentation says that to create a replicated table,
/// you should use a Replicated* table engine.
/// The first 2 arguments are the keeper path and the replica name.
/// The following arguments must be the arguments that the table engine takes.
/// See https://clickhouse.com/docs/engines/table-engines/mergetree-family/replication for more details.
///
/// Since there may be issues with renaming tables, we don't use the database or table name shortcuts in the path.
/// This method requires that the macros for {{shard}} and {{replica}} are defined in the ClickHouse configuration.
/// This method should only be called if a cluster name is provided.
fn get_replicated_table_engine_name(
    table_engine_name: &str,
    table_name: &str,
    database: &str,
    engine_args: &[&str],
) -> String {
    let keeper_path = format!("'/clickhouse/tables/{{shard}}/{database}/{table_name}'");
    if engine_args.is_empty() {
        format!("Replicated{table_engine_name}({keeper_path}, '{{replica}}')")
    } else {
        let engine_args_str = engine_args.join(", ");
        format!("Replicated{table_engine_name}({keeper_path}, '{{replica}}', {engine_args_str})")
    }
}

/// ClickHouse uses backslashes to escape quotes and all other special sequences in strings.
/// In certain cases, we'll need to use a string as a literal in a ClickHouse query.
/// e.g. to compare a raw string containing user input to strings in the database.
/// These may contain single quotes and backslashes, for example, if the user input contains doubly-serialized JSON.
/// This function will escape single quotes and backslashes in the input string so that the comparison will be accurate.
pub fn escape_string_for_clickhouse_literal(s: &str) -> String {
    #![expect(clippy::needless_raw_string_hashes)]
    s.replace(r#"\"#, r#"\\"#).replace(r#"'"#, r#"\'"#)
}

async fn write_mock<T: Serialize + Send + Sync>(
    rows: Rows<'_, T>,
    table: TableName,
    tables: &mut RwLockWriteGuard<'_, HashMap<String, Vec<serde_json::Value>>>,
) -> Result<(), Error> {
    for row in rows.as_json()?.iter() {
        tables
            .entry(table.as_str().to_string())
            .or_default()
            .push(serde_json::Value::String(row.clone()));
    }
    Ok(())
}

/// A wrapper type used with `write_non_batched`
pub enum Rows<'a, T: Serialize + Send + Sync> {
    /// The rows will be serialized to JSON before being sent to ClickHouse
    Unserialized(&'a [T]),
    /// The rows are already serialized to JSON, and will be sent to ClickHouse as-is
    Serialized(&'a [String]),
}

impl<T: Serialize + Send + Sync> Rows<'_, T> {
    fn as_json(&self) -> Result<Cow<'_, [String]>, Error> {
        match self {
            Rows::Unserialized(rows) => {
                let json: Result<Vec<String>, _> = rows
                    .iter()
                    .map(|row| {
                        serde_json::to_string(row).map_err(|e| {
                            Error::new(ErrorDetails::Serialization {
                                message: e.to_string(),
                            })
                        })
                    })
                    .collect();
                json.map(Cow::Owned)
            }
            Rows::Serialized(rows) => Ok(Cow::Borrowed(rows)),
        }
    }
}

async fn write_production<T: Serialize + Send + Sync>(
    database_url: &SecretString,
    client: &Client,
    rows: Rows<'_, T>,
    table: TableName,
    batch: Option<&BatchSender>,
) -> Result<(), Error> {
    let is_empty = match &rows {
        Rows::Unserialized(rows) => rows.is_empty(),
        Rows::Serialized(rows) => rows.is_empty(),
    };
    // Empty rows is a no-op
    if is_empty {
        return Ok(());
    }
    let rows_json = rows.as_json();

    if let Some(batch_sender) = batch {
        batch_sender
            .add_to_batch(table, rows_json?.into_owned())
            .await?;
        return Ok(());
    }

    let rows_json = rows_json?.join("\n");
    let table = table.as_str();

    // We can wait for the async insert since we're spawning a new tokio task to do the insert
    let query = format!(
        "INSERT INTO {table}\n\
        SETTINGS async_insert=1, wait_for_async_insert=1\n\
        FORMAT JSONEachRow\n\
        {rows_json}"
    );

    let response = client
        .post(database_url.expose_secret())
        .body(query)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::ClickHouseQuery {
                message: format!("{e:?}"),
            })
        })?;

    match response.status() {
        reqwest::StatusCode::OK => Ok(()),
        _ => Err(Error::new(ErrorDetails::ClickHouseQuery {
            message: response
                .text()
                .await
                .unwrap_or_else(|e| format!("Failed to get response text: {e}")),
        })),
    }
}

fn set_clickhouse_format_settings(database_url: &mut Url) {
    const OVERRIDDEN_SETTINGS: [(&str, &str); 3] = [
        ("input_format_skip_unknown_fields", "0"),
        ("input_format_null_as_default", "0"),
        ("output_format_json_quote_64bit_integers", "1"),
    ];

    let existing_pairs: Vec<(String, String)> = database_url
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();

    database_url.query_pairs_mut().clear();

    for (key, value) in existing_pairs {
        if OVERRIDDEN_SETTINGS
            .iter()
            .any(|(setting_key, _)| *setting_key == key.as_str())
        {
            tracing::warn!(
                "Your ClickHouse connection URL has the setting '{}' but it will be overridden.",
                key
            );
        } else {
            database_url.query_pairs_mut().append_pair(&key, &value);
        }
    }

    for (setting_key, setting_value) in &OVERRIDDEN_SETTINGS {
        database_url
            .query_pairs_mut()
            .append_pair(setting_key, setting_value);
    }
}

#[cfg(any(not(feature = "e2e_tests"), test))]
fn validate_clickhouse_url_get_db_name(url: &Url) -> Result<Option<String>, Error> {
    // Check the scheme
    match url.scheme() {
        "http" | "https" => {}
        "clickhouse" | "clickhousedb" => {
            return Err(ErrorDetails::Config {
                message: format!(
                    "Invalid scheme in ClickHouse URL: '{}'. Use 'http' or 'https' instead.",
                    url.scheme()
                ),
            }
            .into());
        }
        _ => {
            return Err(ErrorDetails::Config {
                message: format!(
                "Invalid scheme in ClickHouse URL: '{}'. Only 'http' and 'https' are supported.",
                url.scheme()
            ),
            }
            .into())
        }
    }

    // Validate the host
    if url.host().is_none() {
        return Err(ErrorDetails::Config {
            message: "Missing hostname in ClickHouse URL".to_string(),
        }
        .into());
    }

    // Validate that none of the query strings have key "database"
    if url.query_pairs().any(|(key, _)| key == "database") {
        return Err(ErrorDetails::Config {
            message: "The query string 'database' is not allowed in the ClickHouse URL".to_string(),
        }
        .into());
    }
    // username, password, and query-strings are optional, so we don't need to validate them

    // Validate that the path is either empty or ends with the database name (a single segment)
    let mut path_segments: Vec<_> = url
        .path_segments()
        .map(Iterator::collect)
        .unwrap_or_default();
    if let Some(last) = path_segments.last() {
        if last.is_empty() {
            path_segments.pop();
        }
    }
    Ok(match path_segments.len() {
        0 => None, // Empty path is valid
        1 => {
            if path_segments[0].is_empty() {
                return Err(ErrorDetails::Config {
                    message: "The database name in the path of the ClickHouse URL cannot be empty".to_string(),
                }
                .into());
            }
            Some(path_segments[0].to_string())
        }
        _ => return Err(ErrorDetails::Config {
            message: "The path of the ClickHouse URL must be of length 0 or 1, and end with the database name if set".to_string(),
        }
        .into()),
    })
}

#[derive(Debug)]
pub struct ExternalDataInfo {
    pub external_data_name: String, // The name of the external data table that was used in the query
    pub structure: String, // Must be a ClickHouse structure string, e.g. "id UInt32, name String"
    pub format: String,    // Must be a ClickHouse format string, e.g. "JSONEachRow"
    pub data: String,      // Must be valid ClickHouse data in the given format
}

#[derive(Debug)]
pub struct ClickHouseResponse {
    pub response: String,
    pub metadata: ClickHouseResponseMetadata,
}

#[derive(Debug, Deserialize)]
pub struct ClickHouseResponseMetadata {
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_u64_from_str")]
    pub read_rows: u64,
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_u64_from_str")]
    pub written_rows: u64,
}

fn deserialize_u64_from_str<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<u64>().map_err(serde::de::Error::custom)
}

/// The format of the data that will be returned from / sent to ClickHouse.
/// Currently only used in the query builder.
/// TODO: use across the codebase.
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub enum ClickhouseFormat {
    #[default]
    JsonEachRow,
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_set_clickhouse_format_settings() {
        let mut database_url = Url::parse("http://chuser:chpassword@localhost:8123/").unwrap();
        set_clickhouse_format_settings(&mut database_url);
        assert_eq!(database_url.to_string(), "http://chuser:chpassword@localhost:8123/?input_format_skip_unknown_fields=0&input_format_null_as_default=0&output_format_json_quote_64bit_integers=1");

        let mut database_url = Url::parse("http://chuser:chpassword@localhost:8123/?input_format_skip_unknown_fields=1&input_format_defaults_for_omitted_fields=1&input_format_null_as_default=1").unwrap();
        set_clickhouse_format_settings(&mut database_url);
        assert_eq!(database_url.to_string(), "http://chuser:chpassword@localhost:8123/?input_format_defaults_for_omitted_fields=1&input_format_skip_unknown_fields=0&input_format_null_as_default=0&output_format_json_quote_64bit_integers=1");
    }

    #[test]
    fn test_validate_clickhouse_url() {
        let database_url = Url::parse("http://chuser:chpassword@localhost:8123/").unwrap();
        let result = validate_clickhouse_url_get_db_name(&database_url).unwrap();
        assert_eq!(result, None);

        let database_url = Url::parse("clickhouse://localhost:8123/").unwrap();
        let err = validate_clickhouse_url_get_db_name(&database_url).unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::Config {
                message:
                    "Invalid scheme in ClickHouse URL: 'clickhouse'. Use 'http' or 'https' instead."
                        .to_string(),
            }
            .into()
        );

        let database_url = Url::parse("https://localhost:8123/").unwrap();
        let result = validate_clickhouse_url_get_db_name(&database_url).unwrap();
        assert_eq!(result, None);

        let database_url =
            Url::parse("http://username:password@localhost:8123/database?k=v").unwrap();
        let result = validate_clickhouse_url_get_db_name(&database_url).unwrap();
        assert_eq!(result, Some("database".to_string()));

        let database_url = Url::parse("https://localhost:443/").unwrap();
        assert!(validate_clickhouse_url_get_db_name(&database_url)
            .unwrap()
            .is_none());

        let database_url = Url::parse("http://default:password@clickhouse.cloud.io:443").unwrap();
        assert!(validate_clickhouse_url_get_db_name(&database_url)
            .unwrap()
            .is_none());

        let database_url = Url::parse("http://localhost:8123").unwrap();
        assert!(validate_clickhouse_url_get_db_name(&database_url).is_ok());

        let database_url =
            Url::parse("http://chuser:chpassword@localhost:8123/?database=tensorzero_e2e_tests")
                .unwrap();
        let err = validate_clickhouse_url_get_db_name(&database_url).unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::Config {
                message: "The query string 'database' is not allowed in the ClickHouse URL"
                    .to_string(),
            }
            .into()
        );

        let database_url =
            Url::parse("http://chuser:chpassword@localhost:8123/database/tensorzero_e2e_tests")
                .unwrap();
        let err = validate_clickhouse_url_get_db_name(&database_url).unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::Config {
                message: "The path of the ClickHouse URL must be of length 0 or 1, and end with the database name if set".to_string(),
            }
            .into()
        );

        let database_url =
            Url::parse("http://chuser:chpassword@localhost:8123/database?foo=bar").unwrap();
        let database = validate_clickhouse_url_get_db_name(&database_url).unwrap();
        assert_eq!(database, Some("database".to_string()));

        let database_url = Url::parse("http://chuser:chpassword@localhost:8123/database/").unwrap();
        let database = validate_clickhouse_url_get_db_name(&database_url).unwrap();
        assert_eq!(database, Some("database".to_string()));
    }

    #[test]
    fn test_escape_string_for_clickhouse_comparison() {
        // Test basic escaping of single quotes
        #![expect(clippy::needless_raw_string_hashes)]
        assert_eq!(
            escape_string_for_clickhouse_literal("test's string"),
            r#"test\'s string"#
        );

        // Test basic escaping of backslashes
        assert_eq!(
            escape_string_for_clickhouse_literal(r#"test\string"#),
            r#"test\\string"#
        );

        // Test escaping of both single quotes and backslashes
        assert_eq!(
            escape_string_for_clickhouse_literal(r#"test\'s string"#),
            r#"test\\\'s string"#
        );

        // Test with JSON-like content that has escaped quotes
        assert_eq!(
            escape_string_for_clickhouse_literal(r#"{"key":"value with a \", and a '"}"#),
            r#"{"key":"value with a \\", and a \'"}"#
        );

        // Test with empty string
        assert_eq!(escape_string_for_clickhouse_literal(""), "");

        // Test with multiple backslashes and quotes
        assert_eq!(
            escape_string_for_clickhouse_literal(r#"\\\'test\'\\"#),
            r#"\\\\\\\'test\\\'\\\\"#
        );

        // Test with alternating backslashes and quotes
        assert_eq!(
            escape_string_for_clickhouse_literal(r#"\'\'\'"#),
            r#"\\\'\\\'\\\'"#
        );
    }

    #[test]
    fn test_get_replicated_table_engine_name_basic() {
        let result = get_replicated_table_engine_name("MergeTree", "test_table", "test_db", &[]);
        assert_eq!(
            result,
            "ReplicatedMergeTree('/clickhouse/tables/{shard}/test_db/test_table', '{replica}')"
        );
    }

    #[test]
    fn test_get_replicated_table_engine_name_with_single_arg() {
        let result =
            get_replicated_table_engine_name("MergeTree", "users", "analytics", &["ORDER BY id"]);
        assert_eq!(
            result,
            "ReplicatedMergeTree('/clickhouse/tables/{shard}/analytics/users', '{replica}', ORDER BY id)"
        );
    }

    #[test]
    fn test_get_replicated_table_engine_name_with_multiple_args() {
        let result = get_replicated_table_engine_name(
            "ReplacingMergeTree",
            "events",
            "production",
            &["version_column", "ORDER BY (timestamp, user_id)"],
        );
        assert_eq!(
            result,
            "ReplicatedReplacingMergeTree('/clickhouse/tables/{shard}/production/events', '{replica}', version_column, ORDER BY (timestamp, user_id))"
        );
    }

    #[test]
    fn test_get_replicated_table_engine_name_with_complex_names() {
        let result = get_replicated_table_engine_name(
            "CollapsingMergeTree",
            "user_activity_log",
            "metrics_database",
            &["sign_column"],
        );
        assert_eq!(
            result,
            "ReplicatedCollapsingMergeTree('/clickhouse/tables/{shard}/metrics_database/user_activity_log', '{replica}', sign_column)"
        );
    }

    #[test]
    fn test_get_replicated_table_engine_name_empty_args() {
        let result = get_replicated_table_engine_name("Log", "simple_table", "simple_db", &[]);
        assert_eq!(
            result,
            "ReplicatedLog('/clickhouse/tables/{shard}/simple_db/simple_table', '{replica}')"
        );
    }

    #[test]
    fn test_get_replicated_table_engine_name_special_characters() {
        let result = get_replicated_table_engine_name(
            "MergeTree",
            "table_with_underscores",
            "db-with-dashes",
            &["'primary_key'", "PARTITION BY toYYYYMM(date)"],
        );
        assert_eq!(
            result,
            "ReplicatedMergeTree('/clickhouse/tables/{shard}/db-with-dashes/table_with_underscores', '{replica}', 'primary_key', PARTITION BY toYYYYMM(date))"
        );
    }
}
