use async_trait::async_trait;
use http::{HeaderMap, HeaderValue};
use reqwest::multipart::Form;
use reqwest::multipart::Part;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

use crate::config::BatchWritesConfig;
use crate::db::clickhouse::batching::BatchSender;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::migration_manager::migrations::check_table_exists;
use crate::db::clickhouse::BatchWriterHandle;
use crate::db::clickhouse::ClickHouseClient;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::ClickHouseResponse;
use crate::db::clickhouse::ClickHouseResponseMetadata;
use crate::db::clickhouse::ExternalDataInfo;
use crate::db::clickhouse::GetMaybeReplicatedTableEngineNameArgs;
use crate::db::clickhouse::HealthCheckable;
use crate::db::clickhouse::Rows;
use crate::db::clickhouse::TableName;
use crate::error::DelayedError;
use crate::error::DisplayOrDebugGateway;
use crate::error::Error;
use crate::error::ErrorDetails;

/// Production implementation of ClickHouseClient
#[derive(Debug, Clone)]
pub struct ProductionClickHouseClient {
    database_url: SecretString,
    sanitized_database_url: String,
    cluster_name: Option<String>,
    database: String,
    client: Client,
    batch_sender: Option<Arc<BatchSender>>,
}

impl ProductionClickHouseClient {
    /// Create a new production ClickHouse client from a database URL.
    pub(crate) async fn new(
        database_url: SecretString,
        cluster_name: Option<String>,
        database: String,
        batch_config: BatchWritesConfig,
    ) -> Result<Self, Error> {
        let parsed_database_url = Url::parse(database_url.expose_secret()).map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("Invalid ClickHouse database URL: {e}"),
            })
        })?;

        let username = if parsed_database_url.username().is_empty() {
            None
        } else {
            Some(
                urlencoding::decode(parsed_database_url.username())
                    .map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!("Failed to decode ClickHouse username from URL: {e}"),
                        })
                    })?
                    .into_owned(),
            )
        };

        let password = match parsed_database_url.password() {
            Some(password) => Some(SecretString::from(
                urlencoding::decode(password)
                    .map_err(|e| {
                        Error::new(ErrorDetails::Config {
                            message: format!("Failed to decode ClickHouse password from URL: {e}"),
                        })
                    })?
                    .into_owned(),
            )),
            None => None,
        };

        let mut sanitized_database_url = parsed_database_url.clone();
        if !parsed_database_url.username().is_empty() {
            sanitized_database_url.set_username("").map_err(|()| {
                Error::new(ErrorDetails::Config {
                    message: "Failed to sanitize ClickHouse URL username".to_string(),
                })
            })?;
        }
        if parsed_database_url.password().is_some() {
            sanitized_database_url.set_password(None).map_err(|()| {
                Error::new(ErrorDetails::Config {
                    message: "Failed to sanitize ClickHouse URL password".to_string(),
                })
            })?;
        }

        let mut client = Self {
            database_url,
            sanitized_database_url: sanitized_database_url.to_string(),
            cluster_name,
            database,
            client: make_clickhouse_http_client(username, password)?,
            batch_sender: None,
        };

        // Create the batch sender if enabled
        if batch_config.enabled {
            // Create a temporary ClickHouseConnectionInfo without the batch_sender
            // (since the batcher itself always performs direct writes)
            let temp_connection_info = ClickHouseConnectionInfo {
                inner: Arc::new(client.clone()),
            };
            let batcher = BatchSender::new(temp_connection_info, batch_config)?;
            client.batch_sender = Some(Arc::new(batcher));
        }

        // If the connection is unhealthy, we won't be able to run / check migrations. So we just fail here.
        let temp_connection_info = ClickHouseConnectionInfo {
            inner: Arc::new(client.clone()),
        };
        temp_connection_info.inner.health().await?;
        Ok(client)
    }
}

fn make_clickhouse_http_client(
    username: Option<String>,
    password: Option<SecretString>,
) -> Result<Client, Error> {
    let mut headers = HeaderMap::new();
    if let Some(username) = username.as_ref() {
        headers.insert(
        "X-ClickHouse-User",
        HeaderValue::from_str(username).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseConnection {
                message: format!("Failed to build ClickHouse HTTP client because username contains invalid bytes: {e}"),
            })
        })?,
    );
    }
    if let Some(password) = password {
        let mut password_header_value = HeaderValue::from_str(password.expose_secret()).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseConnection {
                message: format!("Failed to build ClickHouse HTTP client because password contains invalid bytes: {e}"),
            })
        })?;
        password_header_value.set_sensitive(true);
        headers.insert("X-ClickHouse-Key", password_header_value);
    }
    Client::builder()
        .default_headers(headers)
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

// Trait implementations for ProductionClickHouseClient
#[async_trait]
impl ClickHouseClient for ProductionClickHouseClient {
    fn database_url(&self) -> &SecretString {
        &self.database_url
    }

    fn cluster_name(&self) -> &Option<String> {
        &self.cluster_name
    }

    fn database(&self) -> &str {
        &self.database
    }

    fn is_batching_enabled(&self) -> bool {
        self.batch_sender.is_some()
    }

    fn batcher_join_handle(&self) -> Option<BatchWriterHandle> {
        self.batch_sender.as_ref().map(|s| s.writer_handle.clone())
    }

    async fn write_batched_internal(
        &self,
        rows: Vec<String>,
        table: TableName,
    ) -> Result<(), Error> {
        write_production(
            self,
            Rows::<String>::Serialized(&rows),
            table,
            self.batch_sender.as_deref(),
        )
        .await
    }

    async fn write_non_batched_internal(
        &self,
        rows: Vec<String>,
        table: TableName,
    ) -> Result<(), Error> {
        write_production(self, Rows::<String>::Serialized(&rows), table, None).await
    }

    async fn run_query_synchronous(
        &self,
        query: String,
        parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, Error> {
        self.run_query_synchronous_delayed_err(query, parameters)
            .await
            .map_err(|e| e.log())
    }

    async fn run_query_synchronous_delayed_err(
        &self,
        query: String,
        parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, DelayedError> {
        let mut database_url = Url::parse(&self.sanitized_database_url).map_err(|e| DelayedError::new(ErrorDetails::ClickHouseQuery { message: format!("Error parsing ClickHouse URL: {e}. This should never happen. Please submit a bug report at https://github.com/tensorzero/tensorzero/issues/new") }))?;
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
        database_url
            .query_pairs_mut()
            .append_pair("join_algorithm", "auto");
        let res = self
            .client
            .post(database_url)
            .body(query)
            .send()
            .await
            .map_err(|e| {
                DelayedError::new(ErrorDetails::ClickHouseQuery {
                    message: DisplayOrDebugGateway::new(e).to_string(),
                })
            })?;
        let status = res.status();

        // Get the ClickHouse summary info from the headers
        let metadata = if let Some(summary) = res.headers().get("x-clickhouse-summary") {
            // NOTE: X-Clickhouse-Summary is a ClickHouse-specific header that contains information about the query execution.
            // It is not formally specified in the ClickHouse documentation so we only warn if it isn't working but won't error here.
            let summary_str = summary.to_str().map_err(|e| {
                DelayedError::new(ErrorDetails::ClickHouseQuery {
                    message: format!("Failed to parse x-clickhouse-summary header: {e}"),
                })
            })?;

            serde_json::from_str::<ClickHouseResponseMetadata>(summary_str).map_err(|e| {
                DelayedError::new(ErrorDetails::ClickHouseQuery {
                    message: format!("Failed to deserialize x-clickhouse-summary: {e}"),
                })
            })?
        } else {
            tracing::warn!("No x-clickhouse-summary header found in ClickHouse response");
            ClickHouseResponseMetadata {
                read_rows: 0,
                written_rows: 0,
            }
        };

        let response_body = res.text().await.map_err(|e| {
            DelayedError::new(ErrorDetails::ClickHouseQuery {
                message: DisplayOrDebugGateway::new(e).to_string(),
            })
        })?;

        match status {
            reqwest::StatusCode::OK => Ok(ClickHouseResponse {
                response: response_body,
                metadata,
            }),
            _ => Err(DelayedError::new(ErrorDetails::ClickHouseQuery {
                message: response_body,
            })),
        }
    }

    async fn run_query_with_external_data(
        &self,
        external_data: ExternalDataInfo,
        query: String,
    ) -> Result<ClickHouseResponse, Error> {
        let database_url = Url::parse(&self.sanitized_database_url).map_err(|_| {
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

        let res = self
            .client
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

            serde_json::from_str::<ClickHouseResponseMetadata>(summary_str).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseQuery {
                    message: format!("Failed to deserialize x-clickhouse-summary: {e}"),
                })
            })?
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

    async fn check_database_and_migrations_table_exists(&self) -> Result<bool, Error> {
        let database_url = Url::parse(&self.sanitized_database_url).map_err(|_| {
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
            .append_pair("param_name", &self.database)
            .finish();
        let query = "SELECT COUNT() FROM system.databases WHERE name={name:String}".to_string();
        let response = self
            .client
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
        let text = response.text().await.map_err(|e| {
            Error::new(ErrorDetails::ClickHouseQuery {
                message: format!("Failed to fetch response text: {e}"),
            })
        })?;

        // Check if the request was successful before trying to parse the response
        if !status.is_success() {
            return Err(Error::new(ErrorDetails::ClickHouseConnection {
                message: format!(
                    "ClickHouse query failed with status {}: {}",
                    status.as_u16(),
                    text
                ),
            }));
        }

        let count: u8 = text.trim().parse().map_err(|e| {
            Error::new(ErrorDetails::ClickHouseQuery {
                message: format!("Failed to parse count response as u8: {e}"),
            })
        })?;
        if count == 0 {
            // The database doesn't exist
            return Ok(false);
        }

        // Create a temporary wrapper to call check_table_exists
        let temp_connection_info = ClickHouseConnectionInfo {
            inner: Arc::new(self.clone()),
        };
        let migrations_table_exists =
            check_table_exists(&temp_connection_info, "TensorZeroMigration", "0000").await?;

        Ok(migrations_table_exists)
    }

    async fn create_database_and_migrations_table(&self) -> Result<(), Error> {
        let database_url = Url::parse(&self.sanitized_database_url).map_err(|_| {
            Error::new(ErrorDetails::Config {
                message: "Invalid ClickHouse database URL".to_string(),
            })
        })?;
        let on_cluster_name = self.get_on_cluster_name();
        let query = format!(
            "CREATE DATABASE IF NOT EXISTS {}{on_cluster_name}",
            self.database
        );
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

        let response = self
            .client
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

        // Note - we do *not* run this as a normal migration
        // We decided to add this table after we had already created lots of migrations.
        // We create this table immediately after creating the database, so that
        // we can insert rows into it when running migrations
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

        // Create a temporary wrapper to call run_query_synchronous
        let temp_connection_info = ClickHouseConnectionInfo {
            inner: Arc::new(self.clone()),
        };
        temp_connection_info
            .run_query_synchronous_no_params(query)
            .await
            .map(|_| ())?;
        Ok(())
    }

    fn is_cluster_configured(&self) -> bool {
        self.cluster_name.is_some()
    }

    fn get_on_cluster_name(&self) -> String {
        match &self.cluster_name {
            Some(cluster_name) => format!(" ON CLUSTER {cluster_name} "),
            None => String::new(),
        }
    }

    fn get_maybe_replicated_table_engine_name(
        &self,
        args: GetMaybeReplicatedTableEngineNameArgs<'_>,
    ) -> String {
        let GetMaybeReplicatedTableEngineNameArgs {
            table_engine_name,
            table_name,
            engine_args,
        } = args;
        match &self.cluster_name {
            Some(_) => get_replicated_table_engine_name(
                table_engine_name,
                table_name,
                &self.database,
                engine_args,
            ),
            None => {
                let engine_args_str = engine_args.join(", ");
                format!("{table_engine_name}({engine_args_str})")
            }
        }
    }

    fn client_type(&self) -> ClickHouseClientType {
        ClickHouseClientType::Production
    }
}

#[async_trait]
impl HealthCheckable for ProductionClickHouseClient {
    async fn health(&self) -> Result<(), Error> {
        // We need to ping the /ping endpoint to check if ClickHouse is healthy
        let mut ping_url = Url::parse(&self.sanitized_database_url).map_err(|_| {
            Error::new(ErrorDetails::Config {
                message: "Invalid ClickHouse database URL".to_string(),
            })
        })?;
        ping_url.set_path("/ping");
        ping_url.set_query(None);

        let timeout = Duration::from_secs(180);

        match self.client.get(ping_url).timeout(timeout).send().await {
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

async fn write_production<T: Serialize + Send + Sync>(
    client: &ProductionClickHouseClient,
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
        .client
        .post(client.sanitized_database_url.as_str())
        .body(query)
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::ClickHouseQuery {
                message: format!("{e:?}"),
            })
        })?;

    let status = response.status();
    let response_body = response
        .text()
        .await
        .unwrap_or_else(|e| format!("Failed to get response text: {e}"));

    match status {
        reqwest::StatusCode::OK => Ok(()),
        _ => Err(Error::new(ErrorDetails::ClickHouseQuery {
            message: response_body,
        })),
    }
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

#[cfg(test)]
mod tests {

    use super::*;

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
