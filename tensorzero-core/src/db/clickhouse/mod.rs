use async_trait::async_trait;
use secrecy::SecretString;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::sync::Arc;
use url::Url;

use crate::config::BatchWritesConfig;
use crate::db::clickhouse::batching::BatchWriterHandle;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::clickhouse_client::DisabledClickHouseClient;
use crate::db::clickhouse::clickhouse_client::ProductionClickHouseClient;
use crate::db::HealthCheckable;
use crate::error::DelayedError;
use crate::error::{Error, ErrorDetails};

pub use clickhouse_client::ClickHouseClient;
pub use table_name::TableName;

#[cfg(any(test, feature = "pyo3"))]
use crate::db::clickhouse::clickhouse_client::FakeClickHouseClient;

mod batching;
pub mod clickhouse_client; // Public because tests will use clickhouse_client::FakeClickHouseClient and clickhouse_client::MockClickHouseClient
pub mod dataset_queries;
pub mod feedback;
pub mod inference_queries;
pub mod migration_manager;
pub mod query_builder;
mod select_queries;
mod table_name;

#[cfg(test)]
mod mock_clickhouse_connection_info;
#[cfg(test)]
pub(crate) use mock_clickhouse_connection_info::MockClickHouseConnectionInfo;

#[cfg(any(test, feature = "e2e_tests"))]
pub mod test_helpers;

/// Wrapper for ClickHouse client implementations
#[derive(Debug, Clone)]
pub struct ClickHouseConnectionInfo {
    inner: Arc<dyn ClickHouseClient>,
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
        let database = validate_clickhouse_url_get_db_name(&database_url)?
            .unwrap_or_else(|| "tensorzero_e2e_tests".to_string());

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

        let client = ProductionClickHouseClient::new(
            database_url.clone(),
            cluster_name,
            database.clone(),
            batch_config,
        )
        .await?;
        Ok(Self {
            inner: Arc::new(client),
        })
    }

    pub fn new_disabled() -> Self {
        Self {
            inner: Arc::new(DisabledClickHouseClient),
        }
    }

    #[cfg(test)]
    pub fn new_mock(inner: Arc<dyn ClickHouseClient>) -> Self {
        Self { inner }
    }

    #[cfg(any(test, feature = "pyo3"))]
    pub fn new_fake() -> Self {
        Self {
            inner: Arc::new(FakeClickHouseClient::new(true)),
        }
    }

    pub fn database_url(&self) -> &SecretString {
        self.inner.database_url()
    }

    /// Returns the cluster name
    pub fn cluster_name(&self) -> &Option<String> {
        self.inner.cluster_name()
    }

    /// Returns the database name
    pub fn database(&self) -> &str {
        self.inner.database()
    }

    pub fn is_batching_enabled(&self) -> bool {
        self.inner.is_batching_enabled()
    }

    pub fn batcher_join_handle(&self) -> Option<BatchWriterHandle> {
        self.inner.batcher_join_handle()
    }

    /// Writes rows to ClickHouse using our batched write implementation
    /// (if enabled in the config)
    /// The provided rows might not yet be sent to ClickHouse when this function completes.
    pub async fn write_batched(
        &self,
        rows: &[impl Serialize + Send + Sync],
        table: TableName,
    ) -> Result<(), Error> {
        let rows_json: Result<Vec<String>, Error> = rows
            .iter()
            .map(|row| {
                serde_json::to_string(row).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: e.to_string(),
                    })
                })
            })
            .collect();
        self.inner.write_batched_internal(rows_json?, table).await
    }

    /// Write rows to ClickHouse without, without using our batched write implementation.
    /// The provided rows will have been sent to ClickHouse when this function completes.
    pub async fn write_non_batched<T: Serialize + Send + Sync>(
        &self,
        rows: Rows<'_, T>,
        table: TableName,
    ) -> Result<(), Error> {
        let rows_json = rows.as_json()?;
        self.inner
            .write_non_batched_internal(rows_json.into_owned(), table)
            .await
    }

    /// Test helper: reads from the table `table` in our mock DB and returns an element that has (serialized) `column` equal to `value`.
    /// Returns None if no such element is found.
    #[cfg(test)]
    #[expect(clippy::missing_panics_doc)]
    pub async fn read(&self, table: &str, column: &str, value: &str) -> Option<serde_json::Value> {
        // Only FakeClickHouseClient supports this
        let inner_any = &self.inner as &dyn std::any::Any;
        if let Some(fake) = inner_any.downcast_ref::<FakeClickHouseClient>() {
            let data = fake.data.read().await;
            let table = data.get(table).unwrap();
            for row in table {
                if let Some(value_in_row) = row.get(column) {
                    if value_in_row.as_str() == Some(value) {
                        return Some(row.clone());
                    }
                }
            }
            None
        } else {
            panic!("read() is only supported on FakeClickHouseClient")
        }
    }

    pub fn is_cluster_configured(&self) -> bool {
        self.inner.is_cluster_configured()
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
        self.inner.run_query_synchronous(query, parameters).await
    }

    pub async fn run_query_synchronous_delayed_err(
        &self,
        query: String,
        parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, DelayedError> {
        self.inner
            .run_query_synchronous_delayed_err(query, parameters)
            .await
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
        self.inner
            .run_query_with_external_data(external_data, query)
            .await
    }

    pub async fn check_database_and_migrations_table_exists(&self) -> Result<bool, Error> {
        self.inner
            .check_database_and_migrations_table_exists()
            .await
    }

    pub async fn create_database_and_migrations_table(&self) -> Result<(), Error> {
        self.inner.create_database_and_migrations_table().await
    }

    pub fn get_on_cluster_name(&self) -> String {
        self.inner.get_on_cluster_name()
    }

    pub fn get_maybe_replicated_table_engine_name(
        &self,
        args: GetMaybeReplicatedTableEngineNameArgs<'_>,
    ) -> String {
        self.inner.get_maybe_replicated_table_engine_name(args)
    }

    pub fn client_type(&self) -> ClickHouseClientType {
        self.inner.client_type()
    }
}

impl Display for ClickHouseConnectionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.client_type() {
            ClickHouseClientType::Production => {
                write!(f, "enabled (database = {})", self.database())
            }
            ClickHouseClientType::Fake => write!(f, "fake"),
            ClickHouseClientType::Disabled => write!(f, "disabled"),
        }
    }
}

// Update the HealthCheckable implementation to delegate to the trait
#[async_trait]
impl HealthCheckable for ClickHouseConnectionInfo {
    async fn health(&self) -> Result<(), Error> {
        self.inner.health().await
    }
}

pub struct GetMaybeReplicatedTableEngineNameArgs<'a> {
    pub table_engine_name: &'a str,
    pub table_name: &'a str,
    pub engine_args: &'a [&'a str],
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
}
