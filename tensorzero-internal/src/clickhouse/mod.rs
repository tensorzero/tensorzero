use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::RwLockWriteGuard;
use url::Url;

pub mod migration_manager;
#[cfg(any(test, feature = "e2e_tests"))]
pub mod test_helpers;

use crate::error::{Error, ErrorDetails};

#[derive(Debug, Clone)]
pub enum ClickHouseConnectionInfo {
    Disabled,
    Mock {
        mock_data: Arc<RwLock<HashMap<String, Vec<serde_json::Value>>>>,
        healthy: bool,
    },
    Production {
        database_url: SecretString,
        database: String,
        client: Client,
    },
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
    pub async fn new(database_url: &str) -> Result<Self, Error> {
        // Add a query string for the database using the URL crate
        let mut database_url = Url::parse(database_url).map_err(|_| {
            Error::new(ErrorDetails::Config {
                message: "Invalid ClickHouse database URL".to_string(),
            })
        })?;

        #[allow(unused_variables)]
        let database = validate_clickhouse_url_get_db_name(&database_url)?
            .unwrap_or_else(|| "default".to_string());

        #[cfg(feature = "e2e_tests")]
        let database = "tensorzero_e2e_tests".to_string();

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

        let connection_info = Self::Production {
            database_url,
            database,
            client: Client::new(),
        };
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

    pub async fn write(
        &self,
        rows: &[impl Serialize + Send + Sync],
        table: &str,
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
            } => write_production(database_url, client, rows, table).await,
        }
    }

    /// Test helper: reads from the table `table` in our mock DB and returns an element that has (serialized) `column` equal to `value`.
    /// Returns None if no such element is found.
    #[cfg(test)]
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

    pub async fn health(&self) -> Result<(), Error> {
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

                let timeout = if cfg!(feature = "e2e_tests") {
                    // Set a long timeout to try to debug batch tests
                    std::time::Duration::from_secs(60)
                } else {
                    // If ClickHouse is healthy, it should respond within 1000ms
                    std::time::Duration::from_millis(1000)
                };

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

    pub async fn run_query(
        &self,
        query: String,
        parameters: Option<&HashMap<&str, &str>>,
    ) -> Result<String, Error> {
        match self {
            Self::Disabled => Ok("".to_string()),
            Self::Mock { .. } => Ok("".to_string()),
            Self::Production {
                database_url,
                client,
                ..
            } => {
                let mut database_url = Url::parse(database_url.expose_secret()).map_err(|e| Error::new(ErrorDetails::ClickHouseQuery { message: format!("Error parsing ClickHouse URL: {e}. This should never happen. Please submit a bug report at https://github.com/tensorzero/tensorzero/issues/new") }))?;
                // Add query parameters if provided
                if let Some(params) = parameters {
                    for (key, value) in params {
                        let param_key = format!("param_{}", key);
                        database_url
                            .query_pairs_mut()
                            .append_pair(&param_key, value);
                    }
                }
                let response = client
                    .post(database_url)
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
                    reqwest::StatusCode::OK => Ok(response_body),
                    _ => Err(Error::new(ErrorDetails::ClickHouseQuery {
                        message: response_body,
                    })),
                }
            }
        }
    }

    pub async fn create_database(&self) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Mock { .. } => Ok(()),
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
                let query = format!("CREATE DATABASE IF NOT EXISTS {}", database);
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
                    reqwest::StatusCode::OK => Ok(()),
                    _ => Err(Error::new(ErrorDetails::ClickHouseQuery {
                        message: response_body,
                    })),
                }
            }
        }
    }
}

async fn write_mock(
    rows: &[impl Serialize + Send + Sync],
    table: &str,
    tables: &mut RwLockWriteGuard<'_, HashMap<String, Vec<serde_json::Value>>>,
) -> Result<(), Error> {
    for row in rows {
        let row_value = serde_json::to_value(row).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: e.to_string(),
            })
        })?;
        tables.entry(table.to_string()).or_default().push(row_value);
    }
    Ok(())
}

async fn write_production(
    database_url: &SecretString,
    client: &Client,
    rows: &[impl Serialize + Send + Sync],
    table: &str,
) -> Result<(), Error> {
    // Empty rows is a no-op
    if rows.is_empty() {
        return Ok(());
    }

    let rows_json: Result<Vec<String>, _> = rows
        .iter()
        .map(|row| {
            serde_json::to_string(row).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: e.to_string(),
                })
            })
        })
        .collect();

    let rows_json = rows_json?.join("\n");

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
    const OVERRIDDEN_SETTINGS: [&str; 2] = [
        "input_format_skip_unknown_fields",
        "input_format_null_as_default",
    ];

    let existing_pairs: Vec<(String, String)> = database_url
        .query_pairs()
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();

    database_url.query_pairs_mut().clear();

    for (key, value) in existing_pairs {
        if OVERRIDDEN_SETTINGS.contains(&key.as_str()) {
            tracing::warn!(
                "Your ClickHouse connection URL has the setting '{}' but it will be overridden.",
                key
            );
        } else {
            database_url.query_pairs_mut().append_pair(&key, &value);
        }
    }

    for setting in OVERRIDDEN_SETTINGS.iter() {
        database_url.query_pairs_mut().append_pair(setting, "0");
    }
    database_url.query_pairs_mut().finish();
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

    // Validate the port
    if url.port().is_none() {
        return Err(ErrorDetails::Config {
            message: "Missing port in ClickHouse URL".to_string(),
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
    let mut path_segments: Vec<_> = url.path_segments().map(|s| s.collect()).unwrap_or_default();
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_set_clickhouse_format_settings() {
        let mut database_url = Url::parse("http://chuser:chpassword@localhost:8123/").unwrap();
        set_clickhouse_format_settings(&mut database_url);
        assert_eq!(database_url.to_string(), "http://chuser:chpassword@localhost:8123/?input_format_skip_unknown_fields=0&input_format_null_as_default=0");

        let mut database_url = Url::parse("http://chuser:chpassword@localhost:8123/?input_format_skip_unknown_fields=1&input_format_defaults_for_omitted_fields=1&input_format_null_as_default=1").unwrap();
        set_clickhouse_format_settings(&mut database_url);
        assert_eq!(database_url.to_string(), "http://chuser:chpassword@localhost:8123/?input_format_defaults_for_omitted_fields=1&input_format_skip_unknown_fields=0&input_format_null_as_default=0");
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

        let database_url = Url::parse("http://localhost/").unwrap();
        let err = validate_clickhouse_url_get_db_name(&database_url).unwrap_err();
        assert_eq!(
            err,
            ErrorDetails::Config {
                message: "Missing port in ClickHouse URL".to_string(),
            }
            .into()
        );

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
}
