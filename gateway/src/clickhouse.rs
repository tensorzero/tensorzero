use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::RwLockWriteGuard;
use url::Url;

use crate::error::Error;

#[derive(Debug, Clone)]
pub enum ClickHouseConnectionInfo {
    Disabled,
    Mock {
        mock_data: Arc<RwLock<HashMap<String, Vec<serde_json::Value>>>>,
        healthy: bool,
    },
    Production {
        database_url: Url,
        database: String,
        client: Client,
    },
}

impl ClickHouseConnectionInfo {
    pub fn new(database_url: &str) -> Result<Self, Error> {
        // Add a query string for the database using the URL crate
        #[allow(unused_mut)]
        let mut database_url = Url::parse(database_url).map_err(|e| Error::Config {
            message: format!("Invalid ClickHouse database URL: {}", e),
        })?;

        #[cfg(feature = "e2e_tests")]
        let mut database_url = set_e2e_test_database(database_url);

        let mut database = String::new();

        // Get the database name from the query string
        for (key, value) in database_url.query_pairs() {
            if key == "database" {
                database = value.to_string();
                break;
            }
        }
        // If there is no database name, we use the "default" database
        if database.is_empty() {
            database = "default".to_string();
        }

        // Set ClickHouse format settings for some error checking on writes
        database_url
            .query_pairs_mut()
            .append_pair("input_format_skip_unknown_fields", "0")
            .append_pair("input_format_defaults_for_omitted_fields", "0")
            .append_pair("input_format_null_as_default", "0")
            .finish();

        let client = Client::new();

        Ok(Self::Production {
            database_url,
            database,
            client,
        })
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
        row: &(impl Serialize + Send + Sync),
        table: &str,
    ) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Mock { mock_data, .. } => {
                write_mock(row, table, &mut mock_data.write().await).await
            }
            Self::Production {
                database_url,
                client,
                ..
            } => write_production(database_url, client, row, table).await,
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

    pub async fn health(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Disabled => Ok(()),
            Self::Mock { healthy, .. } => {
                if *healthy {
                    Ok(())
                } else {
                    Err("Mock ClickHouse is not healthy".into())
                }
            }
            Self::Production {
                database_url,
                client,
                ..
            } => match client.get(database_url.clone()).send().await {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("ClickHouse is not healthy: {}", e).into()),
            },
        }
    }

    pub async fn run_query(&self, query: String) -> Result<String, Error> {
        match self {
            Self::Disabled => Ok("".to_string()),
            Self::Mock { .. } => Ok("".to_string()),
            Self::Production {
                client,
                database_url,
                ..
            } => {
                let response = client
                    .post(database_url.clone())
                    .body(query)
                    .send()
                    .await
                    .map_err(|e| Error::ClickHouseQuery {
                        message: e.to_string(),
                    })?;

                let status = response.status();

                let response_body = response.text().await.map_err(|e| Error::ClickHouseQuery {
                    message: e.to_string(),
                })?;

                match status {
                    reqwest::StatusCode::OK => Ok(response_body),
                    _ => Err(Error::ClickHouseQuery {
                        message: response_body,
                    }),
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
                    .map_err(|e| Error::ClickHouseQuery {
                        message: e.to_string(),
                    })?;

                let status = response.status();

                let response_body = response.text().await.map_err(|e| Error::ClickHouseQuery {
                    message: e.to_string(),
                })?;

                match status {
                    reqwest::StatusCode::OK => Ok(()),
                    _ => Err(Error::ClickHouseQuery {
                        message: response_body,
                    }),
                }
            }
        }
    }
}

async fn write_mock(
    row: &(impl Serialize + Send + Sync),
    table: &str,
    tables: &mut RwLockWriteGuard<'_, HashMap<String, Vec<serde_json::Value>>>,
) -> Result<(), Error> {
    let row_value = serde_json::to_value(row).map_err(|e| Error::Serialization {
        message: e.to_string(),
    })?;
    tables.entry(table.to_string()).or_default().push(row_value);
    Ok(())
}

async fn write_production(
    database_url: &Url,
    client: &Client,
    row: &(impl Serialize + Send + Sync),
    table: &str,
) -> Result<(), Error> {
    let row_json = serde_json::to_string(row).map_err(|e| Error::Serialization {
        message: e.to_string(),
    })?;

    // We can wait for the async insert since we're spawning a new tokio task to do the insert
    let query = format!(
        "INSERT INTO {table}\n\
        SETTINGS async_insert=1, wait_for_async_insert=1\n\
        FORMAT JSONEachRow\n\
        {row_json}"
    );
    let response = client
        .post(database_url.clone())
        .body(query)
        .send()
        .await
        .map_err(|e| Error::ClickHouseQuery {
            message: e.to_string(),
        })?;
    match response.status() {
        reqwest::StatusCode::OK => Ok(()),
        _ => Err(Error::ClickHouseQuery {
            message: response
                .text()
                .await
                .unwrap_or_else(|e| format!("Failed to get response text: {}", e)),
        }),
    }
}

#[cfg(feature = "e2e_tests")]
/// Sets the database for the ClickHouse client to tensorzero_e2e_tests for the duration of the test.
fn set_e2e_test_database(database_url: Url) -> Url {
    let mut new_database_url = database_url.clone();
    new_database_url.set_query(Some("database=tensorzero_e2e_tests"));
    new_database_url
}
