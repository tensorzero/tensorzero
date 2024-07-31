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
    Mock {
        mock_data: Arc<RwLock<HashMap<String, Vec<serde_json::Value>>>>,
        healthy: bool,
    },
    Production {
        url: String,
        client: Client,
    },
}

impl ClickHouseConnectionInfo {
    pub fn new(base_url: &str, mock: bool, healthy: Option<bool>) -> Self {
        if mock {
            return Self::Mock {
                mock_data: Arc::new(RwLock::new(HashMap::new())),
                healthy: healthy.unwrap_or(true),
            };
        }
        // TODO: parameterize the database name
        let database_name = "tensorzero";
        // Add a query string for the database using the URL crate
        let mut url = Url::parse(base_url).expect("Invalid base URL");
        url.query_pairs_mut().append_pair("database", database_name);
        Self::Production {
            url: url.to_string(),
            client: Client::new(),
        }
    }

    pub async fn write(
        &self,
        row: &(impl Serialize + Send + Sync),
        table: &str,
    ) -> Result<(), Error> {
        match self {
            Self::Mock { mock_data, .. } => {
                write_mock(row, table, &mut mock_data.write().await).await
            }
            Self::Production { url, client } => write_production(url, client, row, table).await,
        }
    }

    /// Test helper: reads from the table `table` in our mock DB and returns an element that has (serialized) `column` equal to `value`.
    /// Returns None if no such element is found.
    #[cfg(test)]
    pub async fn read(&self, table: &str, column: &str, value: &str) -> Option<serde_json::Value> {
        match self {
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
            Self::Production { .. } => unimplemented!(),
        }
    }

    pub async fn health(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Mock { healthy, .. } => {
                if *healthy {
                    Ok(())
                } else {
                    Err("Mock ClickHouse is not healthy".into())
                }
            }
            Self::Production { url, client } => match client.get(url).send().await {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("ClickHouse is not healthy: {}", e).into()),
            },
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
    url: &str,
    client: &Client,
    row: &(impl Serialize + Send + Sync),
    table: &str,
) -> Result<(), Error> {
    let row_json = serde_json::to_string(row).map_err(|e| Error::Serialization {
        message: e.to_string(),
    })?;
    // TODO: allow the user to parameterize whether to wait_for_async_insert
    // Design we'll use:
    //   1. Feedback should wait
    //   2. Allow the user to optionally configure that a function is latency sensitive (default false). If so,
    //      don't wait for the async insert to finish. Otherwise, wait.
    let query = format!(
        "INSERT INTO {table}\n\
     SETTINGS async_insert=1, wait_for_async_insert=0\n\
     FORMAT JSONEachRow\n\
     {row_json}"
    );
    let response = client
        .post(url)
        .body(query.clone())
        .send()
        .await
        .map_err(|e| Error::ClickHouseWrite {
            message: e.to_string(),
        })?;
    match response.status() {
        reqwest::StatusCode::OK => Ok(()),
        _ => Err(Error::ClickHouseWrite {
            message: response
                .text()
                .await
                .unwrap_or_else(|e| format!("Failed to get response text: {}", e)),
        }),
    }
}
