use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::sync::RwLockWriteGuard;

use crate::error::Error;

#[derive(Clone)]
pub enum ClickHouseConnectionInfo {
    Mock {
        mock_data: Arc<RwLock<HashMap<String, Vec<serde_json::Value>>>>,
    },
    Production {
        url: String,
    },
}

impl ClickHouseConnectionInfo {
    pub fn new(base_url: &str, mock: bool) -> Self {
        if mock {
            return Self::Mock {
                mock_data: Arc::new(RwLock::new(HashMap::new())),
            };
        }
        let database_name = "tensorzero";
        // Add a query string for the database
        let database_url = if base_url.contains('?') {
            format!("{base_url}&database={database_name}")
        } else {
            format!("{base_url}?database={database_name}")
        };
        Self::Production {
            url: database_url.to_string(),
        }
    }

    pub async fn write(
        &self,
        client: &Client,
        row: &(impl Serialize + Send + Sync),
        table: &str,
    ) -> Result<(), Error> {
        match self {
            Self::Mock { mock_data } => write_mock(row, table, &mut mock_data.write().await).await,
            Self::Production { url } => write_production(url, client, row, table).await,
        }
    }

    /// Test helper: reads from the table `table` in our mock DB and returns an element that has (serialized) `column` equal to `value`.
    /// Returns None if no such element is found.
    #[cfg(test)]
    pub async fn read(&self, table: &str, column: &str, value: &str) -> Option<serde_json::Value> {
        match self {
            Self::Mock { mock_data } => {
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

    // TODO: use this
    pub async fn health(&self, client: &Client) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Mock { .. } => todo!(),
            Self::Production { url } => {
                client.get(url).send().await?;
                Ok(())
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
    url: &str,
    client: &Client,
    row: &(impl Serialize + Send + Sync),
    table: &str,
) -> Result<(), Error> {
    let row_json = serde_json::to_string(row).map_err(|e| Error::Serialization {
        message: e.to_string(),
    })?;
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
