use reqwest::Client;

use crate::error::Error;

#[derive(Clone)]
pub struct ClickHouseConnectionInfo {
    pub url: String,
}

impl ClickHouseConnectionInfo {
    pub fn new(base_url: &str, database_name: Option<&str>) -> Self {
        let database_name = database_name.unwrap_or("tensorzero");
        // Add a query string for the database
        let database_url = if base_url.contains('?') {
            format!("{base_url}&database={database_name}")
        } else {
            format!("{base_url}?database={database_name}")
        };
        Self {
            url: database_url.to_string(),
        }
    }
}

pub async fn clickhouse_write<T: Serialize>(
    client: &Client,
    connection_info: &ClickHouseConnectionInfo,
    row: &T,
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
        .post(connection_info.url.as_str())
        .body(query)
        .send()
        .await
        .map_err(|e| Error::ClickhouseWrite {
            message: e.to_string(),
        })?;
    match response.status() {
        reqwest::StatusCode::OK => Ok(()),
        _ => Err(Error::ClickhouseWrite {
            message: response
                .text()
                .await
                .unwrap_or_else(|e| format!("Failed to get response text: {}", e)),
        }),
    }
}

// TODO: use this
pub async fn clickhouse_health(
    client: &Client,
    connection_info: &ClickHouseConnectionInfo,
) -> Result<(), Box<dyn std::error::Error>> {
    client.get(connection_info.url.as_str()).send().await?;
    Ok(())
}
