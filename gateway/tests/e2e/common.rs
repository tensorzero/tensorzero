use gateway::clickhouse::ClickHouseConnectionInfo;
use serde_json::Value;
use uuid::Uuid;

pub async fn clickhouse_flush_async_insert(connection_info: &ClickHouseConnectionInfo) {
    let (url, client) = match connection_info {
        ClickHouseConnectionInfo::Mock { .. } => unreachable!(),
        ClickHouseConnectionInfo::Production { url, client } => (url, client),
    };
    client
        .post(url.clone())
        .body("SYSTEM FLUSH ASYNC INSERT QUEUE")
        .send()
        .await
        .expect("Failed to flush ClickHouse");
}

pub async fn select_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;
    let (url, client) = match clickhouse_connection_info {
        ClickHouseConnectionInfo::Mock { .. } => unreachable!(),
        ClickHouseConnectionInfo::Production { url, client } => (url.clone(), client),
    };
    let query = format!(
        "SELECT * FROM Inference WHERE id = '{}' FORMAT JSONEachRow",
        inference_id
    );
    let response = client
        .post(url)
        .body(query)
        .send()
        .await
        .expect("Failed to query ClickHouse");
    let text = response.text().await.ok()?;
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub async fn select_model_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;
    let (url, client) = match clickhouse_connection_info {
        ClickHouseConnectionInfo::Mock { .. } => unreachable!(),
        ClickHouseConnectionInfo::Production { url, client } => (url.clone(), client),
    };
    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id = '{}' FORMAT JSONEachRow",
        inference_id
    );
    let response = client
        .post(url)
        .body(query)
        .send()
        .await
        .expect("Failed to query ClickHouse");
    let text = response.text().await.ok()?;
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}
