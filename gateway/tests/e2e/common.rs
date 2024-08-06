use gateway::clickhouse::ClickHouseConnectionInfo;
use serde_json::Value;
use uuid::Uuid;

pub async fn clickhouse_flush_async_insert(clickhouse: &ClickHouseConnectionInfo) {
    clickhouse
        .run_query("SYSTEM FLUSH ASYNC INSERT QUEUE".to_string())
        .await
        .unwrap();
}

pub async fn select_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM Inference WHERE id = '{}' FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub async fn select_model_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id = '{}' FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}
