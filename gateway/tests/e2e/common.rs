use gateway::clickhouse::ClickHouseConnectionInfo;
use reqwest::Url;
use serde_json::Value;
use uuid::Uuid;

lazy_static::lazy_static! {
    pub static ref CLICKHOUSE_URL: String = std::env::var("CLICKHOUSE_URL").expect("Environment variable CLICKHOUSE_URL must be set");
    static ref GATEWAY_URL: String = std::env::var("GATEWAY_URL").unwrap_or("http://localhost:3000".to_string());
}

pub fn get_gateway_endpoint(endpoint: &str) -> Url {
    let base_url: Url = GATEWAY_URL
        .parse()
        .expect("Invalid gateway URL (check environment variable GATEWAY_URL)");

    base_url.join(endpoint).unwrap()
}

pub async fn get_clickhouse() -> ClickHouseConnectionInfo {
    ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, "tensorzero_e2e_tests", false, None).unwrap()
}

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
