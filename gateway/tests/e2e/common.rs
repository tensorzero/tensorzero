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
    let clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    ClickHouseConnectionInfo::new(clickhouse_url.as_ref())
        .await
        .expect("Failed to connect to ClickHouse")
}

async fn clickhouse_flush_async_insert(clickhouse: &ClickHouseConnectionInfo) {
    clickhouse
        .run_query("SYSTEM FLUSH ASYNC INSERT QUEUE".to_string())
        .await
        .unwrap();
}

pub(crate) async fn select_chat_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ChatInference WHERE id = '{}' FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub(crate) async fn select_json_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM JsonInference WHERE id = '{}' FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub(crate) async fn select_model_inference_clickhouse(
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

pub(crate) async fn select_model_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Vec<Value>> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id = '{}' FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json_rows: Vec<Value> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    if json_rows.is_empty() {
        None
    } else {
        Some(json_rows)
    }
}

pub(crate) async fn select_feedback_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    table_name: &str,
    feedback_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM {} WHERE id = '{}' FORMAT JSONEachRow",
        table_name, feedback_id
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub(crate) async fn select_feedback_tags_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    metric_name: &str,
    tag_key: &str,
    tag_value: &str,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM FeedbackTag WHERE metric_name = '{}' AND key = '{}' AND value = '{}' FORMAT JSONEachRow",
        metric_name, tag_key, tag_value
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub(crate) async fn select_inference_tags_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    function_name: &str,
    tag_key: &str,
    tag_value: &str,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM InferenceTag WHERE function_name = '{}' AND key = '{}' AND value = '{}' FORMAT JSONEachRow",
        function_name, tag_key, tag_value
    );

    let text = clickhouse_connection_info.run_query(query).await.unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}
