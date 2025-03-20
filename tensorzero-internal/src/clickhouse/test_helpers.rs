#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]
use super::ClickHouseConnectionInfo;
use serde_json::Value;
use uuid::Uuid;

lazy_static::lazy_static! {
    pub static ref CLICKHOUSE_URL: String = std::env::var("TENSORZERO_CLICKHOUSE_URL").expect("Environment variable TENSORZERO_CLICKHOUSE_URL must be set");
}

pub async fn get_clickhouse() -> ClickHouseConnectionInfo {
    let clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    let start = std::time::Instant::now();
    println!("Connecting to ClickHouse");
    let res = ClickHouseConnectionInfo::new(clickhouse_url.as_ref())
        .await
        .expect("Failed to connect to ClickHouse");
    println!("Connected to ClickHouse in {:?}", start.elapsed());
    res
}

#[cfg(feature = "e2e_tests")]
pub async fn clickhouse_flush_async_insert(clickhouse: &ClickHouseConnectionInfo) {
    if let Err(e) = clickhouse
        .run_query("SYSTEM FLUSH ASYNC INSERT QUEUE".to_string(), None)
        .await
    {
        tracing::warn!("Failed to run `SYSTEM FLUSH ASYNC INSERT QUEUE`: {}", e);
    }
}

#[allow(dead_code)]
pub async fn select_chat_datapoint_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ChatInferenceDatapoint WHERE id = '{}' LIMIT 1 FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

#[allow(dead_code)]
pub async fn select_json_datapoint_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM JsonInferenceDatapoint WHERE id = '{}' LIMIT 1 FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub async fn select_chat_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM ChatInference WHERE id = '{}' LIMIT 1 FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub async fn select_json_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    // We limit to 1 in case there are duplicate entries (can be caused by a race condition in polling batch inferences)
    let query = format!(
        "SELECT * FROM JsonInference WHERE id = '{}' LIMIT 1 FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub async fn select_model_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    // We limit to 1 in case there are duplicate entries (can be caused by a race condition in polling batch inferences)
    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id = '{}' LIMIT 1 FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub async fn select_model_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Vec<Value>> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    // We limit to 1 in case there are duplicate entries (can be caused by a race condition in polling batch inferences)
    let query = format!(
        "SELECT * FROM ModelInference WHERE inference_id = '{}' FORMAT JSONEachRow",
        inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
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

pub async fn select_inference_tags_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    function_name: &str,
    tag_key: &str,
    tag_value: &str,
    inference_id: Uuid,
) -> Option<Value> {
    #[cfg(feature = "e2e_tests")]
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM InferenceTag WHERE function_name = '{}' AND key = '{}' AND value = '{}' AND inference_id = '{}' FORMAT JSONEachRow",
        function_name, tag_key, tag_value, inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

pub async fn select_batch_model_inference_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    inference_id: Uuid,
) -> Option<Value> {
    let query = format!(
        r#"
        SELECT bmi.*
        FROM BatchModelInference bmi
        INNER JOIN BatchIdByInferenceId bid ON bmi.inference_id = bid.inference_id
        WHERE bid.inference_id = '{}'
        FORMAT JSONEachRow"#,
        inference_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    Some(serde_json::from_str(&text).unwrap())
}

pub async fn select_batch_model_inferences_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_id: Uuid,
) -> Option<Vec<Value>> {
    let query = format!(
        r#"
        SELECT bmi.*
        FROM BatchModelInference bmi
        WHERE bmi.batch_id = '{}'
        FORMAT JSONEachRow"#,
        batch_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json_rows: Vec<Value> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    Some(json_rows)
}

pub async fn select_latest_batch_request_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    batch_id: Uuid,
) -> Option<Value> {
    let query = format!(
        "SELECT * FROM BatchRequest WHERE batch_id = '{}' ORDER BY timestamp DESC LIMIT 1 FORMAT JSONEachRow",
        batch_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

#[cfg(feature = "e2e_tests")]
pub async fn select_feedback_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    table_name: &str,
    feedback_id: Uuid,
) -> Option<Value> {
    clickhouse_flush_async_insert(clickhouse_connection_info).await;

    let query = format!(
        "SELECT * FROM {} WHERE id = '{}' FORMAT JSONEachRow",
        table_name, feedback_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}

#[cfg(feature = "e2e_tests")]
pub async fn select_feedback_by_target_id_clickhouse(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    table_name: &str,
    target_id: Uuid,
) -> Option<Value> {
    let query = format!(
        "SELECT * FROM {} WHERE target_id = '{}' FORMAT JSONEachRow",
        table_name, target_id
    );

    let text = clickhouse_connection_info
        .run_query(query, None)
        .await
        .unwrap();
    let json: Value = serde_json::from_str(&text).ok()?;
    Some(json)
}
