use std::collections::HashMap;

use reqwest::Url;
use tensorzero_core::{
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::datasets::{CLICKHOUSE_DATETIME_FORMAT, DatapointKind},
    feature_flags,
};
use uuid::Uuid;

lazy_static::lazy_static! {
    static ref GATEWAY_URL: String = std::env::var("TENSORZERO_GATEWAY_URL")
        .unwrap_or_else(|_| "http://localhost:3000".to_string());
}

/// Returns true if we're testing against Postgres.
///
/// This is not really perfect because we rely on the tests running in the same context as when we
/// launch the gateway container, but it's true for our CI setup and is good enough for today.
pub fn is_postgres_test() -> bool {
    feature_flags::ENABLE_POSTGRES_READ.get() || feature_flags::ENABLE_POSTGRES_WRITE.get()
}

/// Skips the current test if running against Postgres.
/// Use this for tests that don't have Postgres implementations yet.
///
/// TODO(#5691): Remove this once we have Postgres implementations for all tests.
#[macro_export]
macro_rules! skip_for_postgres {
    () => {
        if $crate::common::is_postgres_test() {
            eprintln!("Skipping test: Postgres implementation not yet available");
            return;
        }
    };
}

pub fn get_gateway_endpoint(endpoint: &str) -> Url {
    let base_url: Url = GATEWAY_URL
        .parse()
        .expect("Invalid gateway URL (check environment variable TENSORZERO_GATEWAY_URL)");

    base_url.join(endpoint).unwrap()
}

pub async fn delete_datapoint(
    clickhouse: &ClickHouseConnectionInfo,
    datapoint_kind: DatapointKind,
    function_name: &str,
    dataset_name: &str,
    datapoint_id: Uuid,
) {
    let datapoint = clickhouse.run_query_synchronous(
        "SELECT * FROM {table_name:Identifier} WHERE dataset_name={dataset_name:String} AND function_name={function_name:String} AND id = {id:String} ORDER BY updated_at DESC LIMIT 1 FORMAT JSONEachRow;".to_string(),
        &HashMap::from([
            ("table_name", datapoint_kind.table_name().as_str()),
            ("function_name", function_name),
            ("dataset_name", dataset_name),
            ("id", datapoint_id.to_string().as_str())
        ])).await.unwrap();

    assert!(
        !datapoint.response.is_empty(),
        "Datapoint not found with params {datapoint_kind:?}, {function_name}, {dataset_name}, {datapoint_id}"
    );

    let mut datapoint_json: serde_json::Value = serde_json::from_str(&datapoint.response).unwrap();

    // We delete datapoints by writing a new row (which ClickHouse will merge)
    // with the 'is_deleted' and 'updated_at' fields modified.
    datapoint_json["is_deleted"] = true.into();
    datapoint_json["updated_at"] =
        format!("{}", chrono::Utc::now().format(CLICKHOUSE_DATETIME_FORMAT)).into();

    clickhouse
        .write_batched(&[datapoint_json], datapoint_kind.table_name())
        .await
        .unwrap();
}
