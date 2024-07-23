#![cfg(test)]
use crate::clickhouse::ClickHouseConnectionInfo;
use reqwest::StatusCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::OnceCell;

pub struct ClickHouseConnectionPool {
    next_db: AtomicUsize,
    url: String,
}

impl ClickHouseConnectionPool {
    async fn new() -> Self {
        let url = std::env::var("API_CLICKHOUSE_URL")
            .expect("Missing environment variable API_CLICKHOUSE_URL");
        let _ = url::Url::parse(&url).expect("Failed to parse API_CLICKHOUSE_URL");
        Self {
            next_db: AtomicUsize::new(2),
            url,
        }
    }

    async fn setup_database(
        &self,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        database_name: &str,
    ) {
        if database_name == "tensorzero" {
            panic!("Can't use database \"tensorzero\" for testing.");
        }
        // We would ideally want to keep this client in the struct and reuse,
        // but this messes up since tests each have their own Tokio runtime.
        let client = reqwest::Client::new();
        // Create a separate URL with allow_suspicious_low_cardinality_types=1
        let low_cardinality_uuid_url = if clickhouse_connection_info.url.contains('?') {
            format!(
                "{}&allow_suspicious_low_cardinality_types=1",
                clickhouse_connection_info.url,
            )
        } else {
            format!(
                "{}?allow_suspicious_low_cardinality_types=1",
                clickhouse_connection_info.url,
            )
        };
        let drop_database_query = format!("DROP DATABASE IF EXISTS {}", database_name);
        let response = client
            // This one needs to be to the root in case the db doesn't exist
            .post(&self.url)
            .body(drop_database_query)
            .send()
            .await
            .expect("Failed to flush database");
        assert_eq!(response.status(), StatusCode::OK);

        // Create the database
        let create_database_query = format!("CREATE DATABASE IF NOT EXISTS {}", database_name);
        let response = client
            // This one needs to be to the root in case the db doesn't exist
            .post(&self.url)
            .body(create_database_query)
            .send()
            .await
            .expect("Failed to create database");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "{}",
            response.text().await.unwrap()
        );
        // Create BooleanMetricFeedback table
        let create_boolean_metric_feedback_query = format!(
            r#"
        CREATE TABLE IF NOT EXISTS {database_name}.BooleanMetricFeedback
        (
            id UUID DEFAULT generateUUIDv7(),
            target_type Enum('episode' = 1, 'inference' = 2),
            target_id UUID,
            metric_name LowCardinality(String),
            value Bool
        ) ENGINE = MergeTree()
        ORDER BY (metric_name, target_id);
        "#
        );
        let response = client
            .post(&low_cardinality_uuid_url)
            .body(create_boolean_metric_feedback_query)
            .send()
            .await
            .expect("Failed to create BooleanMetricFeedback table");
        assert_eq!(response.status(), StatusCode::OK);

        // Create FloatMetricFeedback table
        let create_float_metric_feedback_query = format!(
            r#"
        CREATE TABLE IF NOT EXISTS {database_name}.FloatMetricFeedback
        (
            id UUID DEFAULT generateUUIDv7(),
            target_type Enum('episode' = 1, 'inference' = 2),
            target_id UUID,
            metric_name LowCardinality(String),
            value Float32
        ) ENGINE = MergeTree()
        ORDER BY (metric_name, target_id);
        "#
        );
        let response = client
            .post(&low_cardinality_uuid_url)
            .body(create_float_metric_feedback_query)
            .send()
            .await
            .expect("Failed to create FloatMetricFeedback table");
        assert_eq!(response.status(), StatusCode::OK);

        // Create DemonstrationFeedback table
        let create_demonstration_feedback_query = format!(
            r#"
        CREATE TABLE IF NOT EXISTS {database_name}.DemonstrationFeedback
        (
            id UUID DEFAULT generateUUIDv7(),
            inference_id UUID,
            value String
        ) ENGINE = MergeTree()
        ORDER BY inference_id;
        "#
        );
        let response = client
            .post(&low_cardinality_uuid_url)
            .body(create_demonstration_feedback_query)
            .send()
            .await
            .expect("Failed to create DemonstrationFeedback table");
        assert_eq!(response.status(), StatusCode::OK);

        // Create CommentFeedback table
        let create_comment_feedback_query = format!(
            r#"
        CREATE TABLE IF NOT EXISTS {database_name}.CommentFeedback
        (
            id UUID DEFAULT generateUUIDv7(),
            target_type Enum('episode' = 1, 'inference' = 2),
            target_id UUID,
            value String
        ) ENGINE = MergeTree()
        ORDER BY target_id;
        "#
        );
        let response = client
            .post(&low_cardinality_uuid_url)
            .body(create_comment_feedback_query)
            .send()
            .await
            .expect("Failed to create CommentFeedback table");
        assert_eq!(response.status(), StatusCode::OK);

        // Create Inference table
        let create_inference_query = format!(
            r#"
        CREATE TABLE IF NOT EXISTS {database_name}.Inference
        (
            id UUID,
            function_name LowCardinality(String),
            variant_name LowCardinality(String),
            episode_id UUID,
            input String,
            output String
        ) ENGINE = MergeTree()
        ORDER BY (function_name, variant_name, episode_id);
        "#
        );
        let response = client
            .post(&low_cardinality_uuid_url)
            .body(create_inference_query)
            .send()
            .await
            .expect("Failed to create Inference table");
        assert_eq!(response.status(), StatusCode::OK);

        // Create ModelInference table
        let create_model_inference_query = format!(
            r#"
        CREATE TABLE IF NOT EXISTS {database_name}.ModelInference
        (
            id UUID DEFAULT generateUUIDv7(),
            inference_id UUID,
            input String,
            output String,
            raw_response String,
            input_tokens UInt32,
            output_tokens UInt32
        ) ENGINE = MergeTree()
        ORDER BY inference_id;
        "#
        );
        let response = client
            .post(&low_cardinality_uuid_url)
            .body(create_model_inference_query)
            .send()
            .await
            .expect("Failed to create ModelInference table");
        assert_eq!(response.status(), StatusCode::OK);
    }

    async fn get(&self) -> ClickHouseConnectionInfo {
        let db_num = self.next_db.fetch_add(1, Ordering::Relaxed);
        let db_name = format!("test_{db_num}");
        let clickhouse_connection_info = ClickHouseConnectionInfo::new(&self.url, Some(&db_name));
        self.setup_database(&clickhouse_connection_info, &db_name)
            .await;
        clickhouse_connection_info
    }
}

static CLICKHOUSE_CONNECTION_POOL: OnceCell<ClickHouseConnectionPool> = OnceCell::const_new();

pub async fn get_clickhouse_connection_info() -> ClickHouseConnectionInfo {
    let pool = CLICKHOUSE_CONNECTION_POOL
        .get_or_init(ClickHouseConnectionPool::new)
        .await;

    pool.get().await
}

pub async fn count_table_rows(
    client: &reqwest::Client,
    connection_info: &ClickHouseConnectionInfo,
    table_name: &str,
) -> usize {
    clickhouse_flush_async_insert(client, connection_info).await;

    // In case we re-use an inference ID let's take the most recent
    let query = format!("SELECT COUNT(*) FROM {table_name}");
    let response = client
        .post(connection_info.url.as_str())
        .body(query)
        .send()
        .await
        .expect("Failed to query Clickhouse");
    let text = response
        .text()
        .await
        .expect("Failed to parse ClickHouse response.");
    let rows: usize = text.trim().parse().unwrap();
    rows
}

pub async fn clickhouse_flush_async_insert(
    client: &reqwest::Client,
    connection_info: &ClickHouseConnectionInfo,
) {
    client
        .post(connection_info.url.as_str())
        .body("SYSTEM FLUSH ASYNC INSERT QUEUE")
        .send()
        .await
        .expect("Failed to flush ClickHouse");
}
