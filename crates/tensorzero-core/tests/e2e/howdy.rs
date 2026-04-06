use std::collections::HashSet;
use std::sync::Arc;

use crate::clickhouse::get_clean_clickhouse;
use crate::utils::skip_for_postgres;
use arc_swap::ArcSwap;
use googletest::prelude::*;
use serde_json::json;
use tensorzero::ClientBuilder;
use tensorzero::FeedbackParams;
use tensorzero::InferenceOutput;
use tensorzero::InputMessage;
use tensorzero::InputMessageContent;
use tensorzero::Role;
use tensorzero::{ClientInferenceParams, Input};
use tensorzero_core::config::{Config, ConfigFileGlob, RuntimeOverlay, UninitializedConfig};
use tensorzero_core::db::HowdyQueries;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::clickhouse::migration_manager;
use tensorzero_core::db::clickhouse::migration_manager::RunMigrationManagerArgs;
use tensorzero_core::db::delegating_connection::{DelegatingDatabaseConnection, PrimaryDatastore};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::valkey::ValkeyConnectionInfo;
use tensorzero_core::howdy::{get_deployment_id, get_howdy_report};
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::{Arguments, System, Template, Text};
use tensorzero_core::utils::gateway::GatewayHandle;
use tokio::time::Duration;
use uuid::Uuid;

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_deployment_id() {
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let primary_datastore = PrimaryDatastore::from_test_env();
    let deployment_id = get_deployment_id(&db.clickhouse, &db.postgres, primary_datastore).await;
    expect_that!(deployment_id, ok(not(eq(""))));
}

async fn get_embedded_client(clickhouse: ClickHouseConnectionInfo) -> tensorzero::Client {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("tests/e2e/config/tensorzero.*.toml");
    let config = Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(&config_path).unwrap(),
            false,
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests(),
    );
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: false,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();
    let handle = GatewayHandle::new_with_database_and_http_client(
        config,
        Arc::new(ArcSwap::from_pointee(UninitializedConfig::default())),
        Arc::new(RuntimeOverlay::default()),
        clickhouse,
        PostgresConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        TensorzeroHttpClient::new_testing().unwrap(),
        None,
        HashSet::new(), // available_tools
        HashSet::new(), // tool_whitelist
    )
    .await
    .unwrap();
    ClientBuilder::build_from_state(handle).unwrap()
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_get_howdy_report() {
    // This won't work for Postgres because of 2 reasons:
    // 1. we can't easily set up an isolated Postgres database for a clean start, because of pgcron
    // requirements;
    // 2. queries from Postgres use the refreshed rollup tables, which are not live updated after
    // each request.
    skip_for_postgres!();
    let (clickhouse, _guard) = get_clean_clickhouse(true).await;
    let client = get_embedded_client(clickhouse.clone()).await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let deployment_id = get_deployment_id(
        &clickhouse,
        &PostgresConnectionInfo::Disabled,
        PrimaryDatastore::ClickHouse,
    )
    .await
    .unwrap();
    let howdy_report = get_howdy_report(
        Some(&clickhouse as &(dyn HowdyQueries + Sync)),
        Some(deployment_id.as_str()),
        PrimaryDatastore::ClickHouse,
        Uuid::now_v7(),
    )
    .await
    .unwrap();
    assert_eq!(howdy_report.inference_count, "0");
    assert_eq!(howdy_report.feedback_count, "0");
    assert!(howdy_report.input_token_total.is_none());
    assert!(howdy_report.output_token_total.is_none());
    assert_eq!(
        howdy_report.gateway_version,
        tensorzero_core::endpoints::status::TENSORZERO_VERSION
    );
    assert_eq!(
        howdy_report.observability_backend,
        PrimaryDatastore::ClickHouse,
        "observability_backend should be ClickHouse"
    );
    // Since we're in an e2e test, this should be true
    assert!(howdy_report.dryrun);
    // Send a chat inference and comment feedback
    let params = ClientInferenceParams {
        function_name: Some("basic_test".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "AskJeeves"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let InferenceOutput::NonStreaming(response) = client.inference(params).await.unwrap() else {
        panic!("Expected a non-streaming response");
    };
    // Send comment feedback
    let params = FeedbackParams {
        episode_id: Some(response.episode_id()),
        metric_name: "comment".to_string(),
        value: json!("good job!"),
        ..Default::default()
    };
    client.feedback(params).await.unwrap();
    // Send a json inference and boolean feedback
    let params = ClientInferenceParams {
        function_name: Some("json_success".to_string()),
        input: Input {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "AskJeeves"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(
                        json!({
                            "country": "Japan",
                        })
                        .as_object()
                        .unwrap()
                        .clone(),
                    ),
                })],
            }],
        },
        ..Default::default()
    };

    let InferenceOutput::NonStreaming(response) = client.inference(params).await.unwrap() else {
        panic!("Expected a non-streaming response");
    };
    // Send boolean feedback
    let params = FeedbackParams {
        inference_id: Some(response.inference_id()),
        metric_name: "task_success".to_string(),
        value: json!(true),
        ..Default::default()
    };
    client.feedback(params).await.unwrap();
    // Sleep for 1 second to ensure the feedback is processed
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Get the howdy report again
    let new_howdy_report = get_howdy_report(
        Some(&clickhouse as &(dyn HowdyQueries + Sync)),
        Some(deployment_id.as_str()),
        PrimaryDatastore::ClickHouse,
        Uuid::now_v7(),
    )
    .await
    .unwrap();
    assert!(!new_howdy_report.inference_count.is_empty());
    assert!(!new_howdy_report.feedback_count.is_empty());
    // Since we're in an e2e test, this should be true
    assert!(new_howdy_report.dryrun);

    // Assert that the parsed inference and feedback counts are greater than the old ones
    assert_eq!(new_howdy_report.inference_count, "2");
    assert_eq!(new_howdy_report.feedback_count, "2");
    // Assert that the token counts are also positive now
    let input_token_total: u64 = new_howdy_report.input_token_total.unwrap().parse().unwrap();
    assert_eq!(input_token_total, 20);
    let output_token_total: u64 = new_howdy_report
        .output_token_total
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(output_token_total, 2);
}
