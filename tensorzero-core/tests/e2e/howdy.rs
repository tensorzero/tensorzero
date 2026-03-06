use std::collections::HashSet;
use std::sync::Arc;

use crate::clickhouse::get_clean_clickhouse;
use crate::utils::skip_for_postgres;
use googletest::prelude::*;
use serde_json::json;
use tensorzero::ClientBuilder;
use tensorzero::FeedbackParams;
use tensorzero::InferenceOutput;
use tensorzero::InputMessage;
use tensorzero::InputMessageContent;
use tensorzero::Role;
use tensorzero::{ClientInferenceParams, Input};
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::clickhouse::migration_manager;
use tensorzero_core::db::clickhouse::migration_manager::RunMigrationManagerArgs;
use tensorzero_core::db::delegating_connection::{DelegatingDatabaseConnection, PrimaryDatastore};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::valkey::ValkeyConnectionInfo;
use tensorzero_core::howdy::HowdyReportBody;
use tensorzero_core::howdy::{get_deployment_id, get_howdy_report};
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::{Arguments, System, Template, Text};
use tensorzero_core::utils::gateway::GatewayHandle;
use tokio::time::Duration;

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

    // Swap in the custom clean clickhouse.
    let mut db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    db.clickhouse = clickhouse.clone();

    let primary_datastore = PrimaryDatastore::from_test_env();

    let initial_report = get_howdy_report(&db, "test_deployment_id", primary_datastore)
        .await
        .unwrap();

    expect_that!(
        initial_report,
        matches_pattern!(HowdyReportBody {
            gateway_version: eq(&tensorzero_core::endpoints::status::TENSORZERO_VERSION),
            observability_backend: eq(&primary_datastore),
            dryrun: eq(&true),
            ..
        })
    );

    // Capture values for later comparison
    let initial_inferences: u64 = initial_report.inference_count.parse().unwrap();
    let initial_feedbacks: u64 = initial_report.feedback_count.parse().unwrap();
    let initial_input_tokens: u64 = initial_report
        .input_token_total
        .as_deref()
        .unwrap_or("0")
        .parse()
        .unwrap();
    let initial_output_tokens: u64 = initial_report
        .output_token_total
        .as_deref()
        .unwrap_or("0")
        .parse()
        .unwrap();

    // Send a chat inference and comment feedback
    let client = get_embedded_client(clickhouse).await;
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

    // Get the howdy report again and verify deltas
    let new_report = get_howdy_report(&db, "test_deployment_id", primary_datastore).await;
    expect_that!(
        new_report,
        ok(matches_pattern!(HowdyReportBody {
            inference_count: predicate(
                // inference count should increase by 2
                |count: &String| count.parse::<u64>().unwrap() == initial_inferences + 2
            ),
            feedback_count: predicate(
                // feedback count should increase by 2
                |count: &String| count.parse::<u64>().unwrap() == initial_feedbacks + 2
            ),
            input_token_total: some(predicate(
                // input tokens should increase by 20
                |tokens: &String| tokens.parse::<u64>().unwrap() == initial_input_tokens + 20
            )),
            output_token_total: some(predicate(
                // output tokens should increase by 2
                |tokens: &String| tokens.parse::<u64>().unwrap() == initial_output_tokens + 2
            )),
            gateway_version: eq(&tensorzero_core::endpoints::status::TENSORZERO_VERSION),
            observability_backend: eq(&primary_datastore),
            dryrun: eq(&true),
            ..
        }))
    );
}
