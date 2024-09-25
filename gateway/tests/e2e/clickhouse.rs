use gateway::clickhouse_migration_manager;
use gateway::clickhouse_migration_manager::migrations::migration_0000::Migration0000;
use gateway::clickhouse_migration_manager::migrations::migration_0002::Migration0002;
use gateway::{
    clickhouse::ClickHouseConnectionInfo,
    clickhouse_migration_manager::migrations::migration_0001::Migration0001,
};
use reqwest::Client;
use serde_json::json;
use tracing_test::traced_test;
use uuid::Uuid;

use crate::common::{get_clickhouse, CLICKHOUSE_URL};

fn get_clean_clickhouse() -> ClickHouseConnectionInfo {
    let database = format!(
        "tensorzero_e2e_tests_migration_manager_{}",
        Uuid::now_v7().simple()
    );
    let mut clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path("");
    clickhouse_url.set_query(Some(format!("database={}", database).as_str()));

    ClickHouseConnectionInfo::Production {
        database_url: clickhouse_url,
        database: database.clone(),
        client: Client::new(),
    }
}

#[tokio::test]
async fn test_clickhouse_migration_manager() {
    let clickhouse = get_clean_clickhouse();
    // NOTE:
    // We need to split the test into two sub-functions so we can reset `traced_test`'s subscriber between each call.
    // Otherwise, `logs_contain` will return true if the first call triggers a log message that the second call shouldn't trigger.

    #[traced_test]
    async fn first(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager for the first time... it should apply the migrations
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();

        assert!(clean_start);
        assert!(logs_contain("Applying migration: Migration0000"));
        assert!(logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));
    }

    #[traced_test]
    async fn second(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0001 {
            clickhouse,
            clean_start: true, // For testing purposes, we know there is no data to migrate and it is a clean start
        })
        .await
        .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(logs_contain("Applying migration: Migration0001"));
        assert!(logs_contain("Migration succeeded: Migration0001"));
    }

    #[traced_test]
    async fn third(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0001 {
            clickhouse,
            clean_start: true, // For testing purposes, we know there is no data to migrate and it is a clean start
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0001"));
        assert!(!logs_contain("Migration succeeded: Migration0001"));
        assert!(logs_contain("Applying migration: Migration0002"));
        assert!(logs_contain("Migration succeeded: Migration0002"));
    }

    #[traced_test]
    async fn fourth(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0001 {
            clickhouse,
            clean_start: true, // For testing purposes, we know there is no data to migrate and it is a clean start
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0001"));
        assert!(!logs_contain("Migration succeeded: Migration0001"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
    }

    first(&clickhouse).await;
    second(&clickhouse).await;
    third(&clickhouse).await;
    fourth(&clickhouse).await;
    let database = clickhouse.database();
    tracing::info!("Attempting to drop test database: {database}");

    clickhouse
        .run_query(format!("DROP DATABASE {database}"))
        .await
        .unwrap();
}

#[tokio::test]
async fn test_bad_clickhouse_write() {
    let clickhouse = get_clickhouse().await;
    // "name" should be "metric_name" here but we are using the wrong field on purpose to check that the write fails
    let payload =
        json!({"target_id": Uuid::now_v7(), "value": true, "name": "test", "id": Uuid::now_v7()});
    let err = clickhouse
        .write(&payload, "BooleanMetricFeedback")
        .await
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("Unknown field found while parsing JSONEachRow format: name"));
}

#[tokio::test]
async fn test_clean_clickhouse_start() {
    let clickhouse = get_clean_clickhouse();
    let start = std::time::Instant::now();
    clickhouse_migration_manager::run(&clickhouse)
        .await
        .unwrap();
    let duration = start.elapsed();
    assert!(
        duration < std::time::Duration::from_secs(10),
        "Migrations took longer than 10 seconds: {duration:?}"
    );
}
