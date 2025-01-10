use gateway::clickhouse::ClickHouseConnectionInfo;
use gateway::clickhouse_migration_manager;
use gateway::clickhouse_migration_manager::migrations::migration_0000::Migration0000;
use gateway::clickhouse_migration_manager::migrations::migration_0002::Migration0002;
use gateway::clickhouse_migration_manager::migrations::migration_0003::Migration0003;
use gateway::clickhouse_migration_manager::migrations::migration_0004::Migration0004;
use gateway::clickhouse_migration_manager::migrations::migration_0005::Migration0005;
use gateway::clickhouse_migration_manager::migrations::migration_0006::Migration0006;
use gateway::clickhouse_migration_manager::migrations::migration_0007::Migration0007;
use gateway::clickhouse_migration_manager::migrations::migration_0008::Migration0008;
use gateway::clickhouse_migration_manager::migrations::migration_0009::Migration0009;
use gateway::clickhouse_migration_manager::migrations::migration_0010::Migration0010;
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
    clickhouse.create_database().await.unwrap();
    // Run it twice to test that it is a no-op the second time
    clickhouse.create_database().await.unwrap();
    // NOTE:
    // We need to split the test into sub-functions so we can reset `traced_test`'s subscriber between each call.
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
        assert!(!logs_contain("ERROR"));
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
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(logs_contain("Applying migration: Migration0002"));
        assert!(logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("ERROR"));
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
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(logs_contain("Applying migration: Migration0003"));
        assert!(logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("ERROR"));
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
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(logs_contain("Applying migration: Migration0004"));
        assert!(logs_contain("Migration succeeded: Migration0004"));
        assert!(!logs_contain("ERROR"));
    }

    #[traced_test]
    async fn fifth(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0005 { clickhouse })
            .await
            .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("Applying migration: Migration0004"));
        assert!(!logs_contain("Migration succeeded: Migration0004"));
        assert!(logs_contain("Applying migration: Migration0005"));
        assert!(logs_contain("Migration succeeded: Migration0005"));
        assert!(!logs_contain("ERROR"));
    }

    #[traced_test]
    async fn sixth(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0005 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0006 { clickhouse })
            .await
            .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("Applying migration: Migration0004"));
        assert!(!logs_contain("Migration succeeded: Migration0004"));
        assert!(!logs_contain("Applying migration: Migration0005"));
        assert!(!logs_contain("Migration succeeded: Migration0005"));
        assert!(logs_contain("Applying migration: Migration0006"));
        assert!(logs_contain("Migration succeeded: Migration0006"));
    }

    #[traced_test]
    async fn seventh(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0005 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0006 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0007 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("Applying migration: Migration0004"));
        assert!(!logs_contain("Migration succeeded: Migration0004"));
        assert!(!logs_contain("Applying migration: Migration0005"));
        assert!(!logs_contain("Migration succeeded: Migration0005"));
        assert!(!logs_contain("Applying migration: Migration0006"));
        assert!(!logs_contain("Migration succeeded: Migration0006"));
        assert!(logs_contain("Applying migration: Migration0007"));
        assert!(logs_contain("Migration succeeded: Migration0007"));
    }

    #[traced_test]
    async fn eighth(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0005 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0006 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0007 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0008 { clickhouse })
            .await
            .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("Applying migration: Migration0004"));
        assert!(!logs_contain("Migration succeeded: Migration0004"));
        assert!(!logs_contain("Applying migration: Migration0005"));
        assert!(!logs_contain("Migration succeeded: Migration0005"));
        assert!(!logs_contain("Applying migration: Migration0006"));
        assert!(!logs_contain("Migration succeeded: Migration0006"));
        assert!(!logs_contain("Applying migration: Migration0007"));
        assert!(!logs_contain("Migration succeeded: Migration0007"));
        assert!(logs_contain("Applying migration: Migration0008"));
        assert!(logs_contain("Migration succeeded: Migration0008"));
    }

    #[traced_test]
    async fn ninth(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0005 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0006 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0007 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0008 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0009 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("Applying migration: Migration0004"));
        assert!(!logs_contain("Migration succeeded: Migration0004"));
        assert!(!logs_contain("Applying migration: Migration0005"));
        assert!(!logs_contain("Migration succeeded: Migration0005"));
        assert!(!logs_contain("Applying migration: Migration0006"));
        assert!(!logs_contain("Migration succeeded: Migration0006"));
        assert!(!logs_contain("Applying migration: Migration0007"));
        assert!(!logs_contain("Migration succeeded: Migration0007"));
        assert!(!logs_contain("Applying migration: Migration0008"));
        assert!(!logs_contain("Migration succeeded: Migration0008"));
        assert!(logs_contain("Applying migration: Migration0009"));
        assert!(logs_contain("Migration succeeded: Migration0009"));
    }

    #[traced_test]
    async fn tenth(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0005 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0006 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0007 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0008 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0009 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0010 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("Applying migration: Migration0004"));
        assert!(!logs_contain("Migration succeeded: Migration0004"));
        assert!(!logs_contain("Applying migration: Migration0005"));
        assert!(!logs_contain("Migration succeeded: Migration0005"));
        assert!(!logs_contain("Applying migration: Migration0006"));
        assert!(!logs_contain("Migration succeeded: Migration0006"));
        assert!(!logs_contain("Applying migration: Migration0007"));
        assert!(!logs_contain("Migration succeeded: Migration0007"));
        assert!(!logs_contain("Applying migration: Migration0008"));
        assert!(!logs_contain("Migration succeeded: Migration0008"));
        assert!(!logs_contain("Applying migration: Migration0009"));
        assert!(!logs_contain("Migration succeeded: Migration0009"));
        assert!(logs_contain("Applying migration: Migration0010"));
        assert!(logs_contain("Migration succeeded: Migration0010"));
    }

    #[traced_test]
    async fn eleventh(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        let clean_start =
            clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
                .await
                .unwrap();
        // We know that the first migration was run so clean start should be false
        assert!(!clean_start);
        clickhouse_migration_manager::run_migration(&Migration0002 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0003 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0004 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0005 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0006 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0007 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0008 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0009 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0010 {
            clickhouse,
            clean_start: true,
        })
        .await
        .unwrap();

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Applying migration: Migration0002"));
        assert!(!logs_contain("Migration succeeded: Migration0002"));
        assert!(!logs_contain("Applying migration: Migration0003"));
        assert!(!logs_contain("Migration succeeded: Migration0003"));
        assert!(!logs_contain("Applying migration: Migration0004"));
        assert!(!logs_contain("Migration succeeded: Migration0004"));
        assert!(!logs_contain("Applying migration: Migration0005"));
        assert!(!logs_contain("Migration succeeded: Migration0005"));
        assert!(!logs_contain("Applying migration: Migration0006"));
        assert!(!logs_contain("Migration succeeded: Migration0006"));
        assert!(!logs_contain("Applying migration: Migration0007"));
        assert!(!logs_contain("Migration succeeded: Migration0007"));
        assert!(!logs_contain("Applying migration: Migration0008"));
        assert!(!logs_contain("Migration succeeded: Migration0008"));
        assert!(!logs_contain("Applying migration: Migration0009"));
        assert!(!logs_contain("Migration succeeded: Migration0009"));
        assert!(!logs_contain("Applying migration: Migration0010"));
        assert!(!logs_contain("Migration succeeded: Migration0010"));
    }

    first(&clickhouse).await;
    second(&clickhouse).await;
    third(&clickhouse).await;
    fourth(&clickhouse).await;
    fifth(&clickhouse).await;
    sixth(&clickhouse).await;
    seventh(&clickhouse).await;
    eighth(&clickhouse).await;
    ninth(&clickhouse).await;
    tenth(&clickhouse).await;
    eleventh(&clickhouse).await;
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
        .write(&[payload], "BooleanMetricFeedback")
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
