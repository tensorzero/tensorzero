use gateway::clickhouse::ClickHouseConnectionInfo;
use gateway::clickhouse_migration_manager;
use gateway::clickhouse_migration_manager::migrations::migration_0000::Migration0000;
use gateway::clickhouse_migration_manager::migrations::migration_0001::Migration0001;
use reqwest::Client;
use tracing_test::traced_test;

use crate::common::CLICKHOUSE_URL;

#[tokio::test]
async fn test_clickhouse_migration_manager() {
    let database = format!(
        "tensorzero_e2e_tests_migration_manager_{}",
        uuid::Uuid::now_v7().simple()
    );
    let mut clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_query(Some(format!("database={}", database).as_str()));

    let clickhouse = ClickHouseConnectionInfo::Production {
        database_url: clickhouse_url,
        database: database.clone(),
        client: Client::new(),
    };

    // NOTE:
    // We need to split the test into two sub-functions so we can reset `traced_test`'s subscriber between each call.
    // Otherwise, `logs_contain` will return true if the first call triggers a log message that the second call shouldn't trigger.

    #[traced_test]
    async fn first(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager for the first time... it should apply the migrations
        clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
            .await
            .unwrap();

        assert!(logs_contain("Applying migration: Migration0000"));
        assert!(logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));
    }

    #[traced_test]
    async fn second(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0001 { clickhouse })
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
        clickhouse_migration_manager::run_migration(&Migration0000 { clickhouse })
            .await
            .unwrap();
        clickhouse_migration_manager::run_migration(&Migration0001 { clickhouse })
            .await
            .unwrap();
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));

        assert!(!logs_contain("Applying migration: Migration0001"));
        assert!(!logs_contain("Migration succeeded: Migration0001"));
    }

    first(&clickhouse).await;
    second(&clickhouse).await;
    third(&clickhouse).await;

    tracing::info!("Attempting to drop test database: {database}");

    clickhouse
        .run_query(format!("DROP DATABASE {database}"))
        .await
        .unwrap();
}
