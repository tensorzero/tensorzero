use gateway::clickhouse::ClickHouseConnectionInfo;
use gateway::clickhouse_migration_manager;
use gateway::clickhouse_migration_manager::migration_trait::Migration;
use tracing_test::traced_test;

use crate::common::CLICKHOUSE_URL;

#[tokio::test]
async fn test_clickhouse_migration_manager() {
    let database = format!(
        "tensorzero_e2e_tests_migration_manager_{}",
        uuid::Uuid::now_v7().simple()
    );

    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, &database).unwrap();

    // NOTE:
    // We need to split the test into two sub-functions so we can reset `traced_test`'s subscriber between each call.
    // Otherwise, `logs_contain` will return true if the first call triggers a log message that the second call shouldn't trigger.

    #[traced_test]
    async fn first(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager for the first time... it should apply the migrations
        clickhouse_migration_manager::run(clickhouse).await.unwrap();

        assert!(logs_contain("Applying migration: Migration0000"));
        assert!(logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));
    }

    #[traced_test]
    async fn second(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        clickhouse_migration_manager::run(clickhouse).await.unwrap();

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));
    }

    first(&clickhouse).await;
    second(&clickhouse).await;

    tracing::info!("Attempting to drop test database: {database}");

    clickhouse
        .run_query(format!("DROP DATABASE {database}"))
        .await
        .unwrap();
}

#[tokio::test]
async fn thorough_test_clickhouse_migration_manager() {
    let database = format!(
        "tensorzero_e2e_tests_migration_manager_{}",
        uuid::Uuid::now_v7().simple()
    );

    let clickhouse = ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, &database).unwrap();

    let migrations = clickhouse_migration_manager::get_migrations(&clickhouse);

    // Loop over every prefix of migrations and print them
    for i in 1..=migrations.len() {
        let prefix = &migrations[..i];
        run_migrations_only_last_should_occur(prefix).await;
        println!("finished running migration: {}", i - 1);
    }

    #[traced_test]
    async fn run_migrations_only_last_should_occur(migrations: &[Box<dyn Migration>]) {
        for (i, migration) in migrations.iter().enumerate() {
            clickhouse_migration_manager::run_migration(migration, i)
                .await
                .unwrap();
        }

        // For all but the last migration, we should not have applied them because they should
        // already have been applied
        for i in 0..migrations.len() - 1 {
            let migration_name = format!("Migration{:04}", i);
            println!("Migration name: {}", migration_name);
            assert!(!logs_contain(&format!(
                "Applying migration: {}",
                migration_name
            )));
            assert!(!logs_contain(&format!(
                "Migration succeeded: {}",
                migration_name
            )));
            assert!(!logs_contain(&format!(
                "Failed to apply migration: {}",
                migration_name
            )));
            assert!(!logs_contain(&format!(
                "Failed migration success check: {}",
                migration_name
            )));
            assert!(!logs_contain(&format!(
                "Failed to verify migration: {}",
                migration_name
            )));
        }

        // The last migration should have been applied
        let last_migration_name = format!("Migration{:04}", migrations.len() - 1);
        assert!(logs_contain(&format!(
            "Applying migration: {}",
            last_migration_name
        )));
        assert!(logs_contain(&format!(
            "Migration succeeded: {}",
            last_migration_name
        )));
        assert!(!logs_contain(&format!(
            "Failed to apply migration: {}",
            last_migration_name
        )));
        assert!(!logs_contain(&format!(
            "Failed migration success check: {}",
            last_migration_name
        )));
        assert!(!logs_contain(&format!(
            "Failed to verify migration: {}",
            last_migration_name
        )));
    }

    // NOTE:
    // We need to split the test into two sub-functions so we can reset `traced_test`'s subscriber between each call.
    // Otherwise, `logs_contain` will return true if the first call triggers a log message that the second call shouldn't trigger.

    #[traced_test]
    async fn first(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager for the first time... it should apply the migrations
        clickhouse_migration_manager::run(clickhouse).await.unwrap();

        assert!(logs_contain("Applying migration: Migration0000"));
        assert!(logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));
    }

    #[traced_test]
    async fn second(clickhouse: &ClickHouseConnectionInfo) {
        // Run the migration manager again (it should've already been run above)... there should be no changes
        clickhouse_migration_manager::run(clickhouse).await.unwrap();

        assert!(!logs_contain("Applying migration: Migration0000"));
        assert!(!logs_contain("Migration succeeded: Migration0000"));
        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));
    }

    first(&clickhouse).await;
    second(&clickhouse).await;

    tracing::info!("Attempting to drop test database: {database}");

    clickhouse
        .run_query(format!("DROP DATABASE {database}"))
        .await
        .unwrap();
}
