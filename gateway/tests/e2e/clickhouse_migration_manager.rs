use gateway::clickhouse::ClickHouseConnectionInfo;
use gateway::clickhouse_migration_manager;
use tracing_test::traced_test;

lazy_static::lazy_static! {
    static ref CLICKHOUSE_URL: String = std::env::var("CLICKHOUSE_URL").expect("Environment variable CLICKHOUSE_URL must be set");
}

#[tokio::test]
#[traced_test]
async fn test_clickhouse_migration_manager() {
    let clickhouse_connection_info =
        ClickHouseConnectionInfo::new(&CLICKHOUSE_URL, false, None).unwrap();

    // Run the migration manager again (it should've already been run on gateway startup)... there should be no changes
    clickhouse_migration_manager::run(&clickhouse_connection_info)
        .await
        .unwrap();

    // TODO (#69): We need to check these when applying the migrations for the first time (i.e. need customd database).
    assert!(!logs_contain("Applying migration"));
    assert!(!logs_contain("Failed to apply migration"));
    assert!(!logs_contain("Failed migration success check"));
    assert!(!logs_contain("Failed to verify migration"));
}

// TODO (#69, more): Add a test that applies individual migrations in a cumulative (N^2) way.
//
//       In sequence:
//       - Migration0000
//       - Migration0000 (noop), Migration0001
//       - Migration0000 (noop), Migration0001 (noop), Migration0002
//
//       We need to check that previous migrations return false for should_apply() (i.e. are noops).
//
//       This will require the ability to create new databases for tests.
