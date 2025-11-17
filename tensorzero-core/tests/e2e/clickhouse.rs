#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::cell::Cell;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use paste::paste;
use secrecy::ExposeSecret;
use serde_json::json;
use tensorzero_core::utils::testing::reset_capture_logs;
use tokio::runtime::Handle;
use tokio::time::sleep;
use uuid::Uuid;

use tensorzero_core::config::BatchWritesConfig;
use tensorzero_core::db::clickhouse::migration_manager::migration_trait::Migration;
use tensorzero_core::db::clickhouse::migration_manager::migrations::check_table_exists;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0000::Migration0000;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0002::Migration0002;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0003::Migration0003;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0004::Migration0004;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0005::Migration0005;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0006::Migration0006;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0008::Migration0008;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0009::Migration0009;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0011::Migration0011;
use tensorzero_core::db::clickhouse::migration_manager::migrations::migration_0013::Migration0013;
use tensorzero_core::db::clickhouse::migration_manager::MigrationTableState;
use tensorzero_core::db::feedback::FeedbackQueries;
use tensorzero_core::inference::types::ModelInferenceDatabaseInsert;

use tensorzero_core::db::clickhouse::migration_manager::{
    self, get_all_migration_records, make_all_migrations, MigrationRecordDatabaseInsert,
    RunMigrationArgs, RunMigrationManagerArgs,
};
use tensorzero_core::db::clickhouse::test_helpers::{get_clickhouse, CLICKHOUSE_URL};
use tensorzero_core::db::clickhouse::{ClickHouseConnectionInfo, Rows, TableName};
use tensorzero_core::endpoints::status::TENSORZERO_VERSION;
use tensorzero_core::error::{Error, ErrorDetails};

pub struct DeleteDbOnDrop {
    database: String,
    client: ClickHouseConnectionInfo,
    allow_db_missing: bool,
}

impl Drop for DeleteDbOnDrop {
    fn drop(&mut self) {
        eprintln!("Dropping database: {}", self.database);
        let client = self.client.clone();
        let database = self.database.clone();
        let allow_db_missing = self.allow_db_missing;
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async move {
                if allow_db_missing {
                    client
                        .run_query_synchronous_no_params(format!(
                            "DROP DATABASE IF EXISTS {database} SYNC"
                        ))
                        .await
                        .unwrap();
                } else {
                    client
                        .run_query_synchronous_no_params(format!("DROP DATABASE {database} SYNC"))
                        .await
                        .unwrap();
                }
                eprintln!("Database dropped: {database}");
            });
        });
    }
}

/// Creates a fresh ClickHouse database.
/// Returns a `ClickHouseConnectionInfo` for the db, along with a `DeleteDbOnDrop`
/// that deletes the database when the `DeleteDbOnDrop` is dropped (which will
/// happen even if the test panics).
/// This helps to reduce peak disk usage on CI.
/// If `allow_db_missing` is true, then we'll use 'DROP DATABASE IF EXISTS' instead of 'DROP DATABASE'
pub async fn get_clean_clickhouse(
    allow_db_missing: bool,
) -> (ClickHouseConnectionInfo, DeleteDbOnDrop) {
    let database = format!(
        "tensorzero_e2e_tests_migration_manager_{}",
        Uuid::now_v7().simple()
    );
    let mut clickhouse_url = url::Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path(&database);
    let clickhouse_url = clickhouse_url.to_string();
    let clickhouse = ClickHouseConnectionInfo::new(
        &clickhouse_url,
        BatchWritesConfig {
            enabled: false,
            __force_allow_embedded_batch_writes: false,
            flush_interval_ms: 1000,
            max_rows: 100,
        },
    )
    .await
    .unwrap();

    (
        clickhouse.clone(),
        DeleteDbOnDrop {
            database,
            client: clickhouse,
            allow_db_missing,
        },
    )
}

macro_rules! invoke_all_separate_tests {
    ($target_fn:ident, $prefix:ident, [$($migration_num:literal),*]) => {
        // For each value in the literal array, generate a new `#[tokio::test]` function
        // that calls the target function with that value
        // This lets us use `capture_logs()` to independently capture logs for each test
        const _MIGRATIONS_NUM_ARRAY: [usize; tensorzero_core::db::clickhouse::migration_manager::NUM_MIGRATIONS] = [$($migration_num),*];
        $(
            paste! {
                #[tokio::test(flavor = "multi_thread")]
                async fn [<$prefix $migration_num>] () {
                    // Verify that the literal array matches the migrations array
                    for i in 0.._MIGRATIONS_NUM_ARRAY.len() {
                        assert_eq!(_MIGRATIONS_NUM_ARRAY[i], i, "The migration indices array should be a sequential list of numbers");
                    }
                    $target_fn($migration_num).await;
                }
            }

        )*
    }
}

const MANIFEST_PATH: &str = env!("CARGO_MANIFEST_DIR");

async fn count_table_rows(clickhouse: &ClickHouseConnectionInfo, table: &str) -> u64 {
    clickhouse
        .run_query_synchronous_no_params(format!("SELECT count(*) FROM {table}"))
        .await
        .unwrap()
        .response
        .trim()
        .parse()
        .unwrap()
}

async fn insert_large_fixtures(clickhouse: &ClickHouseConnectionInfo) {
    // Insert data so that we test the migration re-creates the tables properly.
    let s3_fixtures_path = std::env::var("TENSORZERO_S3_FIXTURES_PATH")
        .unwrap_or_else(|_| format!("{MANIFEST_PATH}/../ui/fixtures/s3-fixtures"));
    let s3_fixtures_path = &s3_fixtures_path;

    let database_url = clickhouse.database_url();
    let database = clickhouse.database();
    let url = url::Url::parse(database_url.expose_secret()).unwrap();
    let mut host = url.host_str().unwrap();
    if host == "localhost" || host == "127.0.0.1" {
        host = "host.docker.internal";
    }
    let username = url.username();
    let password = urlencoding::decode(url.password().unwrap_or(""))
        .unwrap()
        .to_string();

    // We use our latest fixtures - new columns will get ignored when inserting.
    let insert_futures = [
        ("large_chat_inference_v2.parquet", "ChatInference"),
        ("large_json_inference_v2.parquet", "JsonInference"),
        ("large_chat_model_inference_v2.parquet", "ModelInference"),
        ("large_json_model_inference_v2.parquet", "ModelInference"),
        (
            "large_chat_boolean_feedback.parquet",
            "BooleanMetricFeedback",
        ),
        (
            "large_json_boolean_feedback.parquet",
            "BooleanMetricFeedback",
        ),
        ("large_chat_comment_feedback.parquet", "CommentFeedback"),
        ("large_json_comment_feedback.parquet", "CommentFeedback"),
        (
            "large_chat_demonstration_feedback.parquet",
            "DemonstrationFeedback",
        ),
        (
            "large_json_demonstration_feedback.parquet",
            "DemonstrationFeedback",
        ),
        ("large_chat_float_feedback.parquet", "FloatMetricFeedback"),
        ("large_json_float_feedback.parquet", "FloatMetricFeedback"),
    ]
    .into_iter()
    .map(|(file, table)| {
        let password = password.clone();
        async move {
            // If we are running in CI (TENSORZERO_CI=1), we should have the clickhouse client installed locally
            // so we should not use Docker
            let mut command = if std::env::var("TENSORZERO_CI").is_ok() {
                let mut cmd = tokio::process::Command::new("clickhouse-client");
                cmd.args([
                    "--host",
                    host,
                    "--user",
                    username,
                    "--password",
                    &password,
                    "--database",
                    database,
                    "--query",
                    &format!("INSERT INTO {table} SELECT * FROM file('{file}', 'Parquet')"),
                ]);
                cmd
            } else {
                // If we are running locally, we should use docker so that we can
                // be platform independent in how we insert these files into ClickHouse.
                let mut cmd = tokio::process::Command::new("docker");
                cmd.args([
                    "run",
                    "--add-host=host.docker.internal:host-gateway",
                    "-v",
                    &format!("{s3_fixtures_path}:/s3-fixtures"),
                    "clickhouse:25.4",
                    "clickhouse-client",
                    "--host",
                    host,
                    "--user",
                    username,
                    "--password",
                    &password,
                    "--database",
                    database,
                    "--query",
                    &format!(
                        r"
        INSERT INTO {table} FROM INFILE '/s3-fixtures/{file}' FORMAT Parquet
    "
                    ),
                ]);
                cmd
            };
            assert!(
                command.spawn().unwrap().wait().await.unwrap().success(),
                "Failed to insert {table}"
            );
        }
    });

    futures::future::join_all(insert_futures).await;
}

async fn run_migration_0009_with_data<R: Future<Output = bool>, F: FnOnce() -> R>(
    clickhouse: &ClickHouseConnectionInfo,
    run_migration: F,
) -> bool {
    let initial_boolean_feedback_count: u64 =
        count_table_rows(clickhouse, "BooleanMetricFeedback").await;
    let initial_comment_feedback_count: u64 = count_table_rows(clickhouse, "CommentFeedback").await;
    let initial_demonstration_feedback_count: u64 =
        count_table_rows(clickhouse, "DemonstrationFeedback").await;
    let initial_float_metric_feedback_count: u64 =
        count_table_rows(clickhouse, "FloatMetricFeedback").await;

    let clean_start = run_migration().await;

    let final_boolean_feedback_count: u64 =
        count_table_rows(clickhouse, "BooleanMetricFeedback").await;
    let final_comment_feedback_count: u64 = count_table_rows(clickhouse, "CommentFeedback").await;
    let final_demonstration_feedback_count: u64 =
        count_table_rows(clickhouse, "DemonstrationFeedback").await;
    let final_float_metric_feedback_count: u64 =
        count_table_rows(clickhouse, "FloatMetricFeedback").await;

    assert_eq!(initial_boolean_feedback_count, final_boolean_feedback_count);
    assert_eq!(initial_comment_feedback_count, final_comment_feedback_count);
    assert_eq!(
        initial_demonstration_feedback_count,
        final_demonstration_feedback_count
    );
    assert_eq!(
        initial_float_metric_feedback_count,
        final_float_metric_feedback_count
    );

    clean_start
}

async fn run_migration_0021_with_data<R: Future<Output = bool>, F: FnOnce() -> R>(
    clickhouse: &ClickHouseConnectionInfo,
    run_migration: F,
) -> bool {
    // Our fixtures apply two tags per inference
    let initial_chat_inference_count: u64 = 2 * count_table_rows(clickhouse, "ChatInference").await;
    let initial_json_inference_count: u64 = 2 * count_table_rows(clickhouse, "JsonInference").await;

    let clean_start = run_migration().await;

    let final_tag_inference_count: u64 = count_table_rows(clickhouse, "TagInference").await;
    assert_eq!(
        initial_chat_inference_count + initial_json_inference_count,
        final_tag_inference_count,
    );

    clean_start
}

async fn run_migration_0020_with_data<R: Future<Output = bool>, F: FnOnce() -> R>(
    clickhouse: &ClickHouseConnectionInfo,
    run_migration: F,
) -> bool {
    // Check that the same number of rows are in the tables before and after the migration
    let initial_chat_count: u64 = count_table_rows(clickhouse, "ChatInference").await;
    let initial_json_count: u64 = count_table_rows(clickhouse, "JsonInference").await;

    let clean_start = run_migration().await;

    let final_chat_count: u64 = count_table_rows(clickhouse, "ChatInference").await;
    let final_json_count: u64 = count_table_rows(clickhouse, "JsonInference").await;
    assert_eq!(
        initial_chat_count, final_chat_count,
        "Lost data from ChatInference"
    );
    assert_eq!(
        initial_json_count, final_json_count,
        "Lost data from JsonInference"
    );

    // Check that existing rows are inserted into InferenceById and InferenceByEpisodeId,
    // and check an individual row from ChatInference and JsonInference.

    let final_inference_by_id_count: u64 =
        count_table_rows(clickhouse, "InferenceById FINAL").await;
    let final_inference_by_episode_id_count: u64 =
        count_table_rows(clickhouse, "InferenceByEpisodeId FINAL").await;

    assert_eq!(
        final_inference_by_id_count,
        final_chat_count + final_json_count,
        "Didn't insert all data into InferenceById"
    );
    assert_eq!(
        final_inference_by_episode_id_count,
        final_chat_count + final_json_count,
        "Didn't insert all data into InferenceByEpisodeId"
    );

    let sample_chat_row = clickhouse
        .run_query_synchronous_no_params(
            "SELECT toUInt128(id) as id_uint, toUInt128(episode_id) as episode_id_uint FROM ChatInference LIMIT 1 FORMAT JSONEachRow".to_string(),
        )
        .await
        .unwrap();

    let sample_chat_row_json =
        serde_json::from_str::<serde_json::Value>(&sample_chat_row.response).unwrap();
    let sample_chat_id = sample_chat_row_json["id_uint"].as_str().unwrap();
    let sample_chat_episode_id = sample_chat_row_json["episode_id_uint"].as_str().unwrap();

    let matching_chat_by_id = clickhouse
        .run_query_synchronous_no_params(
            format!("SELECT id_uint, toUInt128(episode_id) as episode_id_uint FROM InferenceById WHERE function_type = 'chat' AND id_uint = '{sample_chat_id}' LIMIT 1 FORMAT JSONEachRow"),
        )
        .await
        .unwrap();

    println!("Matching chat by id: `{}`", matching_chat_by_id.response);

    let matching_chat_by_id_json =
        serde_json::from_str::<serde_json::Value>(&matching_chat_by_id.response).unwrap();
    assert_eq!(
        matching_chat_by_id_json["episode_id_uint"]
            .as_str()
            .unwrap(),
        sample_chat_episode_id
    );

    let matching_chat_by_episode_id = clickhouse
        .run_query_synchronous_no_params(
            format!("SELECT * FROM InferenceByEpisodeId WHERE function_type = 'chat' AND episode_id_uint = '{sample_chat_episode_id}' LIMIT 1 FORMAT JSONEachRow"),
        )
        .await
        .unwrap();

    let matching_chat_by_episode_id_json =
        serde_json::from_str::<serde_json::Value>(&matching_chat_by_episode_id.response).unwrap();
    assert_eq!(
        matching_chat_by_episode_id_json["id_uint"]
            .as_str()
            .unwrap(),
        sample_chat_id
    );

    let sample_json_row = clickhouse
        .run_query_synchronous_no_params(
            "SELECT toUInt128(id) as id_uint, toUInt128(episode_id) as episode_id_uint FROM JsonInference LIMIT 1 FORMAT JSONEachRow".to_string(),
        )
        .await
        .unwrap();

    let sample_json_row_json =
        serde_json::from_str::<serde_json::Value>(&sample_json_row.response).unwrap();
    let sample_json_id = sample_json_row_json["id_uint"].as_str().unwrap();
    let sample_json_episode_id = sample_json_row_json["episode_id_uint"].as_str().unwrap();

    let matching_json_by_id = clickhouse
        .run_query_synchronous_no_params(
            format!("SELECT id_uint, toUInt128(episode_id) as episode_id_uint FROM InferenceById WHERE function_type = 'json' AND id_uint = '{sample_json_id}' LIMIT 1 FORMAT JSONEachRow"),
        )
        .await
        .unwrap();

    let matching_json_by_id_json =
        serde_json::from_str::<serde_json::Value>(&matching_json_by_id.response).unwrap();
    assert_eq!(
        matching_json_by_id_json["episode_id_uint"]
            .as_str()
            .unwrap(),
        sample_json_episode_id
    );

    let matching_json_by_episode_id = clickhouse
        .run_query_synchronous_no_params(
            format!("SELECT * FROM InferenceByEpisodeId WHERE function_type = 'json' AND episode_id_uint = '{sample_json_episode_id}' LIMIT 1 FORMAT JSONEachRow"),
        )
        .await
        .unwrap();

    let matching_json_by_episode_id_json =
        serde_json::from_str::<serde_json::Value>(&matching_json_by_episode_id.response).unwrap();
    assert_eq!(
        matching_json_by_episode_id_json["id_uint"]
            .as_str()
            .unwrap(),
        sample_json_id
    );
    clean_start
}

async fn run_rollback_instructions(
    clickhouse: &ClickHouseConnectionInfo,
    migration: &(dyn Migration + Send + Sync),
) {
    let rollback_instructions = migration.rollback_instructions();
    // The ClickHouse HTTP interface doesn't support sending multiple statements at once
    for line in rollback_instructions.lines() {
        if line.trim().is_empty() {
            continue;
        }
        println!("Running rollback instruction: {line}");
        clickhouse
            .run_query_synchronous_no_params(line.to_string())
            .await
            .unwrap();
    }
}
async fn test_rollback_helper(migration_num: usize) {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let (fresh_clickhouse, _cleanup_fresh_clickhouse) = get_clean_clickhouse(true).await;
    fresh_clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();
    let migrations = make_all_migrations(&fresh_clickhouse);
    println!(
        "Running migrations up to {}",
        migrations[migration_num].name()
    );
    for migration in &migrations[..=migration_num] {
        let name = migration.name();
        println!("Running migration: {name}");
        migration_manager::run_migration(RunMigrationArgs {
            clickhouse: &fresh_clickhouse,
            migration: migration.as_ref(),
            clean_start: false,
            manual_run: false,
            is_replicated: false,
        })
        .await
        .unwrap();
        // Migration0029 only runs if `StaticEvaluationHumanFeedbackFloatView` or `StaticEvaluationHumanFeedbackBooleanView`
        // exists, which were created by the banned migration Migration0023
        let should_succeed = migration.name() != "Migration0029";
        if should_succeed {
            assert!(
                logs_contain(&format!("Migration succeeded: {name}")),
                "Migration {name} should have succeeded"
            );
        }
    }

    run_rollback_instructions(&fresh_clickhouse, &*migrations[migration_num]).await;

    // The rollback for Migration0000 drops the entire database, which will cause '_cleanup_fresh_clickhouse'
    // to try to run commands on a non-existent database.
    // We make sure that the database exists at the end of the rollback to prevent '_cleanup_fresh_clickhouse' from
    // panicking on drop.
    fresh_clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();
}

// Generate tests named 'test_rollback_up_to_migration_index_n' for each migration index `n``
// Each test will run all of the migrations up to and including `n`, and then test that running the rollback
// instructions for `n` succeeds
invoke_all_separate_tests!(
    test_rollback_helper,
    test_rollback_up_to_migration_index_,
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
    ]
);

#[tokio::test(flavor = "multi_thread")]
async fn test_rollback_apply_rollback() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();
    let migrations = make_all_migrations(&clickhouse);
    for migration in migrations {
        let name = migration.name();
        println!("Running migration: {name}");
        migration_manager::run_migration(RunMigrationArgs {
            clickhouse: &clickhouse,
            migration: migration.as_ref(),
            clean_start: false,
            manual_run: false,
            is_replicated: false,
        })
        .await
        .unwrap();

        // Migration0029 only runs if `StaticEvaluationHumanFeedbackFloatView` or `StaticEvaluationHumanFeedbackBooleanView`
        // exists, which were created by the banned migration Migration0023
        let should_succeed = migration.name() != "Migration0029";
        if should_succeed {
            assert!(
                logs_contain(&format!("Migration succeeded: {name}")),
                "Migration {name} should have succeeded"
            );
        }
        println!("Running rollback instructions for {name}");
        run_rollback_instructions(&clickhouse, migration.as_ref()).await;

        // This migration drops the entire database during rollback, so we need to re-create it
        if migration.name() == "Migration0000" {
            sleep(Duration::from_millis(500)).await;
            clickhouse
                .create_database_and_migrations_table()
                .await
                .unwrap();
        }

        println!("Re-apply migration: {name}");
        migration_manager::run_migration(RunMigrationArgs {
            clickhouse: &clickhouse,
            migration: migration.as_ref(),
            clean_start: false,
            manual_run: false,
            is_replicated: false,
        })
        .await
        .unwrap();
        if should_succeed {
            assert!(
                logs_contain(&format!("Migration succeeded: {name}")),
                "Migration {name} should have succeeded"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clickhouse_migration_manager() {
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();
    // Run it twice to test that it is a no-op the second time
    clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();
    let migrations = make_all_migrations(&clickhouse);
    let initial_clean_start = Cell::new(true);
    let manual_run = clickhouse.is_cluster_configured();
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    // This runs all migrations up to and including the given migration number,
    // verifying that only the most recent migration is actually applied.
    let run_migrations_up_to = |migration_num: usize| {
        let migrations = &migrations;
        let clickhouse = &clickhouse;
        let initial_clean_start = &initial_clean_start;
        let logs_contain = &logs_contain;
        async move {
            // All of the previous migrations should have already been run
            for (i, migration) in migrations.iter().enumerate().take(migration_num) {
                let clean_start = migration_manager::run_migration(RunMigrationArgs {
                    clickhouse,
                    migration: migration.as_ref(),
                    clean_start: false,
                    manual_run,
                    is_replicated: false,
                })
                .await
                .unwrap();
                if i == 0 {
                    // We know that the first migration was run in a previous test, so clean start should be false
                    assert!(!clean_start);
                }
                let name = migrations[i].name();
                // Migration0029 always runs (since we want it to write a migration row on a clean start)
                if migration.name() != "Migration0029" {
                    assert!(
                        !logs_contain(&format!("Applying migration: {name}")),
                        "Migration {name} should not have been applied"
                    );
                    assert!(
                        !logs_contain(&format!("Migration succeeded: {name}")),
                        "Migration {name} should not have succeeded (because it wasn't applied)"
                    );
                }
            }
            assert!(!logs_contain("Materialized view `CumulativeUsageView` was not written because it was recently created"),
                "CumulativeUsage backfilling failed.");

            let run_migration = || async {
                migration_manager::run_migration(RunMigrationArgs {
                    clickhouse,
                    migration: migrations[migration_num].as_ref(),
                    clean_start: initial_clean_start.get(),
                    is_replicated: false,
                    manual_run,
                })
                .await
                .unwrap()
            };

            // The latest migration should get applied, since we haven't run it before
            let name = migrations[migration_num].name();

            let clean_start = match name.as_str() {
                "Migration0009" => {
                    // Insert our large fixtures - these will be used for Migration0009 all subsequent migrations
                    insert_large_fixtures(clickhouse).await;
                    // We're going to be inserting data into the tables, so run all subsequent
                    // migrations in non-clean-start mode.
                    initial_clean_start.set(false);
                    run_migration_0009_with_data(clickhouse, run_migration).await
                }
                "Migration0020" => {
                    assert!(
                        !initial_clean_start.get(),
                        "Migration0020 should not be run on a clean start"
                    );
                    run_migration_0020_with_data(clickhouse, run_migration).await
                }
                "Migration0021" => {
                    assert!(
                        !initial_clean_start.get(),
                        "Migration0021 should not be run on a clean start"
                    );
                    run_migration_0021_with_data(clickhouse, run_migration).await
                }
                _ => run_migration().await,
            };

            if migration_num == 0 {
                // When running for the first time, we should have a clean start.
                assert!(clean_start);
            }
            // Migration 0029 will not be applied since 0023 is turned off.
            if name != "Migration0029" {
                assert!(logs_contain(&format!("Applying migration: {name}")));
                assert!(logs_contain(&format!("Migration succeeded: {name}")));
            }
            assert!(!logs_contain("Failed to apply migration"));
            assert!(!logs_contain("Failed migration success check"));
            assert!(!logs_contain("Failed to verify migration"));
            assert!(!logs_contain("ERROR"));
        }
    };
    async fn run_all(
        clickhouse: &ClickHouseConnectionInfo,
        migrations: &[Box<dyn Migration + Send + Sync + '_>],
        logs_contain: &impl Fn(&str) -> bool,
    ) {
        // Now, run all of the migrations, and verify that none of them apply
        for (i, migration) in migrations.iter().enumerate() {
            let clean_start = migration_manager::run_migration(RunMigrationArgs {
                clickhouse,
                migration: migration.as_ref(),
                clean_start: true,
                is_replicated: false,
                manual_run: true,
            })
            .await
            .unwrap();
            if i == 0 {
                // We know that the first migration was run in a previous test, so clean start should be false
                assert!(!clean_start);
            }
            let name = migrations[i].name();
            // Migration0029 always runs
            if name != "Migration0029" {
                assert!(
                    !logs_contain(&format!("Applying migration: {name}")),
                    "Missing log for {name}"
                );
                assert!(
                    !logs_contain(&format!("Migration succeeded: {name}")),
                    "Missing success for {name}"
                );
            }
        }

        assert!(!logs_contain("Failed to apply migration"));
        assert!(!logs_contain("Failed migration success check"));
        assert!(!logs_contain("Failed to verify migration"));
        assert!(!logs_contain("ERROR"));
    }

    for i in 0..migrations.len() {
        reset_capture_logs();
        run_migrations_up_to(i).await;
    }
    reset_capture_logs();

    let rows = get_all_migration_records(&clickhouse).await.unwrap();
    println!("Rows: {rows:#?}");
    let expected_migrations = &migrations;

    // Check that we wrote out migration records for all migrations
    assert_eq!(rows.len(), expected_migrations.len());
    for (migration_record, migration) in rows.iter().zip(expected_migrations.iter()) {
        let MigrationRecordDatabaseInsert {
            migration_id,
            migration_name,
            gateway_version,
            gateway_git_sha,
            execution_time_ms: _,
            applied_at,
        } = migration_record;
        assert_eq!(*migration_id, migration.migration_num().unwrap());
        assert_eq!(*migration_name, migration.name());
        assert_eq!(gateway_version, TENSORZERO_VERSION);
        assert_eq!(
            gateway_git_sha,
            tensorzero_core::built_info::GIT_COMMIT_HASH.unwrap_or("unknown")
        );
        assert!(applied_at.is_some());
    }
    run_all(&clickhouse, &migrations, &logs_contain).await;
    let mut new_rows = get_all_migration_records(&clickhouse).await.unwrap();

    let response = clickhouse
        .run_query_synchronous_no_params(
            "SELECT count FROM CumulativeUsage FINAL WHERE type='input_tokens'".to_string(),
        )
        .await
        .unwrap();
    let input_token_total: u64 = response.response.trim().parse().unwrap();
    assert_eq!(input_token_total, 200000000);
    let response = clickhouse
        .run_query_synchronous_no_params(
            "SELECT count FROM CumulativeUsage FINAL WHERE type='output_tokens'".to_string(),
        )
        .await
        .unwrap();
    let output_token_total: u64 = response.response.trim().parse().unwrap();
    assert_eq!(output_token_total, 200000000);
    // Let's add a ModelInference row with null output tokens only then check the input tokens are correct
    let row = ModelInferenceDatabaseInsert {
        id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        raw_request: String::new(),
        raw_response: String::new(),
        system: None,
        input_messages: String::new(),
        output: String::new(),
        input_tokens: Some(123),
        output_tokens: None,
        response_time_ms: None,
        model_name: String::new(),
        model_provider_name: String::new(),
        ttft_ms: None,
        cached: false,
        finish_reason: None,
    };
    clickhouse
        .write_non_batched(Rows::Unserialized(&[row]), TableName::ModelInference)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let response = clickhouse
        .run_query_synchronous_no_params(
            "SELECT count FROM CumulativeUsage FINAL WHERE type='input_tokens'".to_string(),
        )
        .await
        .unwrap();
    let input_token_total: u64 = response.response.trim().parse().unwrap();
    assert_eq!(input_token_total, 200000123);
    let response = clickhouse
        .run_query_synchronous_no_params(
            "SELECT count FROM CumulativeUsage FINAL WHERE type='output_tokens'".to_string(),
        )
        .await
        .unwrap();
    let output_token_total: u64 = response.response.trim().parse().unwrap();
    assert_eq!(output_token_total, 200000000);

    // Check that the EpisodeById migration worked
    let response = clickhouse
        .run_query_synchronous_no_params("SELECT count() FROM EpisodeById".to_string())
        .await
        .unwrap();
    let episode_count: u64 = response.response.trim().parse().unwrap();
    assert_eq!(episode_count, 20000000);

    // Check that the FeedbackByVariantStatistics migration worked
    let response = clickhouse
        .get_feedback_by_variant("exact_match", "dummy_function", None)
        .await
        .unwrap();
    assert_eq!(response.len(), 1);
    let feedback_by_variant = response.first().unwrap();
    assert_eq!(feedback_by_variant.variant_name, "dummy");
    assert_eq!(feedback_by_variant.count, 2500000);
    assert_eq!(feedback_by_variant.mean, 1.0);
    assert_eq!(feedback_by_variant.variance, Some(0.0));

    // Since we've already ran all of the migrations, we shouldn't have written any new records
    // except for Migration0029 (which runs every time)

    let mut rows = rows;
    let _new_migration0029 = new_rows
        .iter()
        .find(|row| row.migration_name == "Migration0029")
        .unwrap();
    new_rows.retain(|row| row.migration_name != "Migration0029");

    let _old_migration0029 = rows
        .iter()
        .find(|row| row.migration_name == "Migration0029")
        .unwrap();
    rows.retain(|row| row.migration_name != "Migration0029");

    assert_eq!(new_rows, rows);
}

#[tokio::test]
async fn test_bad_clickhouse_write() {
    let clickhouse = get_clickhouse().await;
    // "name" should be "metric_name" here but we are using the wrong field on purpose to check that the write fails
    let payload =
        json!({"target_id": Uuid::now_v7(), "value": true, "name": "test", "id": Uuid::now_v7()});
    let err = clickhouse
        .write_batched(&[payload], TableName::BooleanMetricFeedback)
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Unknown field found while parsing JSONEachRow format: name"),
        "Unexpected error: {err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clean_clickhouse_start() {
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    let database = clickhouse.database();
    let is_manual = clickhouse.is_cluster_configured();
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: is_manual,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    // We also verify here that all tables are either replicated or not replicated as expected
    let response = clickhouse
        .run_query_synchronous_no_params("SHOW TABLES".to_string())
        .await
        .unwrap();
    let tables = response.response.split('\n');
    for table in tables {
        let table = table.trim();
        if table.is_empty() {
            continue;
        }
        let create_table_info = clickhouse
            .run_query_synchronous_no_params(format!("SHOW CREATE TABLE {table}"))
            .await
            .unwrap()
            .response;
        println!("create_table_info: {create_table_info}");
        // We only need to worry about MergeTree tables when checking replication
        if !create_table_info.contains("MergeTree") {
            continue;
        }
        let engine_is_replicated = create_table_info.contains("Replicated");
        let replica_info = clickhouse.run_query_synchronous_no_params(format!(
            "SELECT total_replicas FROM system.replicas WHERE database = '{database}' AND table = '{table}'"
        )).await.unwrap();
        if clickhouse.is_cluster_configured() {
            assert!(
                engine_is_replicated,
                "Table {table} is not replicated but ClickHouse is configured for replication."
            );
            let replica_count: u8 = replica_info.response.trim().parse().unwrap();
            assert_eq!(replica_count, 2);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_startup_without_migration_table() {
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    let is_manual = clickhouse.is_cluster_configured();
    // Run the migrations so we can get the database into a "dirty" state
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: is_manual,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    // Drop the TensorZeroMigration table
    clickhouse
        .run_query_synchronous_no_params(format!(
            "DROP TABLE TensorZeroMigration {} SYNC",
            clickhouse.get_on_cluster_name()
        ))
        .await
        .unwrap();

    // Run the migrations again to ensure that they don't panic and that the table is recreated
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: is_manual,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    // Make sure the table exists
    assert!(
        check_table_exists(&clickhouse, "TensorZeroMigration", "TEST")
            .await
            .unwrap()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_deployment_id_oldest() {
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();
    // Add a row to the DeploymentID table and make sure that it isn't returned
    let new_deployment_id = "foo";
    clickhouse
        .write_non_batched(
            Rows::Unserialized(&[serde_json::json!({
                "deployment_id": new_deployment_id,
            })]),
            TableName::DeploymentID,
        )
        .await
        .unwrap();
    // Run a query that gets the newest deployment ID but since it's final it shouldn't be foo
    let deployment_id = clickhouse
        .run_query_synchronous_no_params(
            "SELECT deployment_id FROM DeploymentID FINAL ORDER BY created_at DESC LIMIT 1"
                .to_string(),
        )
        .await
        .unwrap()
        .response;

    assert_ne!(deployment_id, new_deployment_id);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_clickhouse_migrations() {
    if std::env::var("TENSORZERO_CLICKHOUSE_CLUSTER_NAME").is_ok() {
        // We can't run concurrent migrations on a cluster.
        return;
    }
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    let clickhouse = Arc::new(clickhouse);
    let num_concurrent_starts = 50;
    let mut handles = Vec::with_capacity(num_concurrent_starts);
    for _ in 0..num_concurrent_starts {
        let clickhouse_clone = clickhouse.clone();
        // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
        #[expect(clippy::disallowed_methods)]
        handles.push(tokio::spawn(async move {
            migration_manager::run(RunMigrationManagerArgs {
                clickhouse: &clickhouse_clone,
                is_manual_run: false,
                disable_automatic_migrations: false,
            })
            .await
            .unwrap();
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }

    // We should have written at least one duplicate migration record to `TensorZeroMigration`
    // due to multiple copies of the same migration running concurrently.
    let total_runs = clickhouse
        .run_query_synchronous_no_params("SELECT COUNT(*) FROM TensorZeroMigration".to_string())
        .await
        .unwrap()
        .response
        .trim()
        .parse::<u64>()
        .unwrap();

    let all_migrations = migration_manager::get_all_migration_records(&clickhouse)
        .await
        .unwrap();
    assert!(
        total_runs as usize > all_migrations.len(),
        "Expected more than {} migration runs, but only found {total_runs}",
        all_migrations.len()
    );

    let migrations = migration_manager::make_all_migrations(&clickhouse);
    assert_eq!(
        migration_manager::check_migrations_state(&clickhouse, &migrations).await,
        Ok(MigrationTableState::JustRight)
    );
}

/// Migration 0013 has some checks that enforce that concurrent migrations can't break
/// the database.
/// This test enforces that the migration will error if there would be an invalid database state
/// rather than brick the database.
#[tokio::test(flavor = "multi_thread")]
async fn test_migration_0013_old_table() {
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();

    // When creating a new migration, add it to the end of this array,
    // and adjust the call to `invoke_all!` to include the new array index.
    let migrations: [Box<dyn Migration + '_>; 9] = [
        Box::new(Migration0000 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0002 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0003 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0004 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0005 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0006 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0008 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0009 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0011 {
            clickhouse: &clickhouse,
        }),
    ];

    // Run migrations up to right before 0013
    for migration in migrations {
        migration_manager::run_migration(RunMigrationArgs {
            clickhouse: &clickhouse,
            migration: migration.as_ref(),
            clean_start: true,
            manual_run: false,
            is_replicated: false,
        })
        .await
        .unwrap();
    }
    // Manually create a table that should not exist
    let query = r"
        CREATE TABLE IF NOT EXISTS InferenceById
        (
            id UUID, -- must be a UUIDv7
            function_name LowCardinality(String),
            variant_name LowCardinality(String),
            episode_id UUID, -- must be a UUIDv7,
            function_type Enum('chat' = 1, 'json' = 2)
        ) ENGINE = MergeTree()
        ORDER BY id;
    ";
    let _ = clickhouse
        .run_query_synchronous_no_params(query.to_string())
        .await
        .unwrap();
    let err = migration_manager::run_migration(RunMigrationArgs {
        clickhouse: &clickhouse,
        migration: &Migration0013 {
            clickhouse: &clickhouse,
        },
        clean_start: false,
        manual_run: false,
        is_replicated: false,
    })
    .await
    .unwrap_err();
    assert!(
        err.to_string().contains("InferenceById table is in an invalid state. Please contact TensorZero team.") ||
        err.to_string().contains("SELECT query outputs column with name 'id_uint', which is not found in the target table."),
        "Unexpected error: {err}",
    );
}

/// For this test, we will run all the migrations up to 0011, add some data to
/// the JSONInference table, then run migration 0013.
/// This should fail.
#[tokio::test(flavor = "multi_thread")]
async fn test_migration_0013_data_no_table() {
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();

    // When creating a new migration, add it to the end of this array,
    // and adjust the call to `invoke_all!` to include the new array index.
    let migrations: [Box<dyn Migration + '_>; 9] = [
        Box::new(Migration0000 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0002 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0003 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0004 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0005 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0006 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0008 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0009 {
            clickhouse: &clickhouse,
        }),
        Box::new(Migration0011 {
            clickhouse: &clickhouse,
        }),
    ];

    // Run migrations up to right before 0013
    for migration in migrations {
        migration_manager::run_migration(RunMigrationArgs {
            clickhouse: &clickhouse,
            migration: migration.as_ref(),
            clean_start: true,
            manual_run: false,
            is_replicated: false,
        })
        .await
        .unwrap();
    }

    // Add a row to the JsonInference table (would be very odd to have data in this table
    // but not an InferenceById table).
    let query = r"
        INSERT INTO JsonInference (id, function_name, variant_name, episode_id, input, output, output_schema, inference_params, processing_time_ms)
        VALUES (generateUUIDv7(), 'test_function', 'test_variant', generateUUIDv7(), 'input', 'output', 'output_schema', 'params', 100)
    ";
    let _ = clickhouse
        .run_query_synchronous_no_params(query.to_string())
        .await
        .unwrap();
    let err = migration_manager::run_migration(RunMigrationArgs {
        clickhouse: &clickhouse,
        migration: &Migration0013 {
            clickhouse: &clickhouse,
        },
        clean_start: false,
        manual_run: false,
        is_replicated: false,
    })
    .await
    .unwrap_err();
    assert!(err.to_string()
        .contains("Data already exists in the ChatInference or JsonInference tables and InferenceById or InferenceByEpisodeId is missing. Please contact TensorZero team"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_migrations_clean() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: true,
    })
    .await
    .unwrap();
    assert!(logs_contain("Database not found, assuming clean start"));
    assert!(!logs_contain("All migrations have already been applied"));

    // Run again, and we should skip all migrations
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: true,
    })
    .await
    .unwrap();
    assert!(logs_contain("All migrations have already been applied"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_migrations_fake_row() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let (clickhouse, _cleanup_db) = get_clean_clickhouse(false).await;
    clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();

    struct Migration99999;

    #[async_trait]
    impl Migration for Migration99999 {
        fn name(&self) -> String {
            "Migration99999".to_string()
        }
        async fn can_apply(&self) -> Result<(), Error> {
            Ok(())
        }
        async fn should_apply(&self) -> Result<bool, Error> {
            Ok(true)
        }
        async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
            Ok(())
        }
        async fn has_succeeded(&self) -> Result<bool, Error> {
            Ok(true)
        }
        fn rollback_instructions(&self) -> String {
            "SELECT 1;".to_string()
        }
    }

    let migration_manager_result = migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: false,
        disable_automatic_migrations: false,
    })
    .await;
    assert!(
        !(migration_manager_result.is_err() ^ clickhouse.is_cluster_configured()),
        "Migration manager should fail to run if and only if a cluster is configured"
    );

    // Run our fake migration to insert an unexpected row into `TensorZeroMigration`
    // A subsequent normal run of migrations should *not* skip running migrations,
    // since it will see an unexpected state
    migration_manager::run_migration(RunMigrationArgs {
        clickhouse: &clickhouse,
        migration: &Migration99999,
        manual_run: false,
        clean_start: true,
        is_replicated: false,
    })
    .await
    .unwrap();

    let migrations = migration_manager::make_all_migrations(&clickhouse);
    // If the ClickHouse cluster is configured, migrations aren't applied, so the only migration
    // in the table is Migration99999.
    if clickhouse.is_cluster_configured() {
        assert_eq!(
            migration_manager::check_migrations_state(&clickhouse, &migrations).await,
            Ok(MigrationTableState::Inconsistent)
        );
    // If the ClickHouse cluster isn't configured, migrations are applied, so the migrations table
    // has all the required migrations plus Migration99999.
    } else {
        assert_eq!(
            migration_manager::check_migrations_state(&clickhouse, &migrations).await,
            Ok(MigrationTableState::TooMany)
        );
    }

    let migration_manager_result = migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: false,
        disable_automatic_migrations: false,
    })
    .await;
    assert!(
        !(migration_manager_result.is_err() ^ clickhouse.is_cluster_configured()),
        "Migration manager should fail to run if and only if a cluster is configured"
    );
    assert!(!logs_contain("already been applied"));

    let rows = migration_manager::get_all_migration_records(&clickhouse)
        .await
        .unwrap();

    let mut actual_migration_ids = rows.iter().map(|r| r.migration_id).collect::<Vec<_>>();
    actual_migration_ids.sort();

    let all_migrations = migration_manager::make_all_migrations(&clickhouse);
    let mut expected_migration_ids = all_migrations
        .iter()
        .map(|m| m.migration_num().unwrap())
        .collect::<Vec<_>>();
    expected_migration_ids.push(99999);
    expected_migration_ids.sort();

    if clickhouse.is_cluster_configured() {
        assert_eq!(actual_migration_ids, vec![99999]);
    } else {
        assert_eq!(actual_migration_ids, expected_migration_ids);
    }
}

#[tokio::test(flavor = "multi_thread")]
/// Test the is_manual_run and disable_automatic_migrations flags
async fn test_migration_logic_with_flags() {
    if std::env::var("TENSORZERO_CLICKHOUSE_CLUSTER_NAME").is_ok() {
        // We disable this test for replicated clickhouse because migrations have always been manual
        // in that case.
        return;
    }
    // Test case 1: is_manual_run = false, disable_automatic_migrations = false
    let (clickhouse, _cleanup) = get_clean_clickhouse(true).await;
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: false,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    // Test case 2: false, true
    let (clickhouse, _cleanup) = get_clean_clickhouse(true).await;
    let err = migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: false,
        disable_automatic_migrations: true,
    })
    .await
    .unwrap_err();
    assert_eq!(err, Error::new(ErrorDetails::ClickHouseMigrationsDisabled));
    // Create database to avoid panicking when the database is dropped via DeleteDbOnDrop.drop(),
    // because it won't be created otherwise when migration_manager::run() throws an error.
    clickhouse
        .create_database_and_migrations_table()
        .await
        .unwrap();

    // Test case 3: true, false
    let (clickhouse, _cleanup) = get_clean_clickhouse(true).await;
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: false,
    })
    .await
    .unwrap();

    // Test case 4: true, true
    let (clickhouse, _cleanup) = get_clean_clickhouse(true).await;
    migration_manager::run(RunMigrationManagerArgs {
        clickhouse: &clickhouse,
        is_manual_run: true,
        disable_automatic_migrations: true,
    })
    .await
    .unwrap();
}
