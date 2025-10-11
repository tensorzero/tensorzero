#![expect(clippy::expect_used, clippy::unwrap_used, clippy::missing_panics_doc)]

use crate::{Client, ClientBuilder, ClientBuilderMode};
use lazy_static::lazy_static;
use tempfile::NamedTempFile;
use tensorzero_core::config::BatchWritesConfig;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tokio::runtime::Handle;
use tokio::sync::Mutex as AsyncMutex;
use url::Url;
use uuid::Uuid;

lazy_static! {
    static ref CLEAN_CLICKHOUSE_LOCK: AsyncMutex<()> = AsyncMutex::new(());
}

/// Guard that drops the temporary ClickHouse database created for an embedded gateway test.
pub struct ClickHouseTestDatabaseGuard {
    database: String,
    client: ClickHouseConnectionInfo,
}

impl Drop for ClickHouseTestDatabaseGuard {
    fn drop(&mut self) {
        let database = self.database.clone();
        let client = self.client.clone();

        // Run the DROP DATABASE command synchronously to avoid leaving junk behind in local tests.
        tokio::task::block_in_place(|| {
            let drop_future = async {
                let query = format!("DROP DATABASE IF EXISTS {database} SYNC");
                let _ = client.run_query_synchronous_no_params(query).await;
            };

            match Handle::try_current() {
                Ok(handle) => handle.block_on(drop_future),
                Err(_) => {
                    if let Ok(runtime) = tokio::runtime::Runtime::new() {
                        runtime.block_on(drop_future);
                    }
                }
            }
        });
    }
}

async fn create_clean_clickhouse() -> (ClickHouseConnectionInfo, ClickHouseTestDatabaseGuard) {
    let database = format!("tensorzero_e2e_tests_{}", Uuid::now_v7().simple());

    // Ensure the embedded gateway uses the fresh database name.
    std::env::set_var("TENSORZERO_E2E_TESTS_DATABASE", &database);

    let mut clickhouse_url = Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path("");
    clickhouse_url.set_query(Some(format!("database={database}").as_str()));

    let clickhouse =
        ClickHouseConnectionInfo::new(clickhouse_url.as_ref(), BatchWritesConfig::default())
            .await
            .expect("Failed to connect to ClickHouse");

    (
        clickhouse.clone(),
        ClickHouseTestDatabaseGuard {
            database,
            client: clickhouse,
        },
    )
}

pub async fn make_http_gateway() -> Client {
    ClientBuilder::new(ClientBuilderMode::HTTPGateway {
        url: Url::parse(&get_gateway_endpoint("/")).unwrap(),
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway() -> Client {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../../tensorzero-core/tests/e2e/tensorzero.toml");
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway_no_config() -> Client {
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: None,
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway_with_config(config: &str) -> Client {
    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

pub async fn make_embedded_gateway_with_config_and_postgres(config: &str) -> Client {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for rate limiting tests");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: Some(postgres_url),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

/// Build an embedded gateway backed by a fresh ClickHouse database.
///
/// Returns the client, the ClickHouse connection, and a guard that drops the database once it is
/// no longer needed.
pub async fn make_embedded_gateway_with_clean_clickhouse(
    config: &str,
) -> (
    Client,
    ClickHouseConnectionInfo,
    ClickHouseTestDatabaseGuard,
) {
    let _lock = CLEAN_CLICKHOUSE_LOCK.lock().await;

    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for tests that require Postgres");

    let (clickhouse, guard) = create_clean_clickhouse().await;
    clickhouse
        .create_database_and_migrations_table()
        .await
        .expect("failed to create ClickHouse database for embedded gateway tests");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    // Point the embedded gateway at the freshly-created database.
    let database = clickhouse.database();
    let mut clickhouse_url = Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path("");
    clickhouse_url.set_query(Some(format!("database={database}").as_str()));
    let clickhouse_url_string = clickhouse_url.to_string();

    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url_string),
        postgres_url: Some(postgres_url),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    (client, clickhouse, guard)
}

/// Build an embedded gateway backed by an existing ClickHouse database.
///
/// This is useful for testing cold-start behavior where a new gateway needs to read
/// existing data from the database.
///
/// Returns the client, the ClickHouse connection, and a guard that drops the database once it is
/// no longer needed.
pub async fn make_embedded_gateway_with_existing_clickhouse(
    config: &str,
    existing_clickhouse: &ClickHouseConnectionInfo,
) -> (
    Client,
    ClickHouseConnectionInfo,
    ClickHouseTestDatabaseGuard,
) {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for tests that require Postgres");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    // Point the embedded gateway at the existing database.
    let database = existing_clickhouse.database();
    let mut clickhouse_url = Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path("");
    clickhouse_url.set_query(Some(format!("database={database}").as_str()));
    let clickhouse_url_string = clickhouse_url.to_string();

    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url_string),
        postgres_url: Some(postgres_url),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    // Create a guard for the existing database
    let guard = ClickHouseTestDatabaseGuard {
        database: database.to_string(),
        client: existing_clickhouse.clone(),
    };

    (client, existing_clickhouse.clone(), guard)
}

// We use a multi-threaded runtime so that the embedded gateway can use 'block_on'.
// For consistency, we also use a multi-threaded runtime for the http gateway test.

#[macro_export]
macro_rules! make_gateway_test_functions {
    ($prefix:ident) => {
        paste::paste! {

            #[tokio::test(flavor = "multi_thread")]
            async fn [<$prefix _embedded_gateway>]() {
                $prefix (tensorzero::test_helpers::make_embedded_gateway().await).await;
            }


            #[tokio::test(flavor = "multi_thread")]
            async fn [<$prefix _http_gateway>]() {
                $prefix (tensorzero::test_helpers::make_http_gateway().await).await;
            }
        }
    };
}

fn get_gateway_endpoint(path: &str) -> String {
    let gateway_host =
        std::env::var("TENSORZERO_GATEWAY_HOST").unwrap_or_else(|_| "localhost".to_string());
    let gateway_port =
        std::env::var("TENSORZERO_GATEWAY_PORT").unwrap_or_else(|_| "3000".to_string());
    format!("http://{gateway_host}:{gateway_port}{path}")
}
