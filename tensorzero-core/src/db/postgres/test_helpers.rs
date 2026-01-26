#![expect(clippy::expect_used, clippy::missing_panics_doc, clippy::print_stdout)]
use std::sync::LazyLock;

use sqlx::postgres::PgPoolOptions;

use super::PostgresConnectionInfo;

pub static POSTGRES_URL: LazyLock<String> = LazyLock::new(|| {
    std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("Environment variable TENSORZERO_POSTGRES_URL must be set")
});

pub async fn get_postgres() -> PostgresConnectionInfo {
    let postgres_url = &*POSTGRES_URL;
    let start = std::time::Instant::now();
    println!("Connecting to PostgreSQL");
    let pool = PgPoolOptions::new()
        .connect(postgres_url)
        .await
        .expect("Failed to connect to PostgreSQL");
    println!("Connected to PostgreSQL in {:?}", start.elapsed());
    PostgresConnectionInfo::new_with_pool(pool)
}
