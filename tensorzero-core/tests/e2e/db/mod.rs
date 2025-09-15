use sqlx::postgres::PgPoolOptions;
use tensorzero_core::db::postgres::PostgresConnectionInfo;

mod rate_limit_queries;
mod select_queries;

pub async fn get_test_postgres() -> PostgresConnectionInfo {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL environment variable not set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&postgres_url)
        .await
        .expect("Failed to connect to PostgreSQL");
    let connection_info = PostgresConnectionInfo::new_with_pool(pool);
    connection_info
        .check_migrations()
        .await
        .expect("PostgreSQL migration check failed");
    connection_info
}
