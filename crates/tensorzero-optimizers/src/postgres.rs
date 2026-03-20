use sqlx::migrate;

pub fn make_migrator() -> sqlx::migrate::Migrator {
    migrate!("src/postgres/migrations")
}
