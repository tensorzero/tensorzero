use sqlx::PgPool;

#[derive(Debug, Clone)]
pub enum PostgresConnectionInfo {
    Enabled { pool: PgPool },
    Disabled,
}

impl PostgresConnectionInfo {
    pub fn new_with_pool(pool: PgPool) -> Self {
        Self::Enabled { pool }
    }

    pub fn new_disabled() -> Self {
        Self::Disabled
    }
}
