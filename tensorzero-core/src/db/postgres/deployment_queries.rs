use async_trait::async_trait;
use sqlx::Row;

use crate::db::DeploymentIdQueries;
use crate::db::postgres::PostgresConnectionInfo;
use crate::error::Error;

#[async_trait]
impl DeploymentIdQueries for PostgresConnectionInfo {
    /// Gets or creates a deployment ID in Postgres.
    async fn get_deployment_id(&self) -> Result<String, Error> {
        self.get_or_create_deployment_id().await
    }
}

impl PostgresConnectionInfo {
    /// Gets or creates a deployment ID in Postgres.
    ///
    /// This is analogous to the ClickHouse `DeploymentID` table. The deployment ID
    /// is a blake3 hash of a UUIDv7, generated once and stored as a singleton row.
    /// Race conditions are handled via `ON CONFLICT DO NOTHING`.
    async fn get_or_create_deployment_id(&self) -> Result<String, Error> {
        let pool = self.get_pool_result()?;

        // Try to read existing deployment ID
        let row = sqlx::query("SELECT deployment_id FROM tensorzero.deployment_id LIMIT 1")
            .fetch_optional(pool)
            .await?;

        if let Some(row) = row {
            let deployment_id: String = row.try_get("deployment_id")?;
            return Ok(deployment_id);
        }

        // Generate a new deployment ID (same logic as ClickHouse migration_0033)
        let deployment_id = generate_deployment_id();
        self.insert_deployment_id(&deployment_id).await
    }

    /// Inserts a pre-computed deployment ID into Postgres.
    ///
    /// This supports the case where an existing ClickHouse deployment enables Postgres
    /// and needs to keep the deployment ID consistent across both databases.
    /// Uses `ON CONFLICT DO NOTHING` to handle races, then re-reads the winner.
    pub async fn insert_deployment_id(&self, deployment_id: &str) -> Result<String, Error> {
        let pool = self.get_pool_result()?;

        sqlx::query(
            "INSERT INTO tensorzero.deployment_id (deployment_id) VALUES ($1) ON CONFLICT (dummy) DO NOTHING",
        )
        .bind(deployment_id)
        .execute(pool)
        .await?;

        // Re-read to get the winner in case of a race
        let row = sqlx::query("SELECT deployment_id FROM tensorzero.deployment_id LIMIT 1")
            .fetch_one(pool)
            .await?;

        let deployment_id: String = row.try_get("deployment_id")?;
        Ok(deployment_id)
    }
}

/// Generates a deployment ID as the blake3 hash of a UUIDv7.
/// Returns a 64-character hex string.
fn generate_deployment_id() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(uuid::Uuid::now_v7().as_bytes());
    hasher.finalize().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_id_format() {
        let deployment_id = generate_deployment_id();
        assert_eq!(
            deployment_id.len(),
            64,
            "Deployment ID should be 64 hex characters"
        );
        assert!(
            deployment_id.chars().all(|c| c.is_ascii_hexdigit()),
            "Deployment ID should contain only hex characters"
        );
    }
}
