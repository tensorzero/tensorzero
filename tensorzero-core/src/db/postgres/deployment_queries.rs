use sqlx::Row;

use crate::db::postgres::PostgresConnectionInfo;

impl PostgresConnectionInfo {
    /// Gets or creates a deployment ID in Postgres.
    ///
    /// This is analogous to the ClickHouse `DeploymentID` table. The deployment ID
    /// is a blake3 hash of a UUIDv7, generated once and stored as a singleton row.
    /// Race conditions are handled via `ON CONFLICT DO NOTHING`.
    pub async fn get_or_create_deployment_id(&self) -> Result<String, ()> {
        let pool = self.get_pool().ok_or(())?;

        // Try to read existing deployment ID
        let row = sqlx::query("SELECT deployment_id FROM tensorzero.deployment_id WHERE dummy = 0")
            .fetch_optional(pool)
            .await
            .map_err(|e| {
                tracing::debug!("Failed to query deployment_id from Postgres: {e}");
            })?;

        if let Some(row) = row {
            let deployment_id: String = row.try_get("deployment_id").map_err(|e| {
                tracing::debug!("Failed to get deployment_id column: {e}");
            })?;
            return Ok(deployment_id);
        }

        // Generate a new deployment ID (same logic as ClickHouse migration_0033)
        let deployment_id = generate_deployment_id();

        // Insert with ON CONFLICT DO NOTHING to handle races
        sqlx::query(
            "INSERT INTO tensorzero.deployment_id (deployment_id) VALUES ($1) ON CONFLICT (dummy) DO NOTHING",
        )
        .bind(&deployment_id)
        .execute(pool)
        .await
        .map_err(|e| {
            tracing::debug!("Failed to insert deployment_id into Postgres: {e}");
        })?;

        // Re-read to get the winner in case of a race
        let row = sqlx::query("SELECT deployment_id FROM tensorzero.deployment_id WHERE dummy = 0")
            .fetch_one(pool)
            .await
            .map_err(|e| {
                tracing::debug!("Failed to re-read deployment_id from Postgres: {e}");
            })?;

        let deployment_id: String = row.try_get("deployment_id").map_err(|e| {
            tracing::debug!("Failed to get deployment_id column: {e}");
        })?;

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
