use crate::{
    db::{postgres::PostgresConnectionInfo, ExperimentationQueries},
    error::{Error, ErrorDetails},
};

struct CheckAndSetVariantResponse {
    variant_name: Option<String>,
}

impl ExperimentationQueries for PostgresConnectionInfo {
    /// Attempts to atomically set a variant for a function-episode pair
    /// If it has already been set, returns the existing variant name
    async fn check_and_set_variant_by_episode(
        &self,
        episode_id: uuid::Uuid,
        function_name: &str,
        candidate_variant_name: &str,
    ) -> Result<String, Error> {
        let pool = self.get_pool_result()?;

        let response = sqlx::query_as!(
            CheckAndSetVariantResponse,
            r"INSERT INTO variant_by_episode(function_name, episode_id, variant_name)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (function_name, episode_id) DO UPDATE
                    SET variant_name = variant_by_episode.variant_name -- A no-op to enable RETURNING
                    RETURNING variant_name;",
            &function_name,
            &episode_id,
            &candidate_variant_name
        )
        .fetch_one(pool)
        .await?;
        let variant_name = response.variant_name.ok_or_else(|| {
            Error::new(ErrorDetails::PostgresResult {
                result_type: "check_and_set_variant_by_episode",
                message: "Missing variant name".to_string(),
            })
        })?;

        Ok(variant_name)
    }
}
