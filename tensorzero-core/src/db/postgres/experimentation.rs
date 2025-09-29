use crate::{
    db::{postgres::PostgresConnectionInfo, ExperimentationQueries},
    error::{Error, ErrorDetails},
};

struct CASVariantResponse {
    variant_name: Option<String>,
}

impl ExperimentationQueries for PostgresConnectionInfo {
    async fn compare_and_swap_variant_by_episode(
        &self,
        episode_id: uuid::Uuid,
        function_name: &str,
        candidate_variant_name: &str,
    ) -> Result<String, crate::error::Error> {
        let pool = self.get_pool_result()?;

        let response = sqlx::query_as!(
            CASVariantResponse,
            r"WITH inserted_row AS (
                INSERT INTO variant_by_episode(function_name, episode_id, variant_name)
                VALUES ($1, $2, $3)
                ON CONFLICT (function_name, episode_id) DO NOTHING
                RETURNING variant_name
            )
            SELECT variant_name FROM inserted_row
            UNION ALL
            SELECT variant_name FROM variant_by_episode
            WHERE
                function_name = $1 AND episode_id = $2
                AND NOT EXISTS (SELECT 1 FROM inserted_row);",
            &function_name,
            &episode_id,
            &candidate_variant_name
        )
        .fetch_one(pool)
        .await?;
        let variant_name = response.variant_name.ok_or_else(|| {
            Error::new(ErrorDetails::PostgresResult {
                result_type: "compare_and_swap_variant_by_episode",
                message: "Missing variant name".to_string(),
            })
        })?;

        Ok(variant_name)
    }
}
