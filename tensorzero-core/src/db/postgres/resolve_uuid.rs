use async_trait::async_trait;
use sqlx::QueryBuilder;
use uuid::Uuid;

use crate::db::resolve_uuid::{ResolveUuidQueries, ResolvedObject};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::FunctionType;

use super::PostgresConnectionInfo;

#[derive(sqlx::FromRow)]
struct InferenceRow {
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
}

#[derive(sqlx::FromRow)]
struct DatapointRow {
    dataset_name: String,
    function_name: String,
}

#[async_trait]
impl ResolveUuidQueries for PostgresConnectionInfo {
    async fn resolve_uuid(&self, id: &Uuid) -> Result<Vec<ResolvedObject>, Error> {
        let pool = self.get_pool_result()?;

        // Run all queries concurrently
        let (
            inference_result,
            episode_result,
            boolean_result,
            float_result,
            comment_result,
            demonstration_result,
            chat_datapoint_result,
            json_datapoint_result,
        ) = tokio::try_join!(
            query_inference(pool, id),
            query_episode(pool, id),
            query_exists(pool, "tensorzero.boolean_metric_feedback", id),
            query_exists(pool, "tensorzero.float_metric_feedback", id),
            query_exists(pool, "tensorzero.comment_feedback", id),
            query_exists(pool, "tensorzero.demonstration_feedback", id),
            query_datapoint(pool, "tensorzero.chat_datapoints", id),
            query_datapoint(pool, "tensorzero.json_datapoints", id),
        )?;

        let mut results = Vec::new();

        if let Some((function_name, function_type, variant_name, episode_id)) = inference_result {
            results.push(ResolvedObject::Inference {
                function_name,
                function_type,
                variant_name,
                episode_id,
            });
        }

        if episode_result {
            results.push(ResolvedObject::Episode);
        }

        if boolean_result {
            results.push(ResolvedObject::BooleanFeedback);
        }

        if float_result {
            results.push(ResolvedObject::FloatFeedback);
        }

        if comment_result {
            results.push(ResolvedObject::CommentFeedback);
        }

        if demonstration_result {
            results.push(ResolvedObject::DemonstrationFeedback);
        }

        if let Some(row) = chat_datapoint_result {
            results.push(ResolvedObject::ChatDatapoint {
                dataset_name: row.dataset_name,
                function_name: row.function_name,
            });
        }

        if let Some(row) = json_datapoint_result {
            results.push(ResolvedObject::JsonDatapoint {
                dataset_name: row.dataset_name,
                function_name: row.function_name,
            });
        }

        Ok(results)
    }
}

/// Query chat_inferences and json_inferences for a matching inference ID.
async fn query_inference(
    pool: &sqlx::PgPool,
    id: &Uuid,
) -> Result<Option<(String, FunctionType, String, Uuid)>, Error> {
    // Try chat_inferences first
    let row = sqlx::query_as::<_, InferenceRow>(
        "SELECT function_name, variant_name, episode_id FROM tensorzero.chat_inferences WHERE id = $1 LIMIT 1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to query chat_inferences: {e}"),
        })
    })?;

    if let Some(row) = row {
        return Ok(Some((
            row.function_name,
            FunctionType::Chat,
            row.variant_name,
            row.episode_id,
        )));
    }

    // Try json_inferences
    let row = sqlx::query_as::<_, InferenceRow>(
        "SELECT function_name, variant_name, episode_id FROM tensorzero.json_inferences WHERE id = $1 LIMIT 1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to query json_inferences: {e}"),
        })
    })?;

    if let Some(row) = row {
        return Ok(Some((
            row.function_name,
            FunctionType::Json,
            row.variant_name,
            row.episode_id,
        )));
    }

    Ok(None)
}

/// Query for an episode by checking if any inference has this episode_id.
async fn query_episode(pool: &sqlx::PgPool, id: &Uuid) -> Result<bool, Error> {
    let row = sqlx::query_scalar::<_, i32>(
        "SELECT 1 FROM tensorzero.chat_inferences WHERE episode_id = $1 LIMIT 1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to query episodes in chat_inferences: {e}"),
        })
    })?;

    if row.is_some() {
        return Ok(true);
    }

    let row = sqlx::query_scalar::<_, i32>(
        "SELECT 1 FROM tensorzero.json_inferences WHERE episode_id = $1 LIMIT 1",
    )
    .bind(id)
    .fetch_optional(pool)
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::PostgresQuery {
            message: format!("Failed to query episodes in json_inferences: {e}"),
        })
    })?;

    Ok(row.is_some())
}

/// Check if a row with the given ID exists in the specified table.
async fn query_exists(pool: &sqlx::PgPool, table: &str, id: &Uuid) -> Result<bool, Error> {
    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("SELECT 1 FROM ");
    qb.push(table);
    qb.push(" WHERE id = ");
    qb.push_bind(id);
    qb.push(" LIMIT 1");

    let row = qb
        .build_query_scalar::<i32>()
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query {table}: {e}"),
            })
        })?;

    Ok(row.is_some())
}

/// Query a datapoint table for the given ID and return dataset_name and function_name.
async fn query_datapoint(
    pool: &sqlx::PgPool,
    table: &str,
    id: &Uuid,
) -> Result<Option<DatapointRow>, Error> {
    let mut qb: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("SELECT dataset_name, function_name FROM ");
    qb.push(table);
    qb.push(" WHERE id = ");
    qb.push_bind(id);
    qb.push(" LIMIT 1");

    let row = qb
        .build_query_as::<DatapointRow>()
        .fetch_optional(pool)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::PostgresQuery {
                message: format!("Failed to query {table}: {e}"),
            })
        })?;

    Ok(row)
}
