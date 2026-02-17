//! DICL (Dynamic In-Context Learning) queries for Postgres.
//!
//! This module implements similarity search for DICL examples using pgvector.
//! Since pgvector doesn't have a compatible sqlx 0.9 crate yet, we handle vectors
//! manually using SQL casting.

use sqlx::{PgPool, Row};
use uuid::Uuid;

use async_trait::async_trait;

use crate::db::{DICLExampleWithDistance, DICLQueries, StoredDICLExample};
use crate::error::Error;

use super::PostgresConnectionInfo;

#[async_trait]
impl DICLQueries for PostgresConnectionInfo {
    async fn insert_dicl_example(&self, example: &StoredDICLExample) -> Result<(), Error> {
        self.insert_dicl_examples(std::slice::from_ref(example))
            .await?;
        Ok(())
    }

    async fn insert_dicl_examples(&self, examples: &[StoredDICLExample]) -> Result<u64, Error> {
        if examples.is_empty() {
            return Ok(0);
        }

        let pool = self.get_pool_result()?;

        // Process in batches to avoid query size limits
        // Embeddings can be large, so use smaller batches
        const BATCH_SIZE: usize = 100;

        let mut total_inserted = 0u64;

        for batch in examples.chunks(BATCH_SIZE) {
            let rows_affected = insert_dicl_batch(pool, batch).await?;
            total_inserted += rows_affected;
        }

        Ok(total_inserted)
    }

    async fn get_similar_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        embedding: &[f32],
        limit: u32,
    ) -> Result<Vec<DICLExampleWithDistance>, Error> {
        let pool = self.get_pool_result()?;

        let embedding_str = format_embedding_for_postgres(embedding);

        // Use cosine distance operator <=> for similarity search
        // Results are ordered by distance ascending (most similar first)
        let rows = sqlx::query(
            r"
            SELECT input, output, embedding <=> $4::vector AS cosine_distance
            FROM tensorzero.dicl_examples
            WHERE function_name = $1
              AND variant_name = $2
            ORDER BY cosine_distance ASC
            LIMIT $3
            ",
        )
        .bind(function_name)
        .bind(variant_name)
        .bind(limit as i64)
        .bind(&embedding_str)
        .fetch_all(pool)
        .await?;

        let examples = rows
            .into_iter()
            .map(|row| {
                let input: String = row.get("input");
                let output: String = row.get("output");
                let cosine_distance: f64 = row.get("cosine_distance");

                DICLExampleWithDistance {
                    input,
                    output,
                    cosine_distance: cosine_distance as f32,
                }
            })
            .collect();

        Ok(examples)
    }

    async fn has_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
    ) -> Result<bool, Error> {
        let pool = self.get_pool_result()?;

        let row = sqlx::query(
            r"
            SELECT EXISTS(
                SELECT 1 FROM tensorzero.dicl_examples
                WHERE function_name = $1 AND variant_name = $2
            ) AS exists
            ",
        )
        .bind(function_name)
        .bind(variant_name)
        .fetch_one(pool)
        .await?;

        let exists: bool = row.get("exists");
        Ok(exists)
    }

    async fn delete_dicl_examples(
        &self,
        function_name: &str,
        variant_name: &str,
        namespace: Option<&str>,
    ) -> Result<u64, Error> {
        let pool = self.get_pool_result()?;

        let result = if let Some(ns) = namespace {
            sqlx::query(
                r"
                DELETE FROM tensorzero.dicl_examples
                WHERE function_name = $1 AND variant_name = $2 AND namespace = $3
                ",
            )
            .bind(function_name)
            .bind(variant_name)
            .bind(ns)
            .execute(pool)
            .await?
        } else {
            sqlx::query(
                r"
                DELETE FROM tensorzero.dicl_examples
                WHERE function_name = $1 AND variant_name = $2
                ",
            )
            .bind(function_name)
            .bind(variant_name)
            .execute(pool)
            .await?
        };

        Ok(result.rows_affected())
    }
}

/// Format an embedding as a pgvector-compatible string: "[0.1,0.2,0.3]"
fn format_embedding_for_postgres(embedding: &[f32]) -> String {
    let values: Vec<String> = embedding.iter().map(|v| v.to_string()).collect();
    format!("[{}]", values.join(","))
}

/// Insert a batch of DICL examples using a single query with UNNEST.
async fn insert_dicl_batch(pool: &PgPool, examples: &[StoredDICLExample]) -> Result<u64, Error> {
    if examples.is_empty() {
        return Ok(0);
    }

    // Collect arrays for UNNEST-based batch insert
    let ids: Vec<Uuid> = examples.iter().map(|e| e.id).collect();
    let function_names: Vec<&str> = examples.iter().map(|e| e.function_name.as_str()).collect();
    let variant_names: Vec<&str> = examples.iter().map(|e| e.variant_name.as_str()).collect();
    let namespaces: Vec<&str> = examples.iter().map(|e| e.namespace.as_str()).collect();
    let inputs: Vec<&str> = examples.iter().map(|e| e.input.as_str()).collect();
    let outputs: Vec<&str> = examples.iter().map(|e| e.output.as_str()).collect();
    let embeddings: Vec<String> = examples
        .iter()
        .map(|e| format_embedding_for_postgres(&e.embedding))
        .collect();
    let created_ats: Vec<chrono::DateTime<chrono::Utc>> =
        examples.iter().map(|e| e.created_at).collect();

    // Use UNNEST for efficient batch insert
    // The embedding strings are cast to vector type in SQL
    let result = sqlx::query(
        r"
        INSERT INTO tensorzero.dicl_examples (
            id, function_name, variant_name, namespace, input, output, embedding, created_at
        )
        SELECT * FROM UNNEST(
            $1::uuid[],
            $2::text[],
            $3::text[],
            $4::text[],
            $5::text[],
            $6::text[],
            $7::vector[],
            $8::timestamptz[]
        )
        ",
    )
    .bind(&ids)
    .bind(&function_names)
    .bind(&variant_names)
    .bind(&namespaces)
    .bind(&inputs)
    .bind(&outputs)
    .bind(&embeddings)
    .bind(&created_ats)
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_embedding_for_postgres() {
        let embedding = vec![0.1, 0.2, 0.3];
        let result = format_embedding_for_postgres(&embedding);
        assert_eq!(result, "[0.1,0.2,0.3]");
    }

    #[test]
    fn test_format_embedding_empty() {
        let embedding: Vec<f32> = vec![];
        let result = format_embedding_for_postgres(&embedding);
        assert_eq!(result, "[]");
    }

    #[test]
    fn test_format_embedding_single() {
        let embedding = vec![1.5];
        let result = format_embedding_for_postgres(&embedding);
        assert_eq!(result, "[1.5]");
    }

    #[test]
    fn test_format_embedding_negative_values() {
        let embedding = vec![-0.5, 0.0, 0.5];
        let result = format_embedding_for_postgres(&embedding);
        assert_eq!(result, "[-0.5,0,0.5]");
    }
}
