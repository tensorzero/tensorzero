use super::check_table_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{ErrorDetails, delayed_error::DelayedError};
use async_trait::async_trait;

/// Fixes `function_type` in `InferenceEvaluationRuns` that was incorrectly backfilled by
/// migration 0049. The bug: ClickHouse UUID is non-nullable, so a LEFT JOIN on
/// `ChatInference.id` produces a zero UUID instead of NULL for non-matching rows, causing
/// `count(ci.id) > 0` to always be true and setting every run's `function_type` to `'chat'`.
///
/// This migration re-derives `function_type` by checking `JsonInference` instead and inserts
/// corrected rows (ReplacingMergeTree deduplicates by `run_id_uint`, keeping the latest
/// `updated_at`).
pub struct Migration0050<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0050";

#[async_trait]
impl Migration for Migration0050<'_> {
    async fn can_apply(&self) -> Result<(), DelayedError> {
        let tables_to_check = [
            "InferenceEvaluationRuns",
            "TagInference",
            "JsonInference",
            "BooleanMetricFeedback",
            "FloatMetricFeedback",
        ];

        for table_name in tables_to_check {
            if !check_table_exists(self.clickhouse, table_name, MIGRATION_ID).await? {
                return Err(DelayedError::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("{table_name} table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, DelayedError> {
        // Check if the migration manager has already recorded this migration as successful.
        // If so, skip it. Otherwise, run it once so the manager writes the row.
        let query = format!(
            "SELECT 1 FROM {}.TensorZeroMigration WHERE migration_id = {MIGRATION_ID} LIMIT 1",
            self.clickhouse.database()
        );
        let response = self
            .clickhouse
            .run_query_synchronous_no_params_delayed_err(query)
            .await?;
        Ok(response.response.trim() != "1")
    }

    async fn apply(&self, clean_start: bool) -> Result<(), DelayedError> {
        if clean_start {
            // On a clean start, migration 0049 ran with clean_start too (no backfill),
            // so there is nothing to fix.
            return Ok(());
        }

        // Re-derive all InferenceEvaluationRuns from TagInference and feedback tables.
        // The previous approach read from InferenceEvaluationRuns itself and INNER JOINed
        // with TagInference, which dropped runs whose inferences had no evaluation tags.
        // Instead, we re-run the full backfill from source data with corrected function_type
        // logic: check JsonInference (non-nullable UUID → use countIf with zero-UUID guard).
        // ReplacingMergeTree deduplicates by run_id_uint, keeping the latest updated_at.
        //
        // We LEFT JOIN with existing rows to preserve `source` and `metrics` for runs that
        // were already written correctly (e.g. by normal evaluation writes between migrations
        // 0049 and 0050). Only fall back to backfill-derived values for rows that don't exist.
        self.clickhouse
            .run_query_synchronous_no_params_delayed_err(
                r#"
                INSERT INTO InferenceEvaluationRuns
                (
                    run_id_uint,
                    evaluation_name,
                    function_name,
                    function_type,
                    dataset_name,
                    variant_names,
                    metrics,
                    source,
                    snapshot_hash,
                    created_at,
                    updated_at
                )
                WITH
                evaluation_inferences AS (
                    SELECT
                        toUUIDOrNull(maxIf(value, key = 'tensorzero::evaluation_run_id')) AS run_id,
                        maxIf(value, key = 'tensorzero::evaluation_name') AS evaluation_name,
                        maxIf(value, key = 'tensorzero::dataset_name') AS dataset_name,
                        any(function_name) AS function_name,
                        any(variant_name) AS variant_name,
                        any(snapshot_hash) AS snapshot_hash,
                        inference_id,
                        UUIDv7ToDateTime(inference_id) AS inference_timestamp
                    FROM TagInference
                    WHERE key IN ('tensorzero::evaluation_run_id', 'tensorzero::evaluation_name', 'tensorzero::dataset_name')
                    GROUP BY inference_id
                    HAVING
                        run_id IS NOT NULL
                        AND evaluation_name != ''
                        AND dataset_name != ''
                        AND NOT startsWith(function_name, 'tensorzero::')
                ),
                metrics_flat AS (
                    SELECT
                        ei.run_id AS run_id,
                        b.metric_name AS metric_name,
                        'boolean' AS value_type
                    FROM BooleanMetricFeedback b
                    INNER JOIN evaluation_inferences ei ON b.target_id = ei.inference_id

                    UNION ALL

                    SELECT
                        ei.run_id AS run_id,
                        f.metric_name AS metric_name,
                        'float' AS value_type
                    FROM FloatMetricFeedback f
                    INNER JOIN evaluation_inferences ei ON f.target_id = ei.inference_id
                ),
                metrics_by_run AS (
                    SELECT
                        run_id,
                        concat(
                            '[',
                            arrayStringConcat(
                                groupUniqArray(
                                    concat(
                                        '{"name":', toJSONString(metric_name),
                                        ',"evaluator_name":',
                                        if(
                                            position(metric_name, '::evaluator_name::') > 0,
                                            toJSONString(arrayElement(splitByString('::evaluator_name::', metric_name), 2)),
                                            'null'
                                        ),
                                        ',"value_type":', toJSONString(value_type),
                                        ',"optimize":null}'
                                    )
                                ),
                                ','
                            ),
                            ']'
                        ) AS metrics
                    FROM metrics_flat
                    GROUP BY run_id
                ),
                run_function_types AS (
                    SELECT
                        ei.run_id AS run_id,
                        if(
                            countIf(ji.id != toUUID('00000000-0000-0000-0000-000000000000')) > 0,
                            'json',
                            'chat'
                        ) AS function_type
                    FROM evaluation_inferences ei
                    LEFT JOIN JsonInference ji ON ji.id = ei.inference_id
                    GROUP BY ei.run_id
                ),
                existing_runs AS (
                    SELECT
                        run_id_uint,
                        argMax(metrics, updated_at) AS metrics,
                        argMax(source, updated_at) AS source
                    FROM InferenceEvaluationRuns
                    GROUP BY run_id_uint
                )
                SELECT
                    toUInt128(ei.run_id) AS run_id_uint,
                    any(ei.evaluation_name) AS evaluation_name,
                    any(ei.function_name) AS function_name,
                    any(rft.function_type) AS function_type,
                    any(ei.dataset_name) AS dataset_name,
                    arrayDistinct(groupArray(ei.variant_name)) AS variant_names,
                    coalesce(nullIf(any(er.metrics), ''), any(mbr.metrics), '[]') AS metrics,
                    coalesce(nullIf(any(er.source), ''), 'dataset_name') AS source,
                    if(isNull(any(ei.snapshot_hash)), NULL, lower(hex(any(ei.snapshot_hash)))) AS snapshot_hash,
                    min(ei.inference_timestamp) AS created_at,
                    now64(3) AS updated_at
                FROM evaluation_inferences ei
                LEFT JOIN metrics_by_run mbr ON ei.run_id = mbr.run_id
                LEFT JOIN run_function_types rft ON ei.run_id = rft.run_id
                LEFT JOIN existing_runs er ON toUInt128(ei.run_id) = er.run_id_uint
                GROUP BY ei.run_id
                "#
                .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        // We include 'SELECT 1' so that our test code can run these rollback instructions
        "SELECT 1;".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, DelayedError> {
        // The table already existed; we just inserted corrected rows.
        check_table_exists(self.clickhouse, "InferenceEvaluationRuns", MIGRATION_ID).await
    }
}
