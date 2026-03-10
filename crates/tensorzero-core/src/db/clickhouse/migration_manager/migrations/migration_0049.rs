use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

pub struct Migration0049<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0049";

#[async_trait]
impl Migration for Migration0049<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let tables_to_check = [
            "TagInference",
            "FloatMetricFeedback",
            "BooleanMetricFeedback",
        ];

        for table_name in tables_to_check {
            if !check_table_exists(self.clickhouse, table_name, MIGRATION_ID).await? {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("{table_name} table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        if !check_table_exists(self.clickhouse, "InferenceEvaluationRuns", MIGRATION_ID).await? {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "InferenceEvaluationRuns",
                engine_args: &["updated_at"],
            },
        );
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"
                CREATE TABLE IF NOT EXISTS InferenceEvaluationRuns{on_cluster_name}
                (
                    run_id_uint UInt128,
                    evaluation_name String,
                    function_name String,
                    function_type LowCardinality(String),
                    dataset_name String,
                    variant_names Array(String),
                    metrics String,
                    source LowCardinality(String),
                    snapshot_hash Nullable(String),
                    created_at DateTime64(3, 'UTC'),
                    updated_at DateTime64(3, 'UTC')
                )
                ENGINE = {table_engine_name}
                ORDER BY (run_id_uint)
                "
            ))
            .await?;

        if clean_start {
            return Ok(());
        }

        // Best-effort backfill from historical inference tags and feedback.
        // `metrics.optimize` is unknown for historical runs and is set to null.
        self.clickhouse
            .run_query_synchronous_no_params(
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
                        if(count(ci.id) > 0, 'chat', 'json') AS function_type
                    FROM evaluation_inferences ei
                    LEFT JOIN ChatInference ci ON ci.id = ei.inference_id
                    GROUP BY ei.run_id
                )
                SELECT
                    toUInt128(ei.run_id) AS run_id_uint,
                    any(ei.evaluation_name) AS evaluation_name,
                    any(ei.function_name) AS function_name,
                    any(rft.function_type) AS function_type,
                    any(ei.dataset_name) AS dataset_name,
                    arrayDistinct(groupArray(ei.variant_name)) AS variant_names,
                    coalesce(any(mbr.metrics), '[]') AS metrics,
                    'dataset_name' AS source,
                    if(isNull(any(ei.snapshot_hash)), NULL, lower(hex(any(ei.snapshot_hash)))) AS snapshot_hash,
                    min(ei.inference_timestamp) AS created_at,
                    now64(3) AS updated_at
                FROM evaluation_inferences ei
                LEFT JOIN metrics_by_run mbr ON ei.run_id = mbr.run_id
                LEFT JOIN run_function_types rft ON ei.run_id = rft.run_id
                GROUP BY ei.run_id
                "#
                .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "DROP TABLE IF EXISTS InferenceEvaluationRuns SYNC;".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        check_table_exists(self.clickhouse, "InferenceEvaluationRuns", MIGRATION_ID).await
    }
}
