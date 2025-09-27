use std::time::Duration;

use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_detached_table_exists, check_table_exists};

/// NOTE: This migration supersedes migration_0023.
/// Migration 0023 was inefficient due to its use of joins.
/// In this migration, we create the same table StaticEvaluationHumanFeedback if it doesn't exist.
/// We then drop the old materialized view and create new ones that are more efficient.
/// This migration adds a table StaticEvaluationHumanFeedback that stores human feedback in an easy-to-reference format.
/// This is technically an auxiliary table as the primary store is still the various feedback tables.
/// We also create two materialized views that automatically write to StaticEvaluationHumanFeedback when
/// FloatMetricFeedback and BooleanMetricFeedback are updated with new feedback that contains both
/// tensorzero::datapoint_id and tensorzero::human_feedback tags.
///
/// In particular, since we don't care if there are duplicate rows and we need the migration to work with concurrent gateways,
/// we do a full cutover to the new view, meaning that if we determine that this migration needs to run and we're not clean starting,
/// we:
///   1. Fix a timestamp for the migration
///   2. Drop the old views if they exist
///   3. Create the new views with a timestamp of the migration timestamp + 10 seconds
///   4. Wait 10 seconds
///   5. Insert the data from the missing 10 seconds into the new table.
///
///
/// NOTE: The views created by this migration are StaticEvaluationFloatHumanFeedbackView and StaticEvaluationBooleanHumanFeedbackView.
/// The views created by Migration 0023 are StaticEvaluationHumanFeedbackFloatView and StaticEvaluationHumanFeedbackBooleanView.s
///
/// NOTE: The two views created by this migration are required to be separate as ClickHouse only triggers materialized views
/// on the first table to appear.
pub struct Migration0028<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0028";

#[async_trait]
impl Migration for Migration0028<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        // As long as FloatMetricFeedback and BooleanMetricFeedback exist, we can apply this migration
        if !check_table_exists(self.clickhouse, "FloatMetricFeedback", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "FloatMetricFeedback table does not exist".to_string(),
            }));
        }
        if !check_table_exists(self.clickhouse, "BooleanMetricFeedback", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "BooleanMetricFeedback table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let human_feedback_table_exists = check_table_exists(
            self.clickhouse,
            "StaticEvaluationHumanFeedback",
            MIGRATION_ID,
        )
        .await?;
        let float_materialized_view_exists = check_table_exists(
            self.clickhouse,
            "StaticEvaluationFloatHumanFeedbackView",
            MIGRATION_ID,
        )
        .await?;
        let detached_float_materialized_view_exists = check_detached_table_exists(
            self.clickhouse,
            "StaticEvaluationFloatHumanFeedbackView",
            MIGRATION_ID,
        )
        .await?;
        let boolean_materialized_view_exists = check_table_exists(
            self.clickhouse,
            "StaticEvaluationBooleanHumanFeedbackView",
            MIGRATION_ID,
        )
        .await?;
        let detached_boolean_materialized_view_exists = check_detached_table_exists(
            self.clickhouse,
            "StaticEvaluationBooleanHumanFeedbackView",
            MIGRATION_ID,
        )
        .await?;
        Ok(!human_feedback_table_exists
            // if the views exist in either the active or detached state that is fine for the purposes of 0028
            || !(float_materialized_view_exists || detached_float_materialized_view_exists)
            || !(boolean_materialized_view_exists || detached_boolean_materialized_view_exists))
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        let view_offset = Duration::from_secs(10);
        let current_time = std::time::SystemTime::now();
        let view_timestamp = (current_time
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_secs();
        let view_timestamp_where_clause = if clean_start {
            String::new()
        } else {
            format!("AND UUIDv7ToDateTime(feedback_id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        };
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        self.clickhouse
            .get_create_table_statements(
                "StaticEvaluationHumanFeedback",
                &format!(
                    r"(
                        metric_name LowCardinality(String),
                        datapoint_id UUID,
                        output String,
                        value String,  -- JSON encoded value of the feedback
                        feedback_id UUID,
                        evaluator_inference_id UUID,
                        timestamp DateTime MATERIALIZED UUIDv7ToDateTime(feedback_id)
                    )"
                ),
                &GetMaybeReplicatedTableEngineNameArgs {
                    table_name: "StaticEvaluationHumanFeedback",
                    table_engine_name: "MergeTree",
                    engine_args: &[],
                },
                Some("ORDER BY (metric_name, datapoint_id, output) SETTINGS index_granularity = 256"),
            )
            .await?;

        // Since there cannot have been any StaticEvaluationHumanFeedback rows before this migration runs,
        // we can just create the materialized views in place.

        let static_evaluation_human_feedback_target = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("StaticEvaluationHumanFeedback")
        } else {
            "StaticEvaluationHumanFeedback".to_string()
        };
        let float_metric_feedback_source = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("FloatMetricFeedback")
        } else {
            "FloatMetricFeedback".to_string()
        };
        let inference_by_id_source = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("InferenceById")
        } else {
            "InferenceById".to_string()
        };
        let chat_inference_source = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("ChatInference")
        } else {
            "ChatInference".to_string()
        };
        let json_inference_source = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("JsonInference")
        } else {
            "JsonInference".to_string()
        };
        let boolean_metric_feedback_source = if self.clickhouse.is_sharding_enabled() {
            self.clickhouse.get_local_table_name("BooleanMetricFeedback")
        } else {
            "BooleanMetricFeedback".to_string()
        };

        // Create the materialized view for FloatMetricFeedback
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS StaticEvaluationFloatHumanFeedbackView{on_cluster_name}
            TO {static_evaluation_human_feedback_target}
            AS
                WITH float_human_feedback AS (
                    SELECT
                        metric_name,
                        toUUID(tags['tensorzero::datapoint_id']) AS datapoint_id,
                        toJSONString(value) AS value,
                        id AS feedback_id,
                        target_id,
                        toUUID(tags['tensorzero::evaluator_inference_id']) AS evaluator_inference_id
                        -- we enforce in the feedback endpoint that this is present
                        -- if the tensorzero::datapoint_id is present and the tensorzero::human_feedback
                        -- tag is present.
                    FROM {float_metric_feedback_source}
                    WHERE
                        mapContains(tags, 'tensorzero::human_feedback')
                        AND mapContains(tags, 'tensorzero::datapoint_id')
                        {view_timestamp_where_clause}

                ),
                inference_by_id AS (
                    SELECT id_uint, function_name, function_type, variant_name, episode_id FROM {inference_by_id_source}
                    WHERE id_uint IN (
                        SELECT toUInt128(target_id) FROM float_human_feedback
                    )
                ),
                chat_inference AS (
                    SELECT function_name, variant_name, episode_id, id, output FROM {chat_inference_source}
                    WHERE (function_name, variant_name, episode_id) IN (
                        SELECT function_name, variant_name, episode_id
                        FROM inference_by_id
                    )
                ),
                json_inference AS (
                    SELECT function_name, variant_name, episode_id, id, output FROM {json_inference_source}
                    WHERE (function_name, variant_name, episode_id) IN (
                        SELECT function_name, variant_name, episode_id
                        FROM inference_by_id
                    )
                )
            SELECT
                f.metric_name as metric_name,
                f.datapoint_id as datapoint_id,
                -- Select output based on function_type determined via InferenceById join
                if(i.function_type = 'chat', ci.output, ji.output) AS output,
                f.value AS value,
                f.feedback_id AS feedback_id,
                f.evaluator_inference_id AS evaluator_inference_id
            FROM float_human_feedback f
            INNER JOIN inference_by_id i ON
                toUInt128(f.target_id) = i.id_uint
            LEFT JOIN chat_inference ci ON
                i.function_type = 'chat' AND
                i.function_name = ci.function_name AND
                i.variant_name = ci.variant_name AND
                i.episode_id = ci.episode_id AND
                f.target_id = ci.id
            LEFT JOIN json_inference ji ON
                i.function_type = 'json' AND
                i.function_name = ji.function_name AND
                i.variant_name = ji.variant_name AND
                i.episode_id = ji.episode_id AND
                f.target_id = ji.id;
        ",
            on_cluster_name = on_cluster_name,
            static_evaluation_human_feedback_target = static_evaluation_human_feedback_target,
            float_metric_feedback_source = float_metric_feedback_source,
            inference_by_id_source = inference_by_id_source,
            chat_inference_source = chat_inference_source,
            json_inference_source = json_inference_source,
            view_timestamp_where_clause = view_timestamp_where_clause,
        );
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the materialized view for BooleanMetricFeedback
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS StaticEvaluationBooleanHumanFeedbackView{on_cluster_name}
            TO {static_evaluation_human_feedback_table_name}
            AS
                WITH boolean_human_feedback AS (
                    SELECT
                       metric_name,
                       toUUID(tags['tensorzero::datapoint_id']) AS datapoint_id,
                       toJSONString(value) AS value,
                       id AS feedback_id,
                       target_id,
                       toUUID(tags['tensorzero::evaluator_inference_id']) AS evaluator_inference_id
                       -- we enforce in the feedback endpoint that this is present
                       -- if the tensorzero::datapoint_id is present and the tensorzero::human_feedback
                       -- tag is present.
                   FROM {boolean_metric_feedback_table_name}
                   WHERE
                       mapContains(tags, 'tensorzero::human_feedback')
                       AND mapContains(tags, 'tensorzero::datapoint_id')
                       {view_timestamp_where_clause}

                ),
                inference_by_id AS (
                    SELECT id_uint, function_name, function_type, variant_name, episode_id FROM {inference_by_id_table_name}
                    WHERE id_uint IN (
                        SELECT toUInt128(target_id) FROM boolean_human_feedback
                    )
                ),
                chat_inference AS (
                    SELECT function_name, variant_name, episode_id, id, output FROM {chat_inference_table_name}
                    WHERE (function_name, variant_name, episode_id) IN (
                        SELECT function_name, variant_name, episode_id
                        FROM inference_by_id
                    )
                ),
                json_inference AS (
                    SELECT function_name, variant_name, episode_id, id, output FROM {json_inference_table_name}
                    WHERE (function_name, variant_name, episode_id) IN (
                        SELECT function_name, variant_name, episode_id
                        FROM inference_by_id
                    )
                )
            SELECT
                f.metric_name as metric_name,
                f.datapoint_id as datapoint_id,
                -- Select output based on function_type determined via InferenceById join
                if(i.function_type = 'chat', ci.output, ji.output) AS output,
                f.value AS value,
                f.feedback_id AS feedback_id,
                f.evaluator_inference_id AS evaluator_inference_id
            FROM boolean_human_feedback f
            INNER JOIN inference_by_id i ON
                toUInt128(f.target_id) = i.id_uint
            LEFT JOIN chat_inference ci ON
                i.function_type = 'chat' AND
                i.function_name = ci.function_name AND
                i.variant_name = ci.variant_name AND
                i.episode_id = ci.episode_id AND
                f.target_id = ci.id
            LEFT JOIN json_inference ji ON
                i.function_type = 'json' AND
                i.function_name = ji.function_name AND
                i.variant_name = ji.variant_name AND
                i.episode_id = ji.episode_id AND
                f.target_id = ji.id;
        ",
            on_cluster_name = on_cluster_name,
            static_evaluation_human_feedback_table_name = static_evaluation_human_feedback_target,
            boolean_metric_feedback_table_name = boolean_metric_feedback_source,
            inference_by_id_table_name = inference_by_id_source,
            chat_inference_table_name = chat_inference_source,
            json_inference_table_name = json_inference_source,
            view_timestamp_where_clause = view_timestamp_where_clause,
        );

        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        if !clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;
            let current_timestamp = current_time
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseMigration {
                        id: MIGRATION_ID.to_string(),
                        message: e.to_string(),
                    })
                })?
                .as_secs();
            let insert_timestamp_where_clause = format!(
                r"
                AND UUIDv7ToDateTime(feedback_id) < toDateTime(toUnixTimestamp({view_timestamp}))
                AND UUIDv7ToDateTime(feedback_id) >= toDateTime(toUnixTimestamp({current_timestamp}))"
            );
            
            // For data insertion operations, use distributed table names
            let static_evaluation_human_feedback_target_for_insert = "StaticEvaluationHumanFeedback";
            let float_metric_feedback_source_for_insert = "FloatMetricFeedback";
            let boolean_metric_feedback_source_for_insert = "BooleanMetricFeedback";
            let inference_by_id_source_for_insert = "InferenceById";
            let chat_inference_source_for_insert = "ChatInference";
            let json_inference_source_for_insert = "JsonInference";

            let query = format!(
                r"
                WITH human_feedback AS (
                    SELECT
                        metric_name,
                        toUUID(tags['tensorzero::datapoint_id']) AS datapoint_id,
                        toJSONString(value) AS value,
                        id AS feedback_id,
                        target_id,
                        toUUID(tags['tensorzero::evaluator_inference_id']) AS evaluator_inference_id
                        -- we enforce in the feedback endpoint that this is present
                        -- if the tensorzero::datapoint_id is present and the tensorzero::human_feedback
                        -- tag is present.
                    FROM {float_metric_feedback_source_for_insert}
                    WHERE
                        mapContains(tags, 'tensorzero::human_feedback')
                        AND mapContains(tags, 'tensorzero::datapoint_id')
                        {insert_timestamp_where_clause}
                    UNION ALL
                    SELECT
                       metric_name,
                       toUUID(tags['tensorzero::datapoint_id']) AS datapoint_id,
                       toJSONString(value) AS value,
                       id AS feedback_id,
                       target_id,
                       toUUID(tags['tensorzero::evaluator_inference_id']) AS evaluator_inference_id
                       -- we enforce in the feedback endpoint that this is present
                       -- if the tensorzero::datapoint_id is present and the tensorzero::human_feedback
                       -- tag is present.
                    FROM {boolean_metric_feedback_source_for_insert}
                    WHERE
                       mapContains(tags, 'tensorzero::human_feedback')
                       AND mapContains(tags, 'tensorzero::datapoint_id')
                       {insert_timestamp_where_clause}

                ),
                inference_by_id AS (
                    SELECT id_uint, function_name, function_type, variant_name, episode_id FROM {inference_by_id_source_for_insert}
                    WHERE id_uint IN (
                        SELECT toUInt128(target_id) FROM human_feedback
                    )
                ),
                chat_inference AS (
                    SELECT function_name, variant_name, episode_id, id, output FROM {chat_inference_source_for_insert}
                    WHERE (function_name, variant_name, episode_id) IN (
                        SELECT function_name, variant_name, episode_id
                        FROM inference_by_id
                    )
                ),
                json_inference AS (
                    SELECT function_name, variant_name, episode_id, id, output FROM {json_inference_source_for_insert}
                    WHERE (function_name, variant_name, episode_id) IN (
                        SELECT function_name, variant_name, episode_id
                        FROM inference_by_id
                    )
                )
            INSERT INTO {static_evaluation_human_feedback_target_for_insert}
            SELECT
                f.metric_name as metric_name,
                f.datapoint_id as datapoint_id,
                -- Select output based on function_type determined via InferenceById join
                if(i.function_type = 'chat', ci.output, ji.output) AS output,
                f.value AS value,
                f.feedback_id AS feedback_id,
                f.evaluator_inference_id AS evaluator_inference_id
            FROM human_feedback f
            INNER JOIN inference_by_id i ON
                toUInt128(f.target_id) = i.id_uint
            LEFT JOIN chat_inference ci ON
                i.function_type = 'chat' AND
                i.function_name = ci.function_name AND
                i.variant_name = ci.variant_name AND
                i.episode_id = ci.episode_id AND
                f.target_id = ci.id
            LEFT JOIN json_inference ji ON
                i.function_type = 'json' AND
                i.function_name = ji.function_name AND
                i.variant_name = ji.variant_name AND
                i.episode_id = ji.episode_id AND
                f.target_id = ji.id;
        "
            );
            self.clickhouse
                .run_query_synchronous_no_params(query.to_string())
                .await?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        
        format!(
            "DROP VIEW IF EXISTS StaticEvaluationFloatHumanFeedbackView{on_cluster_name};\
            DROP VIEW IF EXISTS StaticEvaluationBooleanHumanFeedbackView{on_cluster_name};\
            {}",
            self.clickhouse.get_drop_table_rollback_statements("StaticEvaluationHumanFeedback")
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
