use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::check_table_exists;

/// This migration adds a table StaticEvaluationHumanFeedback that stores human feedback in an easy-to-reference format.
/// This is technically an auxiliary table as the primary store is still the various feedback tables.
/// We also create two materialized views that automatically write to StaticEvaluationHumanFeedback when
/// FloatMetricFeedback and BooleanMetricFeedback are updated with new feedback that contains both
/// tensorzero::datapoint_id and tensorzero::human_feedback tags.
pub struct Migration0023<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0023<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        // As long as FloatMetricFeedback and BooleanMetricFeedback exist, we can apply this migration
        if !check_table_exists(self.clickhouse, "FloatMetricFeedback", "0023").await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: "0023".to_string(),
                message: "FloatMetricFeedback table does not exist".to_string(),
            }));
        }
        if !check_table_exists(self.clickhouse, "BooleanMetricFeedback", "0023").await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: "0023".to_string(),
                message: "BooleanMetricFeedback table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let human_feedback_table_exists =
            check_table_exists(self.clickhouse, "StaticEvaluationHumanFeedback", "0023").await?;
        let float_materialized_view_exists = check_table_exists(
            self.clickhouse,
            "StaticEvaluationHumanFeedbackFloatView",
            "0023",
        )
        .await?;
        let boolean_materialized_view_exists = check_table_exists(
            self.clickhouse,
            "StaticEvaluationHumanFeedbackBooleanView",
            "0023",
        )
        .await?;
        Ok(!human_feedback_table_exists
            || !float_materialized_view_exists
            || !boolean_materialized_view_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous(
                r#"CREATE TABLE IF NOT EXISTS StaticEvaluationHumanFeedback (
                    metric_name LowCardinality(String),
                    datapoint_id UUID,
                    output String,
                    value String,  -- JSON encoded value of the feedback
                    feedback_id UUID,
                    evaluator_inference_id UUID,
                    timestamp DateTime MATERIALIZED UUIDv7ToDateTime(feedback_id)
                ) ENGINE = MergeTree()
                ORDER BY (metric_name, datapoint_id, output)
                SETTINGS index_granularity = 256 -- We use a small index granularity to improve lookup performance
            "#.to_string(),
                None,
            )
            .await?;

        // Since there cannot have been any StaticEvaluationHumanFeedback rows before this migration runs,
        // we can just create the materialized views in place.

        // Create the materialized view for FloatMetricFeedback
        let query = r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS StaticEvaluationHumanFeedbackFloatView
            TO StaticEvaluationHumanFeedback
            AS
                SELECT
                    f.metric_name as metric_name,
                    toUUID(f.tags['tensorzero::datapoint_id']) AS datapoint_id,
                    -- Select output based on function_type determined via InferenceById join
                    if(i.function_type = 'chat', ci.output, ji.output) AS output,
                    toJSONString(f.value) AS value,
                    f.id AS feedback_id,
                    -- we enforce in the feedback endpoint that this is present
                    -- if the tensorzero::datapoint_id is present and the tensorzero::human_feedback
                    -- tag is present.
                    toUUID(f.tags['tensorzero::evaluator_inference_id']) AS evaluator_inference_id
                FROM FloatMetricFeedback f
                INNER JOIN InferenceById i ON
                    toUInt128(f.target_id) = i.id_uint
                LEFT JOIN ChatInference ci ON
                    i.function_type = 'chat' AND
                    i.function_name = ci.function_name AND
                    i.variant_name = ci.variant_name AND
                    i.episode_id = ci.episode_id AND
                    f.target_id = ci.id
                LEFT JOIN JsonInference ji ON
                    i.function_type = 'json' AND
                    i.function_name = ji.function_name AND
                    i.variant_name = ji.variant_name AND
                    i.episode_id = ji.episode_id AND
                    f.target_id = ji.id
                WHERE mapContains(f.tags, 'tensorzero::datapoint_id')
                      AND mapContains(f.tags, 'tensorzero::human_feedback');
        "#;
        self.clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        // Create the materialized view for BooleanMetricFeedback
        let query = r#"
        CREATE MATERIALIZED VIEW IF NOT EXISTS StaticEvaluationHumanFeedbackBooleanView
        TO StaticEvaluationHumanFeedback
        AS
            SELECT
                f.metric_name as metric_name,
                toUUID(f.tags['tensorzero::datapoint_id']) AS datapoint_id,
                -- Select output based on function_type determined via InferenceById join
                if(i.function_type = 'chat', ci.output, ji.output) AS output,
                toJSONString(f.value) AS value,
                f.id AS feedback_id,
                -- we enforce in the feedback endpoint that this is present
                -- if the tensorzero::datapoint_id is present and the tensorzero::human_feedback
                -- tag is present.
                toUUID(f.tags['tensorzero::evaluator_inference_id']) AS evaluator_inference_id
            FROM BooleanMetricFeedback f
            INNER JOIN InferenceById i ON
                toUInt128(f.target_id) = i.id_uint
            LEFT JOIN ChatInference ci ON
                i.function_type = 'chat' AND
                i.function_name = ci.function_name AND
                i.variant_name = ci.variant_name AND
                i.episode_id = ci.episode_id AND
                f.target_id = ci.id
            LEFT JOIN JsonInference ji ON
                i.function_type = 'json' AND
                i.function_name = ji.function_name AND
                i.variant_name = ji.variant_name AND
                i.episode_id = ji.episode_id AND
                f.target_id = ji.id
            WHERE mapContains(f.tags, 'tensorzero::datapoint_id')
                  AND mapContains(f.tags, 'tensorzero::human_feedback');
    "#;
        self.clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r#"DROP TABLE IF EXISTS StaticEvaluationHumanFeedback;
        DROP MATERIALIZED VIEW IF EXISTS StaticEvaluationHumanFeedbackFloatView;
        DROP MATERIALIZED VIEW IF EXISTS StaticEvaluationHumanFeedbackBooleanView;
        "#
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
