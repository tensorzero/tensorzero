use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::migration_manager::migrations::{
    check_column_exists, check_table_exists,
};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

const MIGRATION_ID: &str = "0043";

/// TODO
pub struct Migration0043<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl<'a> Migration for Migration0043<'a> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Check if ConfigSnapshot table doesn't exist
        if !check_table_exists(self.clickhouse, "ConfigSnapshot", MIGRATION_ID).await? {
            return Ok(true);
        }

        // Check if any of the tables is missing the snapshot_hash column
        let tables = [
            "BatchModelInference",
            "BatchRequest",
            "BooleanMetricFeedback",
            "BooleanMetricFeedbackByTargetId",
            "BooleanMetricFeedbackByVariant",
            "ChatInference",
            "ChatInferenceDatapoint",
            "CommentFeedback",
            "CommentFeedbackByTargetId",
            "DemonstrationFeedback",
            "DemonstrationFeedbackByInferenceId",
            "DynamicEvaluationRun",
            "DynamicEvaluationRunByProjectName",
            "DynamicEvaluationRunEpisode",
            "FeedbackTag",
            "FloatMetricFeedback",
            "FloatMetricFeedbackByTargetId",
            "InferenceByEpisodeId",
            "InferenceById",
            "InferenceTag",
            "JsonInference",
            "JsonInferenceDatapoint",
            "ModelInference",
            "TagInference",
        ];

        for table in tables {
            if !check_column_exists(self.clickhouse, table, "snapshot_hash", MIGRATION_ID).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Create ConfigSnapshot table
        let create_table_query = "
            CREATE TABLE IF NOT EXISTS ConfigSnapshot (
                config String,
                extra_templates Map(String, String),
                version_hash UInt256,
                tensorzero_version String,
                created_at DateTime64(6) DEFAULT now(),
                last_used DateTime64(6) DEFAULT now()
            ) ENGINE = ReplacingMergeTree(last_used)
            ORDER BY version_hash
            SETTINGS index_granularity = 256
        ";
        self.clickhouse
            .run_query_synchronous_no_params(create_table_query.to_string())
            .await?;

        // Add snapshot_hash column to existing tables
        let tables = [
            "BatchModelInference",
            "BatchRequest",
            "BooleanMetricFeedback",
            "BooleanMetricFeedbackByTargetId",
            "BooleanMetricFeedbackByVariant",
            "ChatInference",
            "ChatInferenceDatapoint",
            "CommentFeedback",
            "CommentFeedbackByTargetId",
            "DemonstrationFeedback",
            "DemonstrationFeedbackByInferenceId",
            "DynamicEvaluationRun",
            "DynamicEvaluationRunByProjectName",
            "DynamicEvaluationRunEpisode",
            "FeedbackTag",
            "FloatMetricFeedback",
            "FloatMetricFeedbackByTargetId",
            "InferenceByEpisodeId",
            "InferenceById",
            "InferenceTag",
            "JsonInference",
            "JsonInferenceDatapoint",
            "ModelInference",
            "TagInference",
        ];

        for table in tables {
            let query = format!(
                "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS snapshot_hash Nullable(UInt256)"
            );
            self.clickhouse
                .run_query_synchronous_no_params(query)
                .await?;
        }

        // Update materialized views to propagate snapshot_hash from source tables
        // Group 1: Feedback indexing views
        let query = "
            ALTER TABLE BooleanMetricFeedbackByTargetIdView MODIFY QUERY
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                snapshot_hash
            FROM BooleanMetricFeedback
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE FloatMetricFeedbackByTargetIdView MODIFY QUERY
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                snapshot_hash
            FROM FloatMetricFeedback
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE CommentFeedbackByTargetIdView MODIFY QUERY
            SELECT
                id,
                target_id,
                target_type,
                value,
                tags,
                snapshot_hash
            FROM CommentFeedback
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE DemonstrationFeedbackByInferenceIdView MODIFY QUERY
            SELECT
                id,
                inference_id,
                value,
                tags,
                snapshot_hash
            FROM DemonstrationFeedback
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Group 2: Inference indexing views
        let query = "
            ALTER TABLE ChatInferenceByIdView MODIFY QUERY
            SELECT
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                episode_id,
                'chat' AS function_type,
                snapshot_hash
            FROM ChatInference
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE JsonInferenceByIdView MODIFY QUERY
            SELECT
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                episode_id,
                'json' AS function_type,
                snapshot_hash
            FROM JsonInference
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE ChatInferenceByEpisodeIdView MODIFY QUERY
            SELECT
                toUInt128(episode_id) as episode_id_uint,
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                'chat' as function_type,
                snapshot_hash
            FROM ChatInference
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE JsonInferenceByEpisodeIdView MODIFY QUERY
            SELECT
                toUInt128(episode_id) as episode_id_uint,
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                'json' as function_type,
                snapshot_hash
            FROM JsonInference
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Group 3: Tag extraction views
        let query = "
            ALTER TABLE TagChatInferenceView MODIFY QUERY
            SELECT
                function_name,
                variant_name,
                episode_id,
                id as inference_id,
                'chat' as function_type,
                key,
                tags[key] as value,
                snapshot_hash
            FROM ChatInference
            ARRAY JOIN mapKeys(tags) as key
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE TagJsonInferenceView MODIFY QUERY
            SELECT
                function_name,
                variant_name,
                episode_id,
                id as inference_id,
                'json' as function_type,
                key,
                tags[key] as value,
                snapshot_hash
            FROM JsonInference
            ARRAY JOIN mapKeys(tags) as key
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Group 4: Feedback by variant views (with JOINs)
        let query = "
            ALTER TABLE FloatMetricFeedbackByVariantView MODIFY QUERY
            WITH
                float_feedback AS (
                    SELECT
                        toUInt128(id) as id_uint,
                        metric_name,
                        target_id,
                        toUInt128(target_id) as target_id_uint,
                        value,
                        tags,
                        snapshot_hash
                    FROM FloatMetricFeedback
                ),
                targets AS (
                    SELECT
                        uint_to_uuid(id_uint) as target_id,
                        function_name,
                        variant_name
                    FROM InferenceById
                    WHERE id_uint IN (SELECT target_id_uint FROM float_feedback)
                    UNION ALL
                    SELECT
                        uint_to_uuid(episode_id_uint) as target_id,
                        function_name,
                        unique_variants[1] as variant_name
                    FROM (
                        SELECT
                            episode_id_uint,
                            function_name,
                            groupUniqArray(variant_name) as unique_variants
                        FROM InferenceByEpisodeId
                        WHERE episode_id_uint IN (SELECT target_id_uint FROM float_feedback)
                        GROUP BY (episode_id_uint, function_name)
                    )
                    WHERE length(unique_variants) = 1
                )
            SELECT
                t.function_name as function_name,
                t.variant_name as variant_name,
                f.metric_name as metric_name,
                f.id_uint as id_uint,
                f.target_id_uint as target_id_uint,
                f.value as value,
                f.tags as feedback_tags,
                f.snapshot_hash as snapshot_hash
            FROM float_feedback f
            JOIN targets t ON f.target_id = t.target_id
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "
            ALTER TABLE BooleanMetricFeedbackByVariantView MODIFY QUERY
            WITH
                boolean_feedback AS (
                    SELECT
                        toUInt128(id) as id_uint,
                        metric_name,
                        target_id,
                        toUInt128(target_id) as target_id_uint,
                        value,
                        tags,
                        snapshot_hash
                    FROM BooleanMetricFeedback
                ),
                targets AS (
                    SELECT
                        uint_to_uuid(id_uint) as target_id,
                        function_name,
                        variant_name
                    FROM InferenceById
                    WHERE id_uint IN (SELECT target_id_uint FROM boolean_feedback)
                    UNION ALL
                    SELECT
                        uint_to_uuid(episode_id_uint) as target_id,
                        function_name,
                        unique_variants[1] as variant_name
                    FROM (
                        SELECT
                            episode_id_uint,
                            function_name,
                            groupUniqArray(variant_name) as unique_variants
                        FROM InferenceByEpisodeId
                        WHERE episode_id_uint IN (SELECT target_id_uint FROM boolean_feedback)
                        GROUP BY (episode_id_uint, function_name)
                    )
                    WHERE length(unique_variants) = 1
                )
            SELECT
                t.function_name as function_name,
                t.variant_name as variant_name,
                f.metric_name as metric_name,
                f.id_uint as id_uint,
                f.target_id_uint as target_id_uint,
                f.value as value,
                f.tags as feedback_tags,
                f.snapshot_hash as snapshot_hash
            FROM boolean_feedback f
            JOIN targets t ON f.target_id = t.target_id
        ";
        self.clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let mut instructions = String::from("DROP TABLE ConfigSnapshot;\n");

        let tables = [
            "BatchModelInference",
            "BatchRequest",
            "BooleanMetricFeedback",
            "BooleanMetricFeedbackByTargetId",
            "BooleanMetricFeedbackByVariant",
            "ChatInference",
            "ChatInferenceDatapoint",
            "CommentFeedback",
            "CommentFeedbackByTargetId",
            "DemonstrationFeedback",
            "DemonstrationFeedbackByInferenceId",
            "DynamicEvaluationRun",
            "DynamicEvaluationRunByProjectName",
            "DynamicEvaluationRunEpisode",
            "FeedbackTag",
            "FloatMetricFeedback",
            "FloatMetricFeedbackByTargetId",
            "InferenceByEpisodeId",
            "InferenceById",
            "InferenceTag",
            "JsonInference",
            "JsonInferenceDatapoint",
            "ModelInference",
            "TagInference",
        ];

        for table in tables {
            instructions.push_str(&format!("ALTER TABLE {table} DROP COLUMN snapshot_hash;\n"));
        }

        // Rollback materialized view modifications
        instructions.push_str(
            "
-- Rollback Group 1: Feedback indexing views
ALTER TABLE BooleanMetricFeedbackByTargetIdView MODIFY QUERY
SELECT
    id,
    target_id,
    metric_name,
    value,
    tags
FROM BooleanMetricFeedback;

ALTER TABLE FloatMetricFeedbackByTargetIdView MODIFY QUERY
SELECT
    id,
    target_id,
    metric_name,
    value,
    tags
FROM FloatMetricFeedback;

ALTER TABLE CommentFeedbackByTargetIdView MODIFY QUERY
SELECT
    id,
    target_id,
    target_type,
    value,
    tags
FROM CommentFeedback;

ALTER TABLE DemonstrationFeedbackByInferenceIdView MODIFY QUERY
SELECT
    id,
    inference_id,
    value,
    tags
FROM DemonstrationFeedback;

-- Rollback Group 2: Inference indexing views
ALTER TABLE ChatInferenceByIdView MODIFY QUERY
SELECT
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    episode_id,
    'chat' AS function_type
FROM ChatInference;

ALTER TABLE JsonInferenceByIdView MODIFY QUERY
SELECT
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    episode_id,
    'json' AS function_type
FROM JsonInference;

ALTER TABLE ChatInferenceByEpisodeIdView MODIFY QUERY
SELECT
    toUInt128(episode_id) as episode_id_uint,
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    'chat' as function_type
FROM ChatInference;

ALTER TABLE JsonInferenceByEpisodeIdView MODIFY QUERY
SELECT
    toUInt128(episode_id) as episode_id_uint,
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    'json' as function_type
FROM JsonInference;

-- Rollback Group 3: Tag extraction views
ALTER TABLE TagChatInferenceView MODIFY QUERY
SELECT
    function_name,
    variant_name,
    episode_id,
    id as inference_id,
    'chat' as function_type,
    key,
    tags[key] as value
FROM ChatInference
ARRAY JOIN mapKeys(tags) as key;

ALTER TABLE TagJsonInferenceView MODIFY QUERY
SELECT
    function_name,
    variant_name,
    episode_id,
    id as inference_id,
    'json' as function_type,
    key,
    tags[key] as value
FROM JsonInference
ARRAY JOIN mapKeys(tags) as key;

-- Rollback Group 4: Feedback by variant views
ALTER TABLE FloatMetricFeedbackByVariantView MODIFY QUERY
WITH
    float_feedback AS (
        SELECT
            toUInt128(id) as id_uint,
            metric_name,
            target_id,
            toUInt128(target_id) as target_id_uint,
            value,
            tags
        FROM FloatMetricFeedback
    ),
    targets AS (
        SELECT
            uint_to_uuid(id_uint) as target_id,
            function_name,
            variant_name
        FROM InferenceById
        WHERE id_uint IN (SELECT target_id_uint FROM float_feedback)
        UNION ALL
        SELECT
            uint_to_uuid(episode_id_uint) as target_id,
            function_name,
            unique_variants[1] as variant_name
        FROM (
            SELECT
                episode_id_uint,
                function_name,
                groupUniqArray(variant_name) as unique_variants
            FROM InferenceByEpisodeId
            WHERE episode_id_uint IN (SELECT target_id_uint FROM float_feedback)
            GROUP BY (episode_id_uint, function_name)
        )
        WHERE length(unique_variants) = 1
    )
SELECT
    t.function_name as function_name,
    t.variant_name as variant_name,
    f.metric_name as metric_name,
    f.id_uint as id_uint,
    f.target_id_uint as target_id_uint,
    f.value as value,
    f.tags as feedback_tags
FROM float_feedback f
JOIN targets t ON f.target_id = t.target_id;

ALTER TABLE BooleanMetricFeedbackByVariantView MODIFY QUERY
WITH
    boolean_feedback AS (
        SELECT
            toUInt128(id) as id_uint,
            metric_name,
            target_id,
            toUInt128(target_id) as target_id_uint,
            value,
            tags
        FROM BooleanMetricFeedback
    ),
    targets AS (
        SELECT
            uint_to_uuid(id_uint) as target_id,
            function_name,
            variant_name
        FROM InferenceById
        WHERE id_uint IN (SELECT target_id_uint FROM boolean_feedback)
        UNION ALL
        SELECT
            uint_to_uuid(episode_id_uint) as target_id,
            function_name,
            unique_variants[1] as variant_name
        FROM (
            SELECT
                episode_id_uint,
                function_name,
                groupUniqArray(variant_name) as unique_variants
            FROM InferenceByEpisodeId
            WHERE episode_id_uint IN (SELECT target_id_uint FROM boolean_feedback)
            GROUP BY (episode_id_uint, function_name)
        )
        WHERE length(unique_variants) = 1
    )
SELECT
    t.function_name as function_name,
    t.variant_name as variant_name,
    f.metric_name as metric_name,
    f.id_uint as id_uint,
    f.target_id_uint as target_id_uint,
    f.value as value,
    f.tags as feedback_tags
FROM boolean_feedback f
JOIN targets t ON f.target_id = t.target_id;
",
        );

        instructions
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(!self.should_apply().await?)
    }
}
