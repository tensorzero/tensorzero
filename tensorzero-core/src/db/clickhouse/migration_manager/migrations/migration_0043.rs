use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::migration_manager::migrations::{
    check_column_exists, check_table_exists,
};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;
use std::time::Duration;
use tokio::time::sleep;

const MIGRATION_ID: &str = "0043";

/// Tables that track snapshot_hash for configuration versioning
const SNAPSHOT_TRACKED_TABLES: &[&str] = &[
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
    "FloatMetricFeedbackByVariant",
    "InferenceByEpisodeId",
    "InferenceById",
    "InferenceTag",
    "JsonInference",
    "JsonInferenceDatapoint",
    "ModelInference",
    "TagInference",
];

/// Materialized views that are updated to propagate snapshot_hash
const SNAPSHOT_MATERIALIZED_VIEWS: &[&str] = &[
    "BooleanMetricFeedbackByTargetIdView",
    "FloatMetricFeedbackByTargetIdView",
    "CommentFeedbackByTargetIdView",
    "DemonstrationFeedbackByInferenceIdView",
    "ChatInferenceByIdView",
    "JsonInferenceByIdView",
    "ChatInferenceByEpisodeIdView",
    "JsonInferenceByEpisodeIdView",
    "TagChatInferenceView",
    "TagJsonInferenceView",
    "ChatInferenceTagView",
    "JsonInferenceTagView",
    "FloatMetricFeedbackByVariantView",
    "BooleanMetricFeedbackByVariantView",
    "BooleanMetricFeedbackTagView",
    "CommentFeedbackTagView",
    "DemonstrationFeedbackTagView",
    "FloatMetricFeedbackTagView",
];

/// This migration sets up the ClickHouse data structures required
/// for snapshotting:
/// * the `ConfigSnapshot table`
/// * adds a `snapshot_hash` column to every table in `SNAPSHOT_TRACKED_TABLES` above
/// * for all tables above which are populated by materialized views, we alter the
///   materialized view as needed to ensure that the `snapshot_hash` propagates.
/// This should allow us to write snapshots + snapshot hashes everywhere, and have
/// much better information about data provenance.
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
        if !check_table_exists(self.clickhouse, "ConfigSnapshot", MIGRATION_ID).await.unwrap() {
            return Ok(true);
        }

        // Check if any of the tables is missing the snapshot_hash column
        for table in SNAPSHOT_TRACKED_TABLES {
            if !check_column_exists(self.clickhouse, table, "snapshot_hash", MIGRATION_ID).await.unwrap() {
                return Ok(true);
            }
        }

        // Check if any materialized view hasn't been updated to include snapshot_hash
        for view in SNAPSHOT_MATERIALIZED_VIEWS {
            let query = format!("SHOW CREATE TABLE {view}");
            let result = self
                .clickhouse
                .run_query_synchronous_no_params(query)
                .await.unwrap();
            if !result.response.contains("snapshot_hash") {
                return Ok(true);
            }
        }

        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        // Create ConfigSnapshot table
        let create_table_query = format!(
            "
            CREATE TABLE IF NOT EXISTS ConfigSnapshot{on_cluster_name} (
                config String,
                extra_templates Map(String, String),
                hash UInt256,
                tensorzero_version String,
                created_at DateTime64(6) DEFAULT now(),
                last_used DateTime64(6) DEFAULT now()
            ) ENGINE = ReplacingMergeTree(last_used)
            ORDER BY hash
            SETTINGS index_granularity = 256
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(create_table_query)
            .await.unwrap();

        // Add snapshot_hash column to existing tables
        for table in SNAPSHOT_TRACKED_TABLES {
            let query = format!(
                "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS snapshot_hash Nullable(UInt256)"
            );
            self.clickhouse
                .run_query_synchronous_no_params(query)
                .await.unwrap();
        }

        // Make sure ClickHouse is aware of all the new columns before migrating the MVs
        // This caused issues on CI but never locally
        sleep(Duration::from_secs(1)).await;

        // Update materialized views to propagate snapshot_hash from source tables
        // Group 1: Feedback indexing views
        let query = format!(
            "
            ALTER TABLE BooleanMetricFeedbackByTargetIdView{on_cluster_name} MODIFY QUERY
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                snapshot_hash
            FROM BooleanMetricFeedback
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE FloatMetricFeedbackByTargetIdView{on_cluster_name} MODIFY QUERY
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                snapshot_hash
            FROM FloatMetricFeedback
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE CommentFeedbackByTargetIdView{on_cluster_name} MODIFY QUERY
            SELECT
                id,
                target_id,
                target_type,
                value,
                tags,
                snapshot_hash
            FROM CommentFeedback
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE DemonstrationFeedbackByInferenceIdView{on_cluster_name} MODIFY QUERY
            SELECT
                id,
                inference_id,
                value,
                tags,
                snapshot_hash
            FROM DemonstrationFeedback
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        // Group 2: Inference indexing views
        let query = format!(
            "
            ALTER TABLE ChatInferenceByIdView{on_cluster_name} MODIFY QUERY
            SELECT
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                episode_id,
                'chat' AS function_type,
                snapshot_hash
            FROM ChatInference
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE JsonInferenceByIdView{on_cluster_name} MODIFY QUERY
            SELECT
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                episode_id,
                'json' AS function_type,
                snapshot_hash
            FROM JsonInference
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE ChatInferenceByEpisodeIdView{on_cluster_name} MODIFY QUERY
            SELECT
                toUInt128(episode_id) as episode_id_uint,
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                'chat' as function_type,
                snapshot_hash
            FROM ChatInference
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE JsonInferenceByEpisodeIdView{on_cluster_name} MODIFY QUERY
            SELECT
                toUInt128(episode_id) as episode_id_uint,
                toUInt128(id) as id_uint,
                function_name,
                variant_name,
                'json' as function_type,
                snapshot_hash
            FROM JsonInference
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        // Group 3: Tag extraction views
        let query = format!(
            "
            ALTER TABLE TagChatInferenceView{on_cluster_name} MODIFY QUERY
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
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE TagJsonInferenceView{on_cluster_name} MODIFY QUERY
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
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE ChatInferenceTagView{on_cluster_name} MODIFY QUERY
            SELECT
                function_name,
                key,
                tags[key] as value,
                id as inference_id,
                snapshot_hash
            FROM ChatInference
            ARRAY JOIN mapKeys(tags) as key
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE JsonInferenceTagView{on_cluster_name} MODIFY QUERY
            SELECT
                function_name,
                key,
                tags[key] as value,
                id as inference_id,
                snapshot_hash
            FROM JsonInference
            ARRAY JOIN mapKeys(tags) as key
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        // Group 4: Feedback by variant views (with JOINs)
        let query = format!(
            "
            ALTER TABLE FloatMetricFeedbackByVariantView{on_cluster_name} MODIFY QUERY
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
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE BooleanMetricFeedbackByVariantView{on_cluster_name} MODIFY QUERY
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
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        // Group 5: Feedback tag views
        let query = format!(
            "
            ALTER TABLE BooleanMetricFeedbackTagView{on_cluster_name} MODIFY QUERY
            SELECT
                metric_name,
                key,
                tags[key] as value,
                id as feedback_id,
                snapshot_hash
            FROM BooleanMetricFeedback
            ARRAY JOIN mapKeys(tags) as key
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE CommentFeedbackTagView{on_cluster_name} MODIFY QUERY
            SELECT
                'comment' as metric_name,
                key,
                tags[key] as value,
                id as feedback_id,
                snapshot_hash
            FROM CommentFeedback
            ARRAY JOIN mapKeys(tags) as key
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE DemonstrationFeedbackTagView{on_cluster_name} MODIFY QUERY
            SELECT
                'demonstration' as metric_name,
                key,
                tags[key] as value,
                id as feedback_id,
                snapshot_hash
            FROM DemonstrationFeedback
            ARRAY JOIN mapKeys(tags) as key
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        let query = format!(
            "
            ALTER TABLE FloatMetricFeedbackTagView{on_cluster_name} MODIFY QUERY
            SELECT
                metric_name,
                key,
                tags[key] as value,
                id as feedback_id,
                snapshot_hash
            FROM FloatMetricFeedback
            ARRAY JOIN mapKeys(tags) as key
        "
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await.unwrap();

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let mut instructions = String::new();

        // Rollback materialized view modifications
        instructions.push_str(&format!(
            "
-- Rollback Group 1: Feedback indexing views
ALTER TABLE BooleanMetricFeedbackByTargetIdView{on_cluster_name} MODIFY QUERY
SELECT
    id,
    target_id,
    metric_name,
    value,
    tags
FROM BooleanMetricFeedback;

ALTER TABLE FloatMetricFeedbackByTargetIdView{on_cluster_name} MODIFY QUERY
SELECT
    id,
    target_id,
    metric_name,
    value,
    tags
FROM FloatMetricFeedback;

ALTER TABLE CommentFeedbackByTargetIdView{on_cluster_name} MODIFY QUERY
SELECT
    id,
    target_id,
    target_type,
    value,
    tags
FROM CommentFeedback;

ALTER TABLE DemonstrationFeedbackByInferenceIdView{on_cluster_name} MODIFY QUERY
SELECT
    id,
    inference_id,
    value,
    tags
FROM DemonstrationFeedback;

-- Rollback Group 2: Inference indexing views
ALTER TABLE ChatInferenceByIdView{on_cluster_name} MODIFY QUERY
SELECT
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    episode_id,
    'chat' AS function_type
FROM ChatInference;

ALTER TABLE JsonInferenceByIdView{on_cluster_name} MODIFY QUERY
SELECT
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    episode_id,
    'json' AS function_type
FROM JsonInference;

ALTER TABLE ChatInferenceByEpisodeIdView{on_cluster_name} MODIFY QUERY
SELECT
    toUInt128(episode_id) as episode_id_uint,
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    'chat' as function_type
FROM ChatInference;

ALTER TABLE JsonInferenceByEpisodeIdView{on_cluster_name} MODIFY QUERY
SELECT
    toUInt128(episode_id) as episode_id_uint,
    toUInt128(id) as id_uint,
    function_name,
    variant_name,
    'json' as function_type
FROM JsonInference;

-- Rollback Group 3: Tag extraction views
ALTER TABLE TagChatInferenceView{on_cluster_name} MODIFY QUERY
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

ALTER TABLE TagJsonInferenceView{on_cluster_name} MODIFY QUERY
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

ALTER TABLE ChatInferenceTagView{on_cluster_name} MODIFY QUERY
SELECT
    function_name,
    key,
    tags[key] as value,
    id as inference_id
FROM ChatInference
ARRAY JOIN mapKeys(tags) as key;

ALTER TABLE JsonInferenceTagView{on_cluster_name} MODIFY QUERY
SELECT
    function_name,
    key,
    tags[key] as value,
    id as inference_id
FROM JsonInference
ARRAY JOIN mapKeys(tags) as key;

-- Rollback Group 4: Feedback by variant views
ALTER TABLE FloatMetricFeedbackByVariantView{on_cluster_name} MODIFY QUERY
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

ALTER TABLE BooleanMetricFeedbackByVariantView{on_cluster_name} MODIFY QUERY
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

-- Rollback Group 5: Feedback tag views
ALTER TABLE BooleanMetricFeedbackTagView{on_cluster_name} MODIFY QUERY
SELECT
    metric_name,
    key,
    tags[key] as value,
    id as feedback_id
FROM BooleanMetricFeedback
ARRAY JOIN mapKeys(tags) as key;

ALTER TABLE CommentFeedbackTagView{on_cluster_name} MODIFY QUERY
SELECT
    'comment' as metric_name,
    key,
    tags[key] as value,
    id as feedback_id
FROM CommentFeedback
ARRAY JOIN mapKeys(tags) as key;

ALTER TABLE DemonstrationFeedbackTagView{on_cluster_name} MODIFY QUERY
SELECT
    'demonstration' as metric_name,
    key,
    tags[key] as value,
    id as feedback_id
FROM DemonstrationFeedback
ARRAY JOIN mapKeys(tags) as key;

ALTER TABLE FloatMetricFeedbackTagView{on_cluster_name} MODIFY QUERY
SELECT
    metric_name,
    key,
    tags[key] as value,
    id as feedback_id
FROM FloatMetricFeedback
ARRAY JOIN mapKeys(tags) as key;
"
        ));
        instructions.push_str(&format!("DROP TABLE ConfigSnapshot{on_cluster_name};\n"));

        for table in SNAPSHOT_TRACKED_TABLES {
            instructions.push_str(&format!("ALTER TABLE {table} DROP COLUMN snapshot_hash;\n"));
        }

        instructions
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(!self.should_apply().await.unwrap())
    }
}
