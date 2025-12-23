# Existing ClickHouse Tables

This document catalogs all existing ClickHouse tables and materialized views in the TensorZero database, for reference during the migration to PostgreSQL.

## Table of Contents

- [Primary Tables](#primary-tables)
  - [Inference Tables](#inference-tables)
  - [Feedback Tables](#feedback-tables)
  - [Batch Tables](#batch-tables)
  - [Datapoint Tables](#datapoint-tables)
  - [Evaluation Tables](#evaluation-tables)
  - [Index/Lookup Tables](#indexlookup-tables)
  - [Statistics Tables](#statistics-tables)
  - [Configuration Tables](#configuration-tables)
  - [Miscellaneous Tables](#miscellaneous-tables)
- [Materialized Views](#materialized-views)

---

## Primary Tables

### Inference Tables

#### ChatInference

Main table for chat inference requests.

```sql
CREATE TABLE ChatInference (
    `id` UUID,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `episode_id` UUID,
    `input` String,
    `output` String,
    `tool_params` String,
    `inference_params` String,
    `processing_time_ms` Nullable(UInt32),
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `tags` Map(String, String) DEFAULT map(),
    `extra_body` Nullable(String),
    `ttft_ms` Nullable(UInt32),
    `dynamic_tools` Array(String),
    `dynamic_provider_tools` Array(String),
    `allowed_tools` Nullable(String),
    `tool_choice` Nullable(String),
    `parallel_tool_calls` Nullable(Bool),
    `snapshot_hash` Nullable(UInt256),
    INDEX inference_id_index id TYPE bloom_filter GRANULARITY 1
) ENGINE = MergeTree
ORDER BY (function_name, variant_name, episode_id)
SETTINGS index_granularity = 8192
```

#### JsonInference

Main table for JSON inference requests.

```sql
CREATE TABLE JsonInference (
    `id` UUID,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `episode_id` UUID,
    `input` String,
    `output` String,
    `output_schema` String,
    `inference_params` String,
    `processing_time_ms` Nullable(UInt32),
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `tags` Map(String, String) DEFAULT map(),
    `extra_body` Nullable(String),
    `auxiliary_content` String,
    `ttft_ms` Nullable(UInt32),
    `snapshot_hash` Nullable(UInt256),
    INDEX inference_id_index id TYPE bloom_filter GRANULARITY 1
) ENGINE = MergeTree
ORDER BY (function_name, variant_name, episode_id)
SETTINGS index_granularity = 8192
```

#### ModelInference

Table for model-level inference details (raw request/response).

```sql
CREATE TABLE ModelInference (
    `id` UUID,
    `inference_id` UUID,
    `raw_request` String,
    `raw_response` String,
    `model_name` LowCardinality(String),
    `model_provider_name` LowCardinality(String),
    `input_tokens` Nullable(UInt32),
    `output_tokens` Nullable(UInt32),
    `response_time_ms` Nullable(UInt32),
    `ttft_ms` Nullable(UInt32),
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `system` Nullable(String),
    `input_messages` String,
    `output` String,
    `cached` Bool DEFAULT false,
    `finish_reason` Nullable(Enum8('stop' = 1, 'length' = 2, 'tool_call' = 3, 'content_filter' = 4, 'unknown' = 5, 'stop_sequence' = 6)),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY inference_id
SETTINGS index_granularity = 8192
```

#### ModelInferenceCache

Cache for model inference responses.

```sql
CREATE TABLE ModelInferenceCache (
    `short_cache_key` UInt64,
    `long_cache_key` FixedString(64),
    `timestamp` DateTime DEFAULT now(),
    `output` String,
    `raw_request` String,
    `raw_response` String,
    `is_deleted` Bool DEFAULT false,
    `input_tokens` Nullable(UInt32),
    `output_tokens` Nullable(UInt32),
    `finish_reason` Nullable(Enum8('stop' = 1, 'length' = 2, 'tool_call' = 3, 'content_filter' = 4, 'unknown' = 5, 'stop_sequence' = 6)),
    INDEX idx_long_cache_key long_cache_key TYPE bloom_filter GRANULARITY 100
) ENGINE = ReplacingMergeTree(timestamp, is_deleted)
PARTITION BY toYYYYMM(timestamp)
PRIMARY KEY short_cache_key
ORDER BY (short_cache_key, long_cache_key)
SETTINGS index_granularity = 256
```

---

### Feedback Tables

#### BooleanMetricFeedback

Feedback with boolean values.

```sql
CREATE TABLE BooleanMetricFeedback (
    `id` UUID,
    `target_id` UUID,
    `metric_name` LowCardinality(String),
    `value` Bool,
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `tags` Map(String, String) DEFAULT map(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (metric_name, target_id)
SETTINGS index_granularity = 8192
```

#### BooleanMetricFeedbackByTargetId

Lookup table for boolean feedback by target.

```sql
CREATE TABLE BooleanMetricFeedbackByTargetId (
    `id` UUID,
    `target_id` UUID,
    `metric_name` LowCardinality(String),
    `value` Bool,
    `tags` Map(String, String),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY target_id
SETTINGS index_granularity = 8192
```

#### BooleanMetricFeedbackByVariant

Boolean feedback organized by variant for analytics.

```sql
CREATE TABLE BooleanMetricFeedbackByVariant (
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `metric_name` LowCardinality(String),
    `id_uint` UInt128,
    `target_id_uint` UInt128,
    `value` Bool,
    `feedback_tags` Map(String, String),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (function_name, metric_name, variant_name, id_uint)
SETTINGS index_granularity = 8192
```

#### FloatMetricFeedback

Feedback with float values.

```sql
CREATE TABLE FloatMetricFeedback (
    `id` UUID,
    `target_id` UUID,
    `metric_name` LowCardinality(String),
    `value` Float32,
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `tags` Map(String, String) DEFAULT map(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (metric_name, target_id)
SETTINGS index_granularity = 8192
```

#### FloatMetricFeedbackByTargetId

Lookup table for float feedback by target.

```sql
CREATE TABLE FloatMetricFeedbackByTargetId (
    `id` UUID,
    `target_id` UUID,
    `metric_name` LowCardinality(String),
    `value` Float32,
    `tags` Map(String, String),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY target_id
SETTINGS index_granularity = 8192
```

#### FloatMetricFeedbackByVariant

Float feedback organized by variant for analytics.

```sql
CREATE TABLE FloatMetricFeedbackByVariant (
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `metric_name` LowCardinality(String),
    `id_uint` UInt128,
    `target_id_uint` UInt128,
    `value` Float32,
    `feedback_tags` Map(String, String),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (function_name, metric_name, variant_name, id_uint)
SETTINGS index_granularity = 8192
```

#### CommentFeedback

Free-form comment feedback.

```sql
CREATE TABLE CommentFeedback (
    `id` UUID,
    `target_id` UUID,
    `target_type` Enum8('inference' = 1, 'episode' = 2),
    `value` String,
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `tags` Map(String, String) DEFAULT map(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY target_id
SETTINGS index_granularity = 8192
```

#### CommentFeedbackByTargetId

Lookup table for comment feedback by target.

```sql
CREATE TABLE CommentFeedbackByTargetId (
    `id` UUID,
    `target_id` UUID,
    `target_type` Enum8('inference' = 1, 'episode' = 2),
    `value` String,
    `tags` Map(String, String),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY target_id
SETTINGS index_granularity = 8192
```

#### DemonstrationFeedback

Demonstration/correction feedback.

```sql
CREATE TABLE DemonstrationFeedback (
    `id` UUID,
    `inference_id` UUID,
    `value` String,
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `tags` Map(String, String) DEFAULT map(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY inference_id
SETTINGS index_granularity = 8192
```

#### DemonstrationFeedbackByInferenceId

Lookup table for demonstration feedback by inference.

```sql
CREATE TABLE DemonstrationFeedbackByInferenceId (
    `id` UUID,
    `inference_id` UUID,
    `value` String,
    `tags` Map(String, String),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY inference_id
SETTINGS index_granularity = 8192
```

#### FeedbackTag

Tags extracted from feedback for querying.

```sql
CREATE TABLE FeedbackTag (
    `metric_name` LowCardinality(String),
    `key` String,
    `value` String,
    `feedback_id` UUID,
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (metric_name, key, value)
SETTINGS index_granularity = 8192
```

---

### Batch Tables

#### BatchModelInference

Model inference requests for batch processing.

```sql
CREATE TABLE BatchModelInference (
    `inference_id` UUID,
    `batch_id` UUID,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `episode_id` UUID,
    `input` String,
    `input_messages` String,
    `system` Nullable(String),
    `tool_params` Nullable(String),
    `inference_params` String,
    `raw_request` String,
    `model_name` LowCardinality(String),
    `model_provider_name` LowCardinality(String),
    `output_schema` Nullable(String),
    `tags` Map(String, String) DEFAULT map(),
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(inference_id),
    `dynamic_tools` Array(String),
    `dynamic_provider_tools` Array(String),
    `allowed_tools` Nullable(String),
    `tool_choice` Nullable(String),
    `parallel_tool_calls` Nullable(Bool),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (batch_id, inference_id)
SETTINGS index_granularity = 8192
```

#### BatchRequest

Batch request metadata and status.

```sql
CREATE TABLE BatchRequest (
    `batch_id` UUID,
    `id` UUID,
    `batch_params` String,
    `model_name` LowCardinality(String),
    `model_provider_name` LowCardinality(String),
    `status` Enum8('pending' = 1, 'completed' = 2, 'failed' = 3),
    `errors` Array(String),
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id),
    `raw_request` String,
    `raw_response` String,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (batch_id, id)
SETTINGS index_granularity = 8192
```

#### BatchIdByInferenceId

Lookup table for batch ID by inference ID.

```sql
CREATE TABLE BatchIdByInferenceId (
    `inference_id` UUID,
    `batch_id` UUID
) ENGINE = MergeTree
ORDER BY inference_id
SETTINGS index_granularity = 8192
```

---

### Datapoint Tables

#### ChatInferenceDatapoint

Datapoints for chat inference datasets.

```sql
CREATE TABLE ChatInferenceDatapoint (
    `dataset_name` LowCardinality(String),
    `function_name` LowCardinality(String),
    `id` UUID,
    `episode_id` Nullable(UUID),
    `input` String,
    `output` Nullable(String),
    `tool_params` String,
    `tags` Map(String, String),
    `auxiliary` String,
    `is_deleted` Bool DEFAULT false,
    `updated_at` DateTime64(6, 'UTC') DEFAULT now64(),
    `staled_at` Nullable(DateTime64(6, 'UTC')),
    `source_inference_id` Nullable(UUID),
    `is_custom` Bool DEFAULT false,
    `name` Nullable(String),
    `dynamic_tools` Array(String),
    `dynamic_provider_tools` Array(String),
    `allowed_tools` Nullable(String),
    `tool_choice` Nullable(String),
    `parallel_tool_calls` Nullable(Bool),
    `snapshot_hash` Nullable(UInt256),
    INDEX id_index id TYPE bloom_filter GRANULARITY 1
) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
ORDER BY (dataset_name, function_name, id)
SETTINGS index_granularity = 8192
```

#### JsonInferenceDatapoint

Datapoints for JSON inference datasets.

```sql
CREATE TABLE JsonInferenceDatapoint (
    `dataset_name` LowCardinality(String),
    `function_name` LowCardinality(String),
    `id` UUID,
    `episode_id` Nullable(UUID),
    `input` String,
    `output` Nullable(String),
    `output_schema` String,
    `tags` Map(String, String),
    `auxiliary` String,
    `is_deleted` Bool DEFAULT false,
    `updated_at` DateTime64(6, 'UTC') DEFAULT now64(),
    `staled_at` Nullable(DateTime64(6, 'UTC')),
    `source_inference_id` Nullable(UUID),
    `is_custom` Bool DEFAULT false,
    `name` Nullable(String),
    `snapshot_hash` Nullable(UInt256),
    INDEX id_index id TYPE bloom_filter GRANULARITY 1
) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
ORDER BY (dataset_name, function_name, id)
SETTINGS index_granularity = 8192
```

#### DynamicInContextLearningExample

Examples for dynamic in-context learning.

```sql
CREATE TABLE DynamicInContextLearningExample (
    `id` UUID,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `namespace` String,
    `input` String,
    `output` String,
    `embedding` Array(Float32),
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id)
) ENGINE = MergeTree
ORDER BY (function_name, variant_name, namespace)
SETTINGS index_granularity = 8192
```

---

### Evaluation Tables

#### DynamicEvaluationRun

Runs of dynamic evaluations.

```sql
CREATE TABLE DynamicEvaluationRun (
    `run_id_uint` UInt128,
    `variant_pins` Map(String, String),
    `tags` Map(String, String),
    `project_name` Nullable(String),
    `run_display_name` Nullable(String),
    `is_deleted` Bool DEFAULT false,
    `updated_at` DateTime64(6, 'UTC') DEFAULT now(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
ORDER BY run_id_uint
SETTINGS index_granularity = 8192
```

#### DynamicEvaluationRunByProjectName

Evaluation runs indexed by project name.

```sql
CREATE TABLE DynamicEvaluationRunByProjectName (
    `run_id_uint` UInt128,
    `variant_pins` Map(String, String),
    `tags` Map(String, String),
    `project_name` String,
    `run_display_name` Nullable(String),
    `is_deleted` Bool DEFAULT false,
    `updated_at` DateTime64(6, 'UTC') DEFAULT now(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
ORDER BY (project_name, run_id_uint)
SETTINGS index_granularity = 8192
```

#### DynamicEvaluationRunEpisode

Episodes within dynamic evaluation runs.

```sql
CREATE TABLE DynamicEvaluationRunEpisode (
    `run_id` UUID,
    `episode_id_uint` UInt128,
    `variant_pins` Map(String, String),
    `datapoint_name` Nullable(String),
    `tags` Map(String, String),
    `is_deleted` Bool DEFAULT false,
    `updated_at` DateTime64(6, 'UTC') DEFAULT now(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
ORDER BY episode_id_uint
SETTINGS index_granularity = 8192
```

#### DynamicEvaluationRunEpisodeByRunId

Evaluation episodes indexed by run ID.

```sql
CREATE TABLE DynamicEvaluationRunEpisodeByRunId (
    `run_id_uint` UInt128,
    `episode_id_uint` UInt128,
    `variant_pins` Map(String, String),
    `tags` Map(String, String),
    `datapoint_name` Nullable(String),
    `is_deleted` Bool DEFAULT false,
    `updated_at` DateTime64(6, 'UTC') DEFAULT now(),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
ORDER BY (run_id_uint, episode_id_uint)
SETTINGS index_granularity = 8192
```

#### StaticEvaluationHumanFeedback

Human feedback for static evaluations.

```sql
CREATE TABLE StaticEvaluationHumanFeedback (
    `metric_name` LowCardinality(String),
    `datapoint_id` UUID,
    `output` String,
    `value` String,
    `feedback_id` UUID,
    `evaluator_inference_id` UUID,
    `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(feedback_id)
) ENGINE = MergeTree
ORDER BY (metric_name, datapoint_id, output)
SETTINGS index_granularity = 256
```

---

### Index/Lookup Tables

#### InferenceById

Lookup table for inferences by ID.

```sql
CREATE TABLE InferenceById (
    `id_uint` UInt128,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `episode_id` UUID,
    `function_type` Enum8('chat' = 1, 'json' = 2),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = ReplacingMergeTree(id_uint)
ORDER BY id_uint
SETTINGS index_granularity = 8192
```

#### InferenceByEpisodeId

Lookup table for inferences by episode ID.

```sql
CREATE TABLE InferenceByEpisodeId (
    `episode_id_uint` UInt128,
    `id_uint` UInt128,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `function_type` Enum8('chat' = 1, 'json' = 2),
    `snapshot_hash` Nullable(UInt256)
) ENGINE = ReplacingMergeTree(id_uint)
ORDER BY (episode_id_uint, id_uint)
SETTINGS index_granularity = 8192
```

#### InferenceTag

Tags extracted from inferences for querying.

```sql
CREATE TABLE InferenceTag (
    `function_name` LowCardinality(String),
    `key` String,
    `value` String,
    `inference_id` UUID,
    `snapshot_hash` Nullable(UInt256)
) ENGINE = MergeTree
ORDER BY (function_name, key, value)
SETTINGS index_granularity = 8192
```

#### TagInference

Extended inference tag lookup with soft delete.

```sql
CREATE TABLE TagInference (
    `key` String,
    `value` String,
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `episode_id` UUID,
    `inference_id` UUID,
    `function_type` Enum8('chat' = 1, 'json' = 2),
    `is_deleted` Bool DEFAULT false,
    `updated_at` DateTime64(6, 'UTC') DEFAULT now64(),
    `snapshot_hash` Nullable(UInt256),
    INDEX inference_id_index inference_id TYPE bloom_filter GRANULARITY 1
) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
ORDER BY (key, value, inference_id)
SETTINGS index_granularity = 8192
```

#### EpisodeById

Aggregated episode information.

```sql
CREATE TABLE EpisodeById (
    `episode_id_uint` UInt128,
    `count` SimpleAggregateFunction(sum, UInt64),
    `inference_ids` AggregateFunction(groupArray, UUID),
    `min_inference_id_uint` SimpleAggregateFunction(min, UInt128),
    `max_inference_id_uint` SimpleAggregateFunction(max, UInt128)
) ENGINE = AggregatingMergeTree
ORDER BY episode_id_uint
SETTINGS index_granularity = 8192
```

---

### Statistics Tables

#### FeedbackByVariantStatistics

Aggregated feedback statistics by variant.

```sql
CREATE TABLE FeedbackByVariantStatistics (
    `function_name` LowCardinality(String),
    `variant_name` LowCardinality(String),
    `metric_name` LowCardinality(String),
    `minute` DateTime,
    `feedback_mean` AggregateFunction(avg, Float32),
    `feedback_variance` AggregateFunction(varSampStable, Float32),
    `count` SimpleAggregateFunction(sum, UInt64)
) ENGINE = AggregatingMergeTree
ORDER BY (function_name, metric_name, variant_name, minute)
SETTINGS index_granularity = 8192
```

#### ModelProviderStatistics

Aggregated model provider performance statistics.

```sql
CREATE TABLE ModelProviderStatistics (
    `model_name` LowCardinality(String),
    `model_provider_name` LowCardinality(String),
    `minute` DateTime,
    `response_time_ms_quantiles` AggregateFunction(quantilesTDigest(...), Nullable(UInt32)),
    `ttft_ms_quantiles` AggregateFunction(quantilesTDigest(...), Nullable(UInt32)),
    `total_input_tokens` AggregateFunction(sum, Nullable(UInt32)),
    `total_output_tokens` AggregateFunction(sum, Nullable(UInt32)),
    `count` AggregateFunction(count, UInt32)
) ENGINE = AggregatingMergeTree
ORDER BY (model_name, model_provider_name, minute)
SETTINGS index_granularity = 8192
```

*Note: The quantile list includes 70 percentile values from 0.001 to 0.999.*

#### CumulativeUsage

Cumulative usage counters.

```sql
CREATE TABLE CumulativeUsage (
    `type` LowCardinality(String),
    `count` UInt64
) ENGINE = SummingMergeTree
ORDER BY type
SETTINGS index_granularity = 8192
```

---

### Configuration Tables

#### ConfigSnapshot

Configuration snapshots for versioning.

```sql
CREATE TABLE ConfigSnapshot (
    `config` String,
    `extra_templates` Map(String, String),
    `hash` UInt256,
    `tensorzero_version` String,
    `created_at` DateTime64(6) DEFAULT now(),
    `last_used` DateTime64(6) DEFAULT now(),
    `tags` Map(String, String) DEFAULT map()
) ENGINE = ReplacingMergeTree(last_used)
ORDER BY hash
SETTINGS index_granularity = 256
```

#### TensorZeroMigration

Migration tracking table.

```sql
CREATE TABLE TensorZeroMigration (
    `migration_id` UInt32,
    `migration_name` String,
    `gateway_version` String,
    `gateway_git_sha` String,
    `applied_at` DateTime64(6, 'UTC') DEFAULT now(),
    `execution_time_ms` UInt64,
    `extra_data` Nullable(String)
) ENGINE = MergeTree
PRIMARY KEY migration_id
ORDER BY migration_id
SETTINGS index_granularity = 8192
```

---

### Miscellaneous Tables

#### DeploymentID

Singleton table for deployment identification.

```sql
CREATE TABLE DeploymentID (
    `deployment_id` String,
    `dummy` UInt32 DEFAULT 0,
    `created_at` DateTime DEFAULT now(),
    `version_number` UInt32 DEFAULT 4294967295 - toUInt32(now())
) ENGINE = ReplacingMergeTree(version_number)
ORDER BY dummy
SETTINGS index_granularity = 8192
```

---

## Materialized Views

Materialized views in ClickHouse populate target tables automatically when source tables receive data.

### Inference Index Views

| View Name | Source Table | Target Table | Purpose |
|-----------|--------------|--------------|---------|
| ChatInferenceByIdView | ChatInference | InferenceById | Index chat inferences by ID |
| ChatInferenceByEpisodeIdView | ChatInference | InferenceByEpisodeId | Index chat inferences by episode |
| JsonInferenceByIdView | JsonInference | InferenceById | Index JSON inferences by ID |
| JsonInferenceByEpisodeIdView | JsonInference | InferenceByEpisodeId | Index JSON inferences by episode |
| EpisodeByIdChatView | ChatInference | EpisodeById | Aggregate episode info from chat |
| EpisodeByIdJsonView | JsonInference | EpisodeById | Aggregate episode info from JSON |

### Tag Extraction Views

| View Name | Source Table | Target Table | Purpose |
|-----------|--------------|--------------|---------|
| ChatInferenceTagView | ChatInference | InferenceTag | Extract tags from chat inferences |
| JsonInferenceTagView | JsonInference | InferenceTag | Extract tags from JSON inferences |
| TagChatInferenceView | ChatInference | TagInference | Extended tag extraction for chat |
| TagJsonInferenceView | JsonInference | TagInference | Extended tag extraction for JSON |
| BooleanMetricFeedbackTagView | BooleanMetricFeedback | FeedbackTag | Extract tags from boolean feedback |
| FloatMetricFeedbackTagView | FloatMetricFeedback | FeedbackTag | Extract tags from float feedback |
| CommentFeedbackTagView | CommentFeedback | FeedbackTag | Extract tags from comment feedback |
| DemonstrationFeedbackTagView | DemonstrationFeedback | FeedbackTag | Extract tags from demonstration feedback |

### Feedback Lookup Views

| View Name | Source Table | Target Table | Purpose |
|-----------|--------------|--------------|---------|
| BooleanMetricFeedbackByTargetIdView | BooleanMetricFeedback | BooleanMetricFeedbackByTargetId | Lookup by target |
| FloatMetricFeedbackByTargetIdView | FloatMetricFeedback | FloatMetricFeedbackByTargetId | Lookup by target |
| CommentFeedbackByTargetIdView | CommentFeedback | CommentFeedbackByTargetId | Lookup by target |
| DemonstrationFeedbackByInferenceIdView | DemonstrationFeedback | DemonstrationFeedbackByInferenceId | Lookup by inference |
| BooleanMetricFeedbackByVariantView | BooleanMetricFeedback | BooleanMetricFeedbackByVariant | Analytics by variant |
| FloatMetricFeedbackByVariantView | FloatMetricFeedback | FloatMetricFeedbackByVariant | Analytics by variant |

### Statistics Views

| View Name | Source Table | Target Table | Purpose |
|-----------|--------------|--------------|---------|
| BooleanMetricFeedbackByVariantStatisticsView | BooleanMetricFeedbackByVariant | FeedbackByVariantStatistics | Aggregate boolean feedback stats |
| FloatMetricFeedbackByVariantStatisticsView | FloatMetricFeedbackByVariant | FeedbackByVariantStatistics | Aggregate float feedback stats |
| ModelProviderStatisticsView | ModelInference | ModelProviderStatistics | Aggregate model performance |
| CumulativeUsageView | ModelInference | CumulativeUsage | Track cumulative token usage |

### Batch Views

| View Name | Source Table | Target Table | Purpose |
|-----------|--------------|--------------|---------|
| BatchIdByInferenceIdView | BatchModelInference | BatchIdByInferenceId | Lookup batch by inference ID |

### Evaluation Views

| View Name | Source Table | Target Table | Purpose |
|-----------|--------------|--------------|---------|
| DynamicEvaluationRunByProjectNameView | DynamicEvaluationRun | DynamicEvaluationRunByProjectName | Index runs by project |
| DynamicEvaluationRunEpisodeByRunIdView | DynamicEvaluationRunEpisode | DynamicEvaluationRunEpisodeByRunId | Index episodes by run |

---

## Key ClickHouse-Specific Features Used

1. **LowCardinality(String)**: Optimized storage for columns with few unique values (function names, variant names, metric names)

2. **UUIDv7ToDateTime()**: Extracts timestamp from UUIDv7 IDs - eliminates need for separate timestamp column storage

3. **Map(String, String)**: Native map type for tags without JSON parsing overhead

4. **Array(String)**: Native array type for dynamic tools lists

5. **ReplacingMergeTree**: Soft delete support via `is_deleted` flag and deduplication via version column

6. **AggregatingMergeTree**: Pre-aggregated statistics with merge-on-read for quantiles, averages, etc.

7. **SummingMergeTree**: Automatic summation for cumulative counters

8. **MaterializedViews**: Automatic population of secondary tables on insert

9. **UInt128 for UUID comparison**: Efficient sorting and comparison of UUIDs as integers

10. **Bloom filter indexes**: Fast existence checks for specific ID lookups

---

## Migration Considerations for PostgreSQL

1. **UUIDv7 timestamp extraction**: Need application-level or trigger-based timestamp extraction
2. **Map/Array columns**: Use JSONB for maps, native arrays or JSONB for arrays
3. **Materialized views**: Replace with triggers or application-level logic
4. **AggregateFunction columns**: Pre-compute aggregates via scheduled jobs or use pg_tdigest extension
5. **LowCardinality**: Not needed in PostgreSQL (handled by planner statistics)
6. **ReplacingMergeTree**: Use standard UPSERT patterns or soft delete triggers
7. **SummingMergeTree**: Use INSERT ... ON CONFLICT DO UPDATE with increment logic
