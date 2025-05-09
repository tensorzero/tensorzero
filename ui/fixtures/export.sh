#! /bin/bash
set -euxo pipefail

# NOTE: This script is mostly for documentation purposes but shows how to dump data from
# a running ClickHouse instance into parquet files.

clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(episode_id) as episode_id, * EXCEPT(id, episode_id) FROM ChatInference INTO OUTFILE 'small_chat_inference.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(inference_id) as inference_id, * EXCEPT(id, inference_id) FROM ModelInference INTO OUTFILE 'small_model_inference.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(episode_id) as episode_id, * EXCEPT(id, episode_id) FROM JsonInference INTO OUTFILE 'small_json_inference.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(target_id) as target_id, * EXCEPT(id, target_id) FROM BooleanMetricFeedback INTO OUTFILE 'small_boolean_metric_feedback.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(target_id) as target_id, * EXCEPT(id, target_id) FROM FloatMetricFeedback INTO OUTFILE 'small_float_metric_feedback.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(target_id) as target_id, * EXCEPT(id, target_id) FROM CommentFeedback INTO OUTFILE 'small_comment_feedback.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(inference_id) as inference_id, * EXCEPT(id, inference_id) FROM DemonstrationFeedback INTO OUTFILE 'small_demonstration_feedback.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(episode_id) as episode_id, toString(source_inference_id) as source_inference_id, * EXCEPT(id, episode_id, source_inference_id) FROM ChatInferenceDatapoint INTO OUTFILE 'small_chat_inference_datapoint.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(id) as id, toString(episode_id) as episode_id, toString(source_inference_id) as source_inference_id, * EXCEPT(id, episode_id, source_inference_id) FROM JsonInferenceDatapoint INTO OUTFILE 'small_json_inference_datapoint.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT * FROM ModelInferenceCache INTO OUTFILE 'small_model_inference_cache.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT * FROM DynamicEvaluationRun INTO OUTFILE 'small_dynamic_evaluation_run.parquet' FORMAT Parquet"
clickhouse client --user chuser --password chpassword --database tensorzero_ui_fixtures "SELECT toString(run_id) as run_id, * EXCEPT(run_id) FROM DynamicEvaluationRunEpisode INTO OUTFILE 'small_dynamic_evaluation_run_episode.parquet' FORMAT Parquet"
