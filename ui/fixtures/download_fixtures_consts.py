from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

PART_SIZE = 8388608

# Map of remote filename (in R2) -> local filename
# When a file needs updating, add version suffix to remote name (e.g., "file_v2.jsonl")
# and keep the local name unchanged (e.g., "file.jsonl")
SMALL_FIXTURES = {
    "model_inference_examples_20260203.jsonl": "model_inference_examples.jsonl",
    "chat_inference_examples_20260123.jsonl": "chat_inference_examples.jsonl",
    "json_inference_examples.jsonl": "json_inference_examples.jsonl",
    "boolean_metric_feedback_examples.jsonl": "boolean_metric_feedback_examples.jsonl",
    "float_metric_feedback_examples.jsonl": "float_metric_feedback_examples.jsonl",
    "demonstration_feedback_examples.jsonl": "demonstration_feedback_examples.jsonl",
    "model_inference_cache_e2e_20260122_183412.jsonl": "model_inference_cache_e2e.jsonl",
    "json_inference_datapoint_examples.jsonl": "json_inference_datapoint_examples.jsonl",
    "chat_inference_datapoint_examples_20260129.jsonl": "chat_inference_datapoint_examples.jsonl",
    "dynamic_evaluation_run_episode_examples.jsonl": "dynamic_evaluation_run_episode_examples.jsonl",
    "jaro_winkler_similarity_feedback.jsonl": "jaro_winkler_similarity_feedback.jsonl",
    "comment_feedback_examples.jsonl": "comment_feedback_examples.jsonl",
    "dynamic_evaluation_run_examples.jsonl": "dynamic_evaluation_run_examples.jsonl",
}

LARGE_FIXTURES = [
    "large_chat_inference_v2.parquet",
    "large_chat_model_inference_v2.parquet",
    "large_json_inference_v2.parquet",
    "large_json_model_inference_v2.parquet",
    "large_chat_boolean_feedback.parquet",
    "large_chat_float_feedback.parquet",
    "large_chat_comment_feedback.parquet",
    "large_chat_demonstration_feedback.parquet",
    "large_json_boolean_feedback.parquet",
    "large_json_float_feedback.parquet",
    "large_json_comment_feedback.parquet",
    "large_json_demonstration_feedback.parquet",
]
R2_PUBLIC_BUCKET_URL = "https://pub-147e9850a60643208c411e70b636e956.r2.dev"
R2_S3_ENDPOINT_URL = "https://19918a216783f0ac9e052233569aef60.r2.cloudflarestorage.com/"

LARGE_FIXTURES_DIR = Path("./large-fixtures")
SMALL_FIXTURES_DIR = Path("./small-fixtures")
