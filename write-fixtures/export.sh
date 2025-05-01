set -euxo pipefail

clickhouse-client --user chuser --password chpassword --database tensorzero_e2e_tests "SELECT toString(id) as id, toString(episode_id) as episode_id, * EXCEPT(id, episode_id) FROM ChatInference INTO OUTFILE 'large_chat_inference.parquet' FORMAT Parquet"
clickhouse-client --user chuser --password chpassword --database tensorzero_e2e_tests "SELECT toString(id) as id, toString(inference_id) as inference_id, * EXCEPT(id, inference_id) FROM ModelInference INTO OUTFILE 'large_model_inference.parquet' FORMAT Parquet"
