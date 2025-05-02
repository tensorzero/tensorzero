set -euxo pipefail

cd "$(dirname "$0")"
aws s3 sync s3://tensorzero-clickhouse-fixtures ./s3-fixtures