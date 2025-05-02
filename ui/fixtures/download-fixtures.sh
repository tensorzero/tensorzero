set -euxo pipefail

cd "$(dirname "$0")"
# This will avoid downloading the file if they already exist on disk (and are up-to-date)
# On CI, we use a namespace runner that symlinks the s3-fixtures directory to /cache
# to avoid needing to download the files on every run
aws s3 sync s3://tensorzero-clickhouse-fixtures ./s3-fixtures --no-sign-request