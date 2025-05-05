#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source ./dummy-env-file.env

for env_example in $(find "$(cd ..; pwd)" -name ".env.example"); do
  env_file="$(dirname "$env_example")/.env"
  if [ ! -f "$env_file" ]; then
    cp ./dummy-env-file.env "$env_file"
    echo "Copied ./dummy-env-file.env to $env_file"
  fi
done

cp ./dummy-gcp-credentials.json /tmp/dummy-gcp-credentials.json

find "$(cd ..; pwd)" -name "docker-compose.yml" -print0 | xargs -0L1 bash -c './check-docker-compose.sh $0 || exit 255'
