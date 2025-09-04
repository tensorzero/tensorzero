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

# Array to collect failed containers
failed_containers=()

# Test each docker-compose.yml file
while IFS= read -r -d '' compose_file; do
  echo "Testing: $compose_file"
  if ! ./check-docker-compose.sh "$compose_file"; then
    failed_containers+=("$compose_file")
    echo "FAILED: $compose_file"
  else
    echo "PASSED: $compose_file"
  fi
  echo "----------------------------------------"
done < <(find "$(cd ..; pwd)" -name "docker-compose.yml" -print0)

# Report results
echo "============================================"
echo "SUMMARY:"
if [ ${#failed_containers[@]} -eq 0 ]; then
  echo "All containers passed!"
else
  echo "Failed containers (${#failed_containers[@]}):"
  for container in "${failed_containers[@]}"; do
    echo "  - $container"
  done
  exit 1
fi
