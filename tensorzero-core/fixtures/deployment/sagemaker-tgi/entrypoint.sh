#!/bin/bash

set -euxo pipefail
export HOME=/home/ec2-user/SageMaker/temp

echo "User: $(whoami)"
text-generation-launcher --port 8081 &

# The flask dev server is good enough for our use case (running e2e tests on CI)
cd /app && uv run flask --app proxy run --port 8080 --host 0.0.0.0 &

# Wait for any process to exit
wait -n
# Exit with status of process that exited first
exit $?
