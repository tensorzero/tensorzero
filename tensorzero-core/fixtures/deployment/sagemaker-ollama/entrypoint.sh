#!/bin/bash

set -euxo pipefail
export HOME=/home/ec2-user/SageMaker/temp

ollama serve &
# Retry pulling, since the server may not have started or the pull may fail
for i in {1..10}; do
    ollama pull gemma3:1b && break || sleep 1
done

# The flask dev server is good enough for our use case (running e2e tests on CI)
uv run flask --app proxy run --port 8080 --host 0.0.0.0
