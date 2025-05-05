#!/bin/bash

# Invoke this script as './deploy.sh VERSION' (e.g. './deploy.sh 0.9.0')
# This will build a Docker image tagged with 'VERSION', push to ECR, and deploy
# it to Sagemaker

set -euxo pipefail
VERSION=$1
IMAGE=637423354485.dkr.ecr.us-east-2.amazonaws.com/sagemaker-ollama:$VERSION

# cd to parent directory
cd "$(dirname "$0")/.."

docker build . -t $IMAGE
docker push $IMAGE
DOCKER_IMAGE_URI=$IMAGE uv run python ./scripts/deploy_endpoint.py $IMAGE
