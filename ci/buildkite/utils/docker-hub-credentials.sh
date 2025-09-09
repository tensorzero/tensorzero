# Set up credentials for docker hub
export DOCKER_HUB_ACCESS_TOKEN=$(buildkite-agent secret get DOCKER_HUB_ACCESS_TOKEN)
if [ -z "$DOCKER_HUB_ACCESS_TOKEN" ]; then
    echo "Error: DOCKER_HUB_ACCESS_TOKEN is not set"
    exit 1
fi

export DOCKER_HUB_USERNAME=$(buildkite-agent secret get DOCKER_HUB_USERNAME)
if [ -z "$DOCKER_HUB_USERNAME" ]; then
    echo "Error: DOCKER_HUB_USERNAME is not set"
    exit 1
fi
