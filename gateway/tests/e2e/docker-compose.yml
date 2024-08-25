services:
  clickhouse:
    container_name: tensorzero-clickhouse-e2e-test
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123" # HTTP port
      - "9000:9000" # Native port
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    healthcheck:
      test:
        [
          "CMD",
          "wget",
          "--no-verbose",
          "--tries=1",
          "--spider",
          "http://localhost:8123/ping",
        ]
      start_period: 30s
      start_interval: 1s
      timeout: 1s
  gateway:
    container_name: tensorzero-gateway-e2e-test
    build:
      context: ../../..
      target: gateway
      args:
        - CARGO_BUILD_FLAGS=--features=e2e_tests
    ports:
      - "3000:3000"
    command: ["gateway", "/app/tensorzero.toml"]
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?Environemnt variable ANTHROPIC_API_KEY is not set}
      - AWS_REGION=${AWS_REGION:?Environment variable AWS_REGION is not set}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:?Environment variable AWS_ACCESS_KEY_ID is not set}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:?Environment variable AWS_SECRET_ACCESS_KEY is not set}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:?Environment variable AZURE_OPENAI_API_KEY is not set}
      - CLICKHOUSE_URL=http://clickhouse:8123
      - FIREWORKS_API_KEY=${FIREWORKS_API_KEY:?Environment variable FIREWORKS_API_KEY is not set}
      - GCP_VERTEX_CREDENTIALS_PATH=/app/gcp-credentials.json
      - MISTRAL_API_KEY=${MISTRAL_API_KEY:?Environment variable MISTRAL_API_KEY is not set}
      - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY is not set}
      - TOGETHER_API_KEY=${TOGETHER_API_KEY:?Environment variable TOGETHER_API_KEY is not set}
      - VLLM_API_KEY=${VLLM_API_KEY:?Environment variable VLLM_API_KEY is not set}
    volumes:
      - ./tensorzero.toml:/app/tensorzero.toml
      - ../../../config:/app/config
      - ${GCP_VERTEX_CREDENTIALS_PATH:?Environment variable GCP_VERTEX_CREDENTIALS_PATH is not set}:/app/gcp-credentials.json
    depends_on:
      clickhouse:
        condition: service_healthy
