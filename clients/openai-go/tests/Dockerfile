# ========== test-runner ==========

FROM golang:1.25.1-alpine AS test-runner

# Install system dependencies
RUN apk --no-cache add ca-certificates git curl

# Install gotestsum
RUN go install gotest.tools/gotestsum@latest

WORKDIR /app

# Copy go.mod and go.sum for dependency management
COPY clients/openai-go/go.mod ./clients/openai-go/
COPY clients/openai-go/go.sum ./clients/openai-go/

WORKDIR /app/clients/openai-go

# Download dependencies
RUN go mod download

# Copy test files and scripts
COPY clients/openai-go/tests/ ./tests/

# Copy test assets
COPY tensorzero-core/tests/e2e/providers/ferris.png ./tensorzero-core/tests/e2e/providers/ferris.png
COPY tensorzero-core/tests/e2e/providers/deepseek_paper.pdf ./tensorzero-core/tests/e2e/providers/deepseek_paper.pdf

# Make the test script executable
RUN chmod +x ./tests/run-tests.sh

# Run the tests
CMD ["./tests/run-tests.sh"]
