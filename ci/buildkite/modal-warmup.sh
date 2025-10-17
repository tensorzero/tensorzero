#!/bin/bash
set -euo pipefail

echo "Starting Modal service warmup..."

# Get Modal credentials from Buildkite secrets
export MODAL_KEY=$(buildkite-agent secret get MODAL_KEY)
if [ -z "$MODAL_KEY" ]; then
    echo "Error: MODAL_KEY is not set"
    exit 1
fi

export MODAL_SECRET=$(buildkite-agent secret get MODAL_SECRET)
if [ -z "$MODAL_SECRET" ]; then
    echo "Error: MODAL_SECRET is not set"
    exit 1
fi

echo "Warming up VLLM Modal instance..."
curl -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
     https://tensorzero--vllm-inference-vllm-inference.modal.run/docs \
     > vllm_modal_logs.txt &

echo "Warming up SGLang Modal instance..."
curl -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
     https://tensorzero--sglang-0-4-10-inference-sglang-inference.modal.run/ \
     > sglang_modal_logs.txt &

echo "Waiting for warmup requests to complete..."
wait

echo "Modal service warmup completed successfully"