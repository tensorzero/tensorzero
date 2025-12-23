# Set up Buildkite test analytics collection
export BUILDKITE_ANALYTICS_TOKEN=$(buildkite-agent secret get LIVE_TESTS_ANALYTICS_ACCESS_TOKEN)
if [ -z "$BUILDKITE_ANALYTICS_TOKEN" ]; then
    echo "Error: BUILDKITE_ANALYTICS_TOKEN is not set"
    exit 1
fi

# Set up all other environment variables
export ANTHROPIC_API_KEY=$(buildkite-agent secret get ANTHROPIC_API_KEY)
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY is not set"
    exit 1
fi

export AWS_ACCESS_KEY_ID=$(buildkite-agent secret get AWS_ACCESS_KEY_ID)
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "Error: AWS_ACCESS_KEY_ID is not set"
    exit 1
fi

export AWS_REGION="us-east-1"

export AWS_SECRET_ACCESS_KEY=$(buildkite-agent secret get AWS_SECRET_ACCESS_KEY)
if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS_SECRET_ACCESS_KEY is not set"
    exit 1
fi

export AZURE_OPENAI_API_BASE=$(buildkite-agent secret get AZURE_OPENAI_API_BASE)
if [ -z "$AZURE_OPENAI_API_BASE" ]; then
    echo "Error: AZURE_OPENAI_API_BASE is not set"
    exit 1
fi

export AZURE_OPENAI_API_KEY=$(buildkite-agent secret get AZURE_OPENAI_API_KEY)
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "Error: AZURE_OPENAI_API_KEY is not set"
    exit 1
fi

export AZURE_OPENAI_EASTUS2_API_KEY=$(buildkite-agent secret get AZURE_OPENAI_EASTUS2_API_KEY)
if [ -z "$AZURE_OPENAI_EASTUS2_API_KEY" ]; then
    echo "Error: AZURE_OPENAI_EASTUS2_API_KEY is not set"
    exit 1
fi

export AZURE_AI_FOUNDRY_API_KEY=$(buildkite-agent secret get AZURE_AI_FOUNDRY_API_KEY)
if [ -z "$AZURE_AI_FOUNDRY_API_KEY" ]; then
    echo "Error: AZURE_AI_FOUNDRY_API_KEY is not set"
    exit 1
fi

export AZURE_OPENAI_DEPLOYMENT_ID=$(buildkite-agent secret get AZURE_OPENAI_DEPLOYMENT_ID)
if [ -z "$AZURE_OPENAI_DEPLOYMENT_ID" ]; then
    echo "Error: AZURE_OPENAI_DEPLOYMENT_ID is not set"
    exit 1
fi

export DEEPSEEK_API_KEY=$(buildkite-agent secret get DEEPSEEK_API_KEY)
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Error: DEEPSEEK_API_KEY is not set"
    exit 1
fi

export FIREWORKS_API_KEY=$(buildkite-agent secret get FIREWORKS_API_KEY)
if [ -z "$FIREWORKS_API_KEY" ]; then
    echo "Error: FIREWORKS_API_KEY is not set"
    exit 1
fi

export FIREWORKS_ACCOUNT_ID=$(buildkite-agent secret get FIREWORKS_ACCOUNT_ID)
if [ -z "$FIREWORKS_ACCOUNT_ID" ]; then
    echo "Error: FIREWORKS_ACCOUNT_ID is not set"
    exit 1
fi

export FORCE_COLOR=1

export GCP_STORAGE_ACCESS_KEY_ID=$(buildkite-agent secret get GCP_STORAGE_ACCESS_KEY_ID)
if [ -z "$GCP_STORAGE_ACCESS_KEY_ID" ]; then
    echo "Error: GCP_STORAGE_ACCESS_KEY_ID is not set"
    exit 1
fi

export GCP_STORAGE_SECRET_ACCESS_KEY=$(buildkite-agent secret get GCP_STORAGE_SECRET_ACCESS_KEY)
if [ -z "$GCP_STORAGE_SECRET_ACCESS_KEY" ]; then
    echo "Error: GCP_STORAGE_SECRET_ACCESS_KEY is not set"
    exit 1
fi

export GOOGLE_AI_STUDIO_API_KEY=$(buildkite-agent secret get GOOGLE_AI_STUDIO_API_KEY)
if [ -z "$GOOGLE_AI_STUDIO_API_KEY" ]; then
    echo "Error: GOOGLE_AI_STUDIO_API_KEY is not set"
    exit 1
fi

export GROQ_API_KEY=$(buildkite-agent secret get GROQ_API_KEY)
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY is not set"
    exit 1
fi

export HYPERBOLIC_API_KEY=$(buildkite-agent secret get HYPERBOLIC_API_KEY)
if [ -z "$HYPERBOLIC_API_KEY" ]; then
    echo "Error: HYPERBOLIC_API_KEY is not set"
    exit 1
fi

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

export MISTRAL_API_KEY=$(buildkite-agent secret get MISTRAL_API_KEY)
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "Error: MISTRAL_API_KEY is not set"
    exit 1
fi

export OPENAI_API_KEY=$(buildkite-agent secret get OPENAI_API_KEY)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    exit 1
fi

export OPENROUTER_API_KEY=$(buildkite-agent secret get OPENROUTER_API_KEY)
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY is not set"
    exit 1
fi

export R2_ACCESS_KEY_ID=$(buildkite-agent secret get R2_ACCESS_KEY_ID)
if [ -z "$R2_ACCESS_KEY_ID" ]; then
    echo "Error: R2_ACCESS_KEY_ID is not set"
    exit 1
fi

export R2_SECRET_ACCESS_KEY=$(buildkite-agent secret get R2_SECRET_ACCESS_KEY)
if [ -z "$R2_SECRET_ACCESS_KEY" ]; then
    echo "Error: R2_SECRET_ACCESS_KEY is not set"
    exit 1
fi

export SGLANG_API_KEY=$(buildkite-agent secret get SGLANG_API_KEY)
if [ -z "$SGLANG_API_KEY" ]; then
    echo "Error: SGLANG_API_KEY is not set"
    exit 1
fi

export TGI_API_KEY=$(buildkite-agent secret get TGI_API_KEY)
if [ -z "$TGI_API_KEY" ]; then
    echo "Error: TGI_API_KEY is not set"
    exit 1
fi

export TOGETHER_API_KEY=$(buildkite-agent secret get TOGETHER_API_KEY)
if [ -z "$TOGETHER_API_KEY" ]; then
    echo "Error: TOGETHER_API_KEY is not set"
    exit 1
fi

export VLLM_API_KEY=$(buildkite-agent secret get VLLM_API_KEY)
if [ -z "$VLLM_API_KEY" ]; then
    echo "Error: VLLM_API_KEY is not set"
    exit 1
fi


export VOYAGE_API_KEY=$(buildkite-agent secret get VOYAGE_API_KEY)
if [ -z "$VOYAGE_API_KEY" ]; then
    echo "Error: VOYAGE_API_KEY is not set"
    exit 1
fi

export XAI_API_KEY=$(buildkite-agent secret get XAI_API_KEY)
if [ -z "$XAI_API_KEY" ]; then
    echo "Error: XAI_API_KEY is not set"
    exit 1
fi
