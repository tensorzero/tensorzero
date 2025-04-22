# Example: OpenAI Codex and TensorZero Integration

This example shows how to use OpenAI Codex with TensorZero — fully open-source and self-hosted.

Why?

- Use every major model provider with Codex:
  [Anthropic](https://www.tensorzero.com/docs/gateway/guides/providers/anthropic),
  [AWS Bedrock](https://www.tensorzero.com/docs/gateway/guides/providers/aws-bedrock),
  [AWS SageMaker](https://www.tensorzero.com/docs/gateway/guides/providers/aws-sagemaker),
  [Azure OpenAI Service](https://www.tensorzero.com/docs/gateway/guides/providers/azure),
  [DeepSeek](https://www.tensorzero.com/docs/gateway/guides/providers/deepseek),
  [Fireworks](https://www.tensorzero.com/docs/gateway/guides/providers/fireworks),
  [GCP Vertex AI Anthropic](https://www.tensorzero.com/docs/gateway/guides/providers/gcp-vertex-ai-anthropic),
  [GCP Vertex AI Gemini](https://www.tensorzero.com/docs/gateway/guides/providers/gcp-vertex-ai-gemini),
  [Google AI Studio (Gemini API)](https://www.tensorzero.com/docs/gateway/guides/providers/google-ai-studio-gemini),
  [Hyperbolic](https://www.tensorzero.com/docs/gateway/guides/providers/hyperbolic),
  [Mistral](https://www.tensorzero.com/docs/gateway/guides/providers/mistral),
  [OpenAI](https://www.tensorzero.com/docs/gateway/guides/providers/openai),
  [Together](https://www.tensorzero.com/docs/gateway/guides/providers/together),
  [vLLM](https://www.tensorzero.com/docs/gateway/guides/providers/vllm),
  [xAI](https://www.tensorzero.com/docs/gateway/guides/providers/xai),
  and [any OpenAI-compatible provider (e.g. Ollama)](https://www.tensorzero.com/docs/gateway/guides/providers/openai-compatible)
- Gain comprehensive observability into your Codex usage, including detailed logs for every LLM call.
- Set up advanced inference features like [retries](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks/), [fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks/), [load balancing](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks/#load-balancing), [inference-time optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations/), [experimentation (A/B testing)](https://www.tensorzero.com/docs/gateway/guides/experimentation/), and more.
- Collect data for fine-tuning and other optimization techniques, and use TensorZero recipes to create custom models for your Codex usage.

## Getting Started

### Prerequisites

1. Install Docker.
2. Install Node 22+.
3. Generate credentials for the providers you want to use (e.g. `ANTHROPIC_API_KEY`).

### Setup

1. Create a `.env` file with your provider credentials. (See `.env.example` for reference.)
2. Run `docker compose up` to start TensorZero.
3. Install Codex:

   > At the time of writing, OpenAI has merged the required features for using multiple providers with Codex, but hasn't released a new version of Codex on `npm`, so you'll need to build it from source.
   >
   > If OpenAI makes a new release, you can skip this step and instead install Codex with `npm i -g @openai/codex`.

   ```bash
   # Clone the Codex repository
   git clone https://github.com/openai/codex.git

   # Install dependencies
   cd codex/codex-cli

   # Install the Codex dependencies
   npm install

   # Build Codex
   npm run build

   # Link the version of Codex you just built
   npm link
   ```

4. Run Codex with TensorZero:
   ```bash
   OPENAI_BASE_URL="http://localhost:3000/openai/v1" OPENAI_API_KEY="not-used" codex -p tensorzero -m tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219
   ```

You can replace `tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219` with any other model supported by TensorZero, e.g. `tensorzero::model_name::mistral::open-mistral-nemo-2407`.

You can also define custom TensorZero functions in the `config/tensorzero.toml` file, and use them with Codex as `tensorzero::function_name::xxx`.
This will enable you to use advanced inference features, collect data for fine-tuning and other optimization recipes, and more.
See our [Quick Start Guide](https://www.tensorzero.com/docs/quickstart/) for more details.
