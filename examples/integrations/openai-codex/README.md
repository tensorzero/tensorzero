# Example: Integrating OpenAI Codex with TensorZero

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
2. Install Node 24.11.0.
3. Generate credentials for the providers you want to use (e.g. `ANTHROPIC_API_KEY`).

### Setup

1. Create a `.env` file with your provider credentials. (See `.env.example` for reference.)
2. Run `docker compose up` to start TensorZero.
3. Install Codex: `npm i -g @openai/codex`
4. Add the TensorZero Gateway to your Codex configuration (`~/.config/config.yaml` or `~/.codex/config.json`):

   ```yaml
   model: "tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219"
   provider: tensorzero
   providers:
     tensorzero:
       name: TensorZero
       baseURL: http://localhost:3000/openai/v1
       envKey: TENSORZERO_API_KEY # not used but required by Codex
     # ... other providers ...
   ```

   ```json
   {
     "model": "tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219",
     "provider": "tensorzero",
     "providers": {
       "tensorzero": {
         "name": "TensorZero",
         "baseURL": "http://localhost:3000/openai/v1",
         "envKey": "TENSORZERO_API_KEY"
       }
     }
   }
   ```

5. Run Codex with TensorZero:
   ```bash
   TENSORZERO_API_KEY="not-used" codex
   # or set the environment variable in your shell and just run `codex`
   ```

You can replace `tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219` with any other model supported by TensorZero, e.g. `tensorzero::model_name::mistral::open-mistral-nemo-2407`.

You can also define custom TensorZero functions in the `config/tensorzero.toml` file, and use them with Codex as `tensorzero::function_name::your_function_name`.
This will enable you to use advanced inference features, collect data for fine-tuning and other optimization recipes, and more.
See our [Quick Start Guide](https://www.tensorzero.com/docs/quickstart/) for more details.

<p align="center"><img src="https://github.com/user-attachments/assets/0a3192e9-f2ed-4b86-b3d0-966cbf6ea14f" alt="OpenAI Codex Observability with the TensorZero UI" /><br><em>OpenAI Codex Observability with the TensorZero UI</em> &mdash; <code>http://localhost:4000/</code></p>
