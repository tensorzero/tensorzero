# Getting Started with NVIDIA NIM

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with NVIDIA NIM API.

## Simple Setup

You can use the short-hand `nvidia_nim::model_name` to use an NVIDIA NIM model with TensorZero, unless you need advanced features like fallbacks or custom credentials.

You can use NVIDIA NIM models in your TensorZero variants by setting the model field to `nvidia_nim::model_name`. For example:

```toml
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "nvidia_nim::meta/llama-3.1-8b-instruct"
```

Additionally, you can set model_name in the inference request to use a specific NVIDIA NIM model, without having to configure a function and variant in TensorZero.

```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "nvidia_nim::meta/llama-3.1-8b-instruct",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of Japan?"
        }
      ]
    }
  }'
```

## Advanced Setup
In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and NVIDIA NIM provider in TensorZero.

For this minimal setup, you'll need just two files in your project directory:
```
Directory config/
  tensorzero.toml
docker-compose.yml
```

## Configuration

Create a minimal configuration file that defines a model and a simple chat function:

**config/tensorzero.toml**
```toml
[models.llama_3_1_8b_instruct]
routing = ["nvidia_nim"]

[models.llama_3_1_8b_instruct.providers.nvidia_nim]
type = "nvidia_nim"
model_name = "meta/llama-3.1-8b-instruct"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "llama_3_1_8b_instruct"
```
See the list of models available on [NVIDIA NIM](https://build.nvidia.com/explore/discover).

## Credentials

- You must set the NVIDIA_API_KEY environment variable before running the gateway.
- You can customize the credential location by setting the api_key_location to env::YOUR_ENVIRONMENT_VARIABLE or dynamic::ARGUMENT_NAME. See the Credential Management guide and Configuration Reference for more information.

## Deployment (Docker Compose)
Create a minimal Docker Compose configuration:

**docker-compose.yml**
```yaml
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/gateway/deployment

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY:?Environment variable NVIDIA_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with:

`docker compose up`

## Inference
Make an inference request to the gateway:

**Using curl (Linux/Mac/Git Bash):**
```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of Japan?"
        }
      ]
    }
  }'
```
**Using PowerShell (Windows):**
```powershell
$body = @{
    function_name = "my_function_name"
    input = @{
        messages = @(
            @{
                role = "user"
                content = "What is the capital of Japan?"
            }
        )
    }
} | ConvertTo-Json -Depth 10

Invoke-WebRequest -Uri "http://localhost:3000/inference" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body
```

## Prerequisites
Before running this example, make sure you have:

- Docker and Docker Compose installed
- An NVIDIA API key from [NVIDIA Build](https://build.nvidia.com/)
- Set your API key as an environment variable: (linux/macOS): `export NVIDIA_API_KEY=your_api_key_here`
- Set your API key as an environment variable (Windows): `$env:NVIDIA_API_KEY=your_api_key_here`



## Quick Start
- Clone or create this directory structure
- Set your API key as an environment variable: (linux/macOS): `export NVIDIA_API_KEY=your_api_key_here`
- Set your API key as an environment variable (Windows): `$env:NVIDIA_API_KEY=your_api_key_here`
- Run `docker compose up`
- Make inference requests to `http://localhost:3000/inference`

## Available Models
Some popular NVIDIA NIM models include:

- `meta/llama-3.1-8b-instruct` - Llama 3.1 8B Instruct model
- `meta/llama-3.1-70b-instruct` - Llama 3.1 70B Instruct model
- `meta/llama-3.1-405b-instruct` - Llama 3.1 405B Instruct model
- `mistralai/mistral-7b-instruct-v0.3` - Mistral 7B Instruct v0.3
- `mistralai/mixtral-8x7b-instruct-v0.1` - Mixtral 8x7B Instruct v0.1

For a complete list, visit [NVIDIA Build](https://build.nvidia.com/explore/discover).
