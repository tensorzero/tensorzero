# Getting Started with NVIDIA NIM

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with NVIDIA NIM API.

## Simple Setup

You can use the short-hand `nvidia_nim::model_name` to use an NVIDIA NIM model with TensorZero, unless you need advanced features like fallbacks or custom credentials.

You can use NVIDIA NIM models in your TensorZero variants by setting the model field to `nvidia_nim::model_name`. For example:

```toml
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "nvidia_nim::llama-3.1-8b-instruct"
```

Additionally, you can set model_name in the inference request to use a specific NVIDIA NIM model, without having to configure a function and variant in TensorZero.

curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "nvidia_nim::llama-3.1-8b-instruct",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of Japan?"
        }
      ]
    }
  }'

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and NVIDIA NIM provider in TensorZero.

For this minimal setup, you'll need just two files in your project directory:

Directory config/
  tensorzero.toml
docker-compose.yml

See the list of models available on NVIDIA NIM. (https://build.nvidia.com/explore/discover)

Credentials
You must set the NVIDIA_API_KEY environment variable before running the gateway.

You can customize the credential location by setting the api_key_location to env::YOUR_ENVIRONMENT_VARIABLE or dynamic::ARGUMENT_NAME. See the Credential Management guide and Configuration Reference for more information.

Deployment (Docker Compose)
Create a minimal Docker Compose configuration:

You can start the gateway with:

docker compose up.

Inference
Make an inference request to the gateway:

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

Prerequisites
Before running this example, make sure you have:

Docker and Docker Compose installed
An NVIDIA API key from NVIDIA Build
Set your API key as an environment variable: export NVIDIA_API_KEY=your_api_key_here


Quick Start
Clone or create this directory structure
Set your NVIDIA API key: export NVIDIA_API_KEY=your_api_key_here
Run docker compose up
Make inference requests to http://localhost:3000/inference

Available Models
Some popular NVIDIA NIM models include:

llama-3.1-8b-instruct - Llama 3.1 8B Instruct model
llama-3.1-70b-instruct - Llama 3.1 70B Instruct model
llama-3.1-405b-instruct - Llama 3.1 405B Instruct model
mistral-7b-instruct-v0.3 - Mistral 7B Instruct v0.3
mixtral-8x7b-instruct-v0.1 - Mixtral 8x7B Instruct v0.1
For a complete list, visit NVIDIA Build.