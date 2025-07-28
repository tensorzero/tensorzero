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