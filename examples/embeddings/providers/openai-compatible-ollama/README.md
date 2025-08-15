# Ollama Embeddings with TensorZero

This example demonstrates how to use Ollama embeddings through TensorZero. Ollama provides local, private embedding models that run entirely on your machine without requiring API keys.

## Overview

This example uses:
- **Ollama**: Local inference server for embedding models
- **nomic-embed-text**: High-quality embedding model (768 dimensions)
- **TensorZero**: Unified interface with observability
- **Docker Compose**: Simple deployment setup

## Prerequisites

1. **Docker and Docker Compose**
2. **Python 3.8+**
3. **At least 4GB RAM** (for the embedding model)

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama and TensorZero Gateway:**
   ```bash
   docker-compose up -d
   ```

   This will:
   - Start Ollama server on port 11434
   - Download the `nomic-embed-text` model (~ 274MB)
   - Start TensorZero Gateway on port 3000

3. **Wait for services to be ready:**
   ```bash
   # Check if Ollama is ready
   curl http://localhost:11434/api/tags
   
   # Check if Gateway is ready
   curl http://localhost:3000/health
   ```

4. **Run the Python example:**
   ```bash
   python main.py
   ```

5. **Run the Node.js example (optional):**
   ```bash
   npm install
   npm start
   ```

## What This Example Shows

### Method 1: TensorZero Client (Recommended)
```python
async with AsyncTensorZeroGateway.build_embedded(
    config_file="config/tensorzero.toml"
) as client:
    response = await client.embed(
        model_name="nomic-embed-text",
        input=texts
    )
```

### Method 2: OpenAI SDK via TensorZero Gateway
```python
client = AsyncOpenAI(base_url="http://localhost:3000/openai/v1")
response = await client.embeddings.create(
    input="Your text here",
    model="nomic-embed-text"
)
```

## Configuration

The `config/tensorzero.toml` configures Ollama as an OpenAI-compatible provider:

```toml
[embedding_models.nomic-embed-text]
routing = ["ollama"]

[embedding_models.nomic-embed-text.providers.ollama]
type = "openai"
model_name = "nomic-embed-text"
api_base = "http://ollama:11434/v1"
```

## Available Models

You can use other Ollama embedding models by:

1. **Pulling the model:**
   ```bash
   docker-compose exec ollama ollama pull all-minilm
   ```

2. **Adding to config:**
   ```toml
   [embedding_models.all-minilm]
   routing = ["ollama"]
   
   [embedding_models.all-minilm.providers.ollama]
   type = "openai"
   model_name = "all-minilm"
   api_base = "http://ollama:11434/v1"
   ```

Popular embedding models in Ollama:
- `nomic-embed-text` - High quality, 768 dimensions
- `all-minilm` - Fast and lightweight, 384 dimensions
- `mxbai-embed-large` - Large context window, 1024 dimensions

## Advantages of Local Embeddings

- **Privacy**: Your data never leaves your machine
- **No API costs**: No per-token charges
- **Low latency**: Local inference is fast
- **Offline capable**: Works without internet
- **Customizable**: Use any model supported by Ollama

## Troubleshooting

**Ollama not starting:**
```bash
docker-compose logs ollama
```

**Model download issues:**
```bash
# Check available models
docker-compose exec ollama ollama list

# Manually pull model
docker-compose exec ollama ollama pull nomic-embed-text
```

**Gateway connection issues:**
```bash
# Check gateway logs
docker-compose logs gateway

# Verify Ollama is accessible from gateway
docker-compose exec gateway curl http://ollama:11434/api/tags
```

## Cleanup

```bash
# Stop services
docker-compose down

# Remove volumes (this will delete downloaded models)
docker-compose down -v
```

## Next Steps

- Try different embedding models available in Ollama
- Integrate with your text processing pipeline
- Scale with multiple Ollama instances
- Explore TensorZero's observability features

## Learn More

- [Ollama Documentation](https://github.com/ollama/ollama)
- [TensorZero Documentation](https://www.tensorzero.com/docs)
- [Nomic Embed Text Model](https://huggingface.co/nomic-ai/nomic-embed-text-v1)