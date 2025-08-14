# OpenAI Embeddings with TensorZero

This example demonstrates how to use OpenAI embeddings through TensorZero, showcasing both the TensorZero client and the OpenAI SDK with TensorZero's patch functionality.

## Overview

TensorZero provides seamless integration with OpenAI's embedding models, offering:

- Unified API for multiple embedding providers
- Automatic observability and logging
- Easy model switching and experimentation
- Token usage tracking

## Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key**: Sign up at [OpenAI](https://platform.openai.com/) and create an API key

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key:**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the Python example:**

   ```bash
   python main.py
   ```

4. **Run the Node.js example (optional):**

   ```bash
   npm install
   npm start
   ```

## What This Example Shows

The example demonstrates two methods for using embeddings:

### Method 1: TensorZero Client (Recommended)

```python
async with AsyncTensorZeroGateway.build_embedded(
    config_file="config/tensorzero.toml"
) as client:
    response = await client.embed(
        model_name="text-embedding-3-small",
        input=texts
    )
```

### Method 2: OpenAI SDK with TensorZero Patch

```python
client = AsyncOpenAI()
client = await patch_openai_client(
    client,
    config_file="config/tensorzero.toml"
)

response = await client.embeddings.create(
    input="Your text here",
    model="text-embedding-3-small"
)
```

## Configuration

The `config/tensorzero.toml` file defines the available embedding models:

- `text-embedding-3-small` - Latest small embedding model (1536 dimensions)
- `text-embedding-3-large` - Latest large embedding model (3072 dimensions)  
- `text-embedding-ada-002` - Legacy embedding model (1536 dimensions)

## Features Demonstrated

- **Single text embedding**: Generate embeddings for individual texts
- **Batch embedding**: Process multiple texts in a single request
- **Custom dimensions**: Specify embedding dimensions (for supported models)
- **Token usage tracking**: Monitor token consumption
- **Model comparison**: Easy switching between embedding models

## Next Steps

- Try different embedding models by changing the `model` parameter
- Experiment with custom dimensions using the `dimensions` parameter
- Integrate with your own text processing pipeline
- Explore TensorZero's observability features with a full gateway deployment

## Learn More

- [TensorZero Documentation](https://www.tensorzero.com/docs)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [TensorZero Gateway Configuration](https://www.tensorzero.com/docs/gateway/configuration)