#!/usr/bin/env python3
"""
TensorZero Ollama Embeddings Example

This example demonstrates how to use Ollama embeddings through TensorZero
using the OpenAI-compatible API. Ollama runs locally via Docker.
"""

import asyncio
from openai import AsyncOpenAI
from tensorzero import AsyncTensorZeroGateway, patch_openai_client


async def main():
    print("🦙 TensorZero Ollama Embeddings Example")
    print("=" * 50)
    
    # Sample texts to embed
    texts = [
        "Local embeddings with Ollama are fast and private.",
        "TensorZero provides a unified interface for multiple embedding providers.",
        "Nomic Embed Text is a high-quality embedding model that runs locally."
    ]
    
    print(f"📝 Input texts:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    print()
    
    # Method 1: Using TensorZero Client (Recommended)
    print("Method 1: TensorZero Client (Recommended)")
    print("-" * 40)
    
    try:
        async with AsyncTensorZeroGateway.build_embedded(
            config_file="config/tensorzero.toml"
        ) as client:
            response = await client.embed(
                model_name="nomic-embed-text",
                input=texts
            )
            
            print(f"✅ Model: {response.model}")
            print(f"📊 Generated {len(response.embeddings)} embeddings")
            print(f"📏 Embedding dimensions: {len(response.embeddings[0])}")
            print(f"🔢 Token usage - Input: {response.usage.input_tokens}, Total: {response.usage.total_tokens}")
            print()
    except Exception as e:
        print(f"❌ Error with TensorZero client: {e}")
        print("   Make sure the gateway is running with: docker-compose up")
        print()
    
    # Method 2: Using OpenAI SDK with TensorZero Gateway via HTTP
    print("Method 2: OpenAI SDK with TensorZero Gateway (HTTP)")
    print("-" * 40)
    
    client = AsyncOpenAI(base_url="http://localhost:3000/openai/v1")
    
    try:
        # Single text embedding
        response = await client.embeddings.create(
            input=texts[0],
            model="nomic-embed-text"
        )
        
        print(f"✅ Model: {response.model}")
        print(f"📊 Generated {len(response.data)} embedding(s)")
        print(f"📏 Embedding dimensions: {len(response.data[0].embedding)}")
        print(f"🔢 Token usage - Prompt: {response.usage.prompt_tokens}, Total: {response.usage.total_tokens}")
        print()
        
        # Batch text embeddings
        response = await client.embeddings.create(
            input=texts,
            model="nomic-embed-text"
        )
        
        print(f"🎯 Batch embedding:")
        print(f"📊 Generated {len(response.data)} embeddings")
        print(f"📏 Embedding dimensions: {len(response.data[0].embedding)}")
        print(f"🔢 Token usage - Prompt: {response.usage.prompt_tokens}, Total: {response.usage.total_tokens}")
        print()
        
        # Show first few dimensions of first embedding
        embedding = response.data[0].embedding
        print(f"🔍 First 5 dimensions of embedding: {embedding[:5]}")
        print()
        
    except Exception as e:
        print(f"❌ Error with OpenAI client: {e}")
        print("   Make sure the gateway is running with: docker-compose up")
        print()
    finally:
        await client.close()
    
    print("✨ Example completed!")
    print()
    print("💡 Tips:")
    print("   - Ollama embeddings run locally and don't require API keys")
    print("   - nomic-embed-text produces 768-dimensional embeddings")
    print("   - You can try other embedding models by pulling them with Ollama")
    print("   - Use 'docker-compose logs ollama' to see Ollama's output")


if __name__ == "__main__":
    asyncio.run(main())