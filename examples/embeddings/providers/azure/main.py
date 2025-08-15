#!/usr/bin/env python3
"""
TensorZero Azure OpenAI Embeddings Example

This example demonstrates how to use Azure OpenAI embeddings through TensorZero
using both the TensorZero client and the OpenAI SDK with TensorZero's patch.
"""

import os
import asyncio
from openai import AsyncOpenAI
from tensorzero import AsyncTensorZeroGateway, patch_openai_client


async def main():
    print("☁️ TensorZero Azure OpenAI Embeddings Example")
    print("=" * 50)
    
    # Sample texts to embed
    texts = [
        "Azure OpenAI provides enterprise-grade AI services.",
        "TensorZero seamlessly integrates with Azure OpenAI deployments.",
        "Enterprise AI applications require robust security and compliance."
    ]
    
    print(f"📝 Input texts:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    print()
    
    # Method 1: Using TensorZero Client (Recommended)
    print("Method 1: TensorZero Client (Recommended)")
    print("-" * 40)
    
    async with AsyncTensorZeroGateway.build_embedded(
        config_file="config/tensorzero.toml"
    ) as client:
        response = await client.embed(
            model_name="text-embedding-3-small",
            input=texts
        )
        
        print(f"✅ Model: {response.model}")
        print(f"📊 Generated {len(response.embeddings)} embeddings")
        print(f"📏 Embedding dimensions: {len(response.embeddings[0])}")
        print(f"🔢 Token usage - Input: {response.usage.input_tokens}, Total: {response.usage.total_tokens}")
        print()
    
    # Method 2: Using OpenAI SDK with TensorZero patch
    print("Method 2: OpenAI SDK with TensorZero patch")
    print("-" * 40)
    
    client = AsyncOpenAI()
    client = await patch_openai_client(
        client,
        config_file="config/tensorzero.toml"
    )
    
    try:
        # Single text embedding
        response = await client.embeddings.create(
            input=texts[0],
            model="text-embedding-3-small"
        )
        
        print(f"✅ Model: {response.model}")
        print(f"📊 Generated {len(response.data)} embedding(s)")
        print(f"📏 Embedding dimensions: {len(response.data[0].embedding)}")
        print(f"🔢 Token usage - Prompt: {response.usage.prompt_tokens}, Total: {response.usage.total_tokens}")
        print()
        
        # Batch text embeddings with different model
        response = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-large"
        )
        
        print(f"🎯 Batch embedding with {response.model}:")
        print(f"📊 Generated {len(response.data)} embeddings")
        print(f"📏 Embedding dimensions: {len(response.data[0].embedding)}")
        print(f"🔢 Token usage - Prompt: {response.usage.prompt_tokens}, Total: {response.usage.total_tokens}")
        print()
        
        # Embedding with custom dimensions (if supported by deployment)
        try:
            response = await client.embeddings.create(
                input="Custom dimensions example",
                model="text-embedding-3-small",
                dimensions=512
            )
            
            print(f"🎛️ Custom dimensions example:")
            print(f"📏 Requested 512 dimensions, got: {len(response.data[0].embedding)}")
            print()
        except Exception as e:
            print(f"ℹ️ Custom dimensions not supported by this deployment: {e}")
            print()
        
        # Show first few dimensions of first embedding
        embedding = response.data[0].embedding
        print(f"🔍 First 5 dimensions of embedding: {embedding[:5]}")
        print()
        
    finally:
        await client.close()
    
    print("✨ Example completed successfully!")
    print()
    print("💡 Tips:")
    print("   - Azure OpenAI deployments are region-specific")
    print("   - Update the endpoint and deployment_id in tensorzero.toml")
    print("   - Consider using different regions for redundancy")
    print("   - Monitor usage through Azure portal")


if __name__ == "__main__":
    # Check for required environment variable
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ Error: AZURE_OPENAI_API_KEY environment variable is required")
        print("   Please set your Azure OpenAI API key:")
        print("   export AZURE_OPENAI_API_KEY='your-api-key-here'")
        print()
        print("   You can find your key in the Azure portal under:")
        print("   Your OpenAI Resource → Keys and Endpoint → Key 1")
        exit(1)
    
    asyncio.run(main())