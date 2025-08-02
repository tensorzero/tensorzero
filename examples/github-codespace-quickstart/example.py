import httpx
import asyncio
import json

async def generate_haiku_example():
    # Create an HTTP client
    async with httpx.AsyncClient() as client:
        # Call TensorZero to generate a haiku
        response = await client.post(
            "http://localhost:3000/inference",
            json={
                "function_name": "generate_haiku",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a haiku about artificial intelligence"
                        }
                    ]
                }
            }
        )
        
        # Get the response
        result = response.json()
        
        # Print the haiku
        print("Generated Haiku:")
        print("-" * 40)
        print(result["content"][0]["content"])
        print("-" * 40)
        
        # Print the inference ID (you can look this up in the UI)
        print(f"\nInference ID: {result['inference_id']}")
        print(f"Model used: {result['model_name']}")

# Run the example
asyncio.run(generate_haiku_example())