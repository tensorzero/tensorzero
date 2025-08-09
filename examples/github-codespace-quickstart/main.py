# Create main.py
import httpx
import asyncio

async def test_tensorzero():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:3000/inference",
            json={
                "function_name": "generate_haiku",
                "input": {
                    "messages": [
                        {"role": "user", "content": "Write a haiku about coding"}
                    ]
                }
            }
        )
        print(response.json())

asyncio.run(test_tensorzero())
