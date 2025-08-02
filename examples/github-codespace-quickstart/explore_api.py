import httpx
import requests
import asyncio
import json

async def debug_tensorzero():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:3000/inference",
            json={
                "function_name": "generate_haiku",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a haiku about AI"
                        }
                    ]
                }
            }
        )
        
        # Print the full response to see the structure
        print("Status Code:", response.status_code)
        print("Response:")
        print(json.dumps(response.json(), indent=2))

def explore_api():
    print("🔍 Exploring TensorZero API endpoints...\n")

    api_attempts = [
        ("POST", "http://localhost:3000/inference", {"function_name": "generate_haiku", "input": {"topic": "test"}}),
        ("GET", "http://localhost:3000/metrics"),
        ("GET", "http://localhost:3000/health"),
        ("GET", "http://localhost:3000/status"),
        ("GET", "http://localhost:4000/api/v1/inferences"),
        ("GET", "http://localhost:4000/inferences"),
        ("GET", "http://localhost:4000/api/gateway/inferences"),
    ]

    for method, url, *data in api_attempts:
        try:
            if method == "GET":
                response = requests.get(url, timeout=2)
            else:
                response = requests.post(url, json=data[0] if data else {}, timeout=2)
                
            print(f"{method} {url}")
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    content = response.json()
                    if isinstance(content, list):
                        print(f"  ✅ Found list with {len(content)} items")
                    elif isinstance(content, dict):
                        print(f"  ✅ Found dict with keys: {list(content.keys())[:5]}")
                except:
                    print(f"  Response: {response.text[:100]}...")
                    
        except Exception as e:
            print(f"{method} {url}")
            print(f"  ❌ Error: {type(e).__name__}")
        print()

if __name__ == "__main__":
    asyncio.run(debug_tensorzero())
    explore_api()