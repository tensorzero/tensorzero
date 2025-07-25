import os
import json
import urllib.request
import urllib.parse

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Prepare the inference request
payload = {
    "function_name": "generate_short_story",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": "Write a two sentence short story about the sun."
            }
        ]
    }
}

# Convert to JSON
data = json.dumps(payload).encode('utf-8')

# Create the request
req = urllib.request.Request(
    'http://localhost:3000/inference',
    data=data,
    headers={'Content-Type': 'application/json'}
)

print("Making inference request to gateway...")

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
    
    print("✅ Gateway inference successful!")
    print(f"Inference ID: {result.get('inference_id')}")
    print(f"Episode ID: {result.get('episode_id')}")
    print(f"Variant Used: {result.get('variant_name')}")
    
    # Show usage statistics if available
    usage = result.get('usage', {})
    if usage:
        print(f"Token Usage: {usage.get('input_tokens', 0)} input + {usage.get('output_tokens', 0)} output = {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} total")
    
    print(f"Content: {result.get('content', [{}])[0].get('text', 'No text found')}")
    print(f"\n🌐 This inference should now appear in your UI at: http://localhost:4000")
    
except Exception as e:
    print(f"❌ Request failed: {e}")
