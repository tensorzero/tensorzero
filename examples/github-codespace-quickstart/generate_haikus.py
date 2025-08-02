import requests
import time

topics = ["moon", "coding", "coffee", "rain", "love", "time", "dreams", "ocean", "stars", "journey"]

for topic in topics:
    response = requests.post(
        "http://localhost:3000/inference",
        json={
            "function_name": "generate_haiku",
            "input": {
                "messages": [{"role": "user", "content": f"Write a haiku about {topic}"}]
            }
        }
    )
    print(f"Haiku about {topic}: {response.json()}")
    time.sleep(1)  # Be nice to the API