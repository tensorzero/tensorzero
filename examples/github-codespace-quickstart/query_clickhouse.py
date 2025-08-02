# query_clickhouse.py
import requests
import json

def query_clickhouse(query):
    """Query ClickHouse directly"""
    try:
        response = requests.post(
            'http://localhost:8123/',
            data=query,
            headers={'X-ClickHouse-Format': 'JSONEachRow'}
        )
        
        if response.status_code == 200:
            # Parse JSON lines
            lines = response.text.strip().split('\n')
            results = [json.loads(line) for line in lines if line]
            return results
        else:
            print(f"Query failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error querying ClickHouse: {e}")
        return None

# Get recent inferences
print("📊 Querying ClickHouse for inferences...\n")

queries = [
    ("Show databases", "SHOW DATABASES"),
    ("Show tables", "SHOW TABLES FROM tensorzero"),
    ("Count inferences", "SELECT COUNT(*) as count FROM tensorzero.InferenceModelInference"),
    ("Recent inferences", """
        SELECT 
            id,
            inference_id,
            function_name,
            substring(output, 1, 100) as output_preview,
            created_at
        FROM tensorzero.InferenceModelInference 
        ORDER BY created_at DESC 
        LIMIT 5
    """),
]

for name, query in queries:
    print(f"\n{name}:")
    print("-" * 50)
    results = query_clickhouse(query)
    if results:
        for row in results:
            print(json.dumps(row, indent=2, default=str))



#     🔍 Exploring TensorZero API endpoints...

# POST http://localhost:3000/inference
#   Status: 400

# GET http://localhost:3000/metrics
#   Status: 200
#   Response: # TYPE request_count counter
# request_count{endpoint="feedback",metric_name="comment"} 5
# request_coun...

# GET http://localhost:3000/health
#   Status: 200
#   ✅ Found dict with keys: ['gateway', 'clickhouse']

# GET http://localhost:3000/status
#   Status: 200
#   ✅ Found dict with keys: ['status', 'version']

# GET http://localhost:4000/api/v1/inferences
#   Status: 404

# GET http://localhost:4000/inferences
#   Status: 404

# GET http://localhost:4000/api/gateway/inferences
#   Status: 404
