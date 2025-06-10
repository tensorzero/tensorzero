# Redis and Authentication Integration in TensorZero

## Overview

TensorZero has integrated Redis for dynamic model configuration and API key-based authentication, enabling multi-tenant support and runtime configuration updates without service restarts. This integration allows the system to serve different models to different users based on their API keys.

## Architecture Components

### 1. Redis Client (`tensorzero-internal/src/redis_client.rs`)

The Redis client manages real-time synchronization of model configurations and API keys:

#### Key Patterns
- `model_table:{model_id}` - Stores model provider configurations
- `api_key:{api_key}` - Stores API key to model mappings

#### Event Subscriptions
- `__keyevent@*__:set` - Captures new/updated keys
- `__keyevent@*__:del` - Captures deleted keys
- `__keyevent@*__:expired` - Captures expired keys

#### Initialization Flow
1. On startup, fetches all existing `model_table:*` and `api_key:*` keys
2. Subscribes to keyspace notifications for real-time updates
3. Maintains connection with automatic reconnection logic

### 2. Authentication Middleware (`tensorzero-internal/src/auth.rs`)

The authentication system validates API keys and maps user-facing model names to internal model IDs:

#### Request Flow
1. Extracts API key from `Authorization` header (supports "Bearer" prefix)
2. Validates key exists in the auth state
3. Retrieves model mapping for the API key
4. Modifies request body: replaces `model` field with `tensorzero::model_name::{model_id}`
5. Passes modified request to downstream handlers

#### Key Features
- Thread-safe with RwLock for concurrent access
- Dynamic updates without service restart
- Graceful handling of missing keys

### 3. Model Table (`tensorzero-internal/src/model_table.rs`)

Manages model configurations and routing:

#### Model Configuration Structure
```json
{
  "model-id": {
    "routing": ["provider1", "provider2"],
    "providers": {
      "provider1": {
        "type": "vllm",
        "model_name": "actual-model-name",
        "api_base": "http://endpoint",
        "api_key_location": "none"
      }
    }
  }
}
```

#### Validation
- Prevents use of reserved prefixes (`tensorzero::`)
- Validates model exists before returning configuration
- Supports shorthand notation (e.g., `openai::gpt-4`)

### 4. OpenAI-Compatible Endpoint (`tensorzero-internal/src/endpoints/openai_compatible.rs`)

Handles the model resolution after authentication:

#### Model Name Parsing
- Detects `tensorzero::model_name::{model_id}` pattern
- Extracts model ID for table lookup
- Falls back to function name resolution if not a model reference

## Data Flow

### 1. Configuration Update Flow
```
Redis SET → Keyspace Event → Redis Client → Update In-Memory State
```

1. External system sets key in Redis (e.g., `api_key:xyz` or `model_table:abc`)
2. Redis publishes keyspace notification
3. Redis client receives event and fetches updated value
4. Updates appropriate in-memory state (Auth or ModelTable)

### 2. Request Authentication Flow
```
Client Request → Auth Middleware → Model Resolution → Provider
```

1. Client sends request with API key in Authorization header
2. Auth middleware validates key and retrieves model mapping
3. Modifies request to include internal model ID
4. OpenAI endpoint resolves model ID to provider configuration
5. Routes request to appropriate provider

## Redis Data Structures

### API Key Structure
```json
{
  "api_key_here": {
    "user-facing-model-name": "internal-model-id",
    "another-model": "another-model-id"
  }
}
```

### Model Table Structure
```json
{
  "model-id": {
    "routing": ["primary-provider", "fallback-provider"],
    "providers": {
      "primary-provider": {
        "type": "provider-type",
        "model_name": "actual-model-name",
        "api_base": "http://provider-endpoint",
        "api_key_location": "header|none"
      }
    }
  }
}
```

## Configuration

### Environment Variables
- `TENSORZERO_REDIS_URL` - Redis connection string (e.g., `redis://default:password@localhost:6379`)

### Gateway Integration
The gateway initializes Redis when the environment variable is set:

```rust
if let Ok(redis_url) = std::env::var("TENSORZERO_REDIS_URL") {
    if !redis_url.is_empty() {
        let redis_client = RedisClient::new(&redis_url, app_state.clone(), auth.clone()).await;
        redis_client.start().await.expect_pretty("Failed to start Redis client");
    }
}
```

## Testing

A test script (`redis-test.py`) is provided for manual testing:

```python
import redis
import json

# Connect to Redis
r = redis.Redis.from_url("redis://default:budpassword@localhost:6378")

# Set model configuration
models = {
    "model-id": {
        "routing": ["vllm"],
        "providers": {
            "vllm": {
                "type": "vllm",
                "model_name": "actual-model",
                "api_base": "http://endpoint/v1",
                "api_key_location": "none"
            }
        }
    }
}
r.set("model_table:model-id", json.dumps(models))

# Set API key mapping
api_keys = {
    "api_key_here": {
        "gpt-4o": "model-id"
    }
}
r.set("api_key:api_key_here", json.dumps(api_keys))
```

## Benefits

1. **Dynamic Configuration**: Add/remove models and API keys without restarting the service
2. **Multi-Tenancy**: Different API keys can access different models
3. **Real-time Updates**: Changes take effect immediately via Redis pub/sub
4. **Scalability**: Multiple gateway instances can share the same Redis backend
5. **Isolation**: Each tenant's model access is isolated by their API key

## Security Considerations

1. **API Key Storage**: Keys are stored in Redis and should be properly secured
2. **Redis Access**: Ensure Redis is not publicly accessible
3. **TLS**: Use Redis TLS connections in production
4. **Key Rotation**: API keys can be rotated by updating Redis entries

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check `TENSORZERO_REDIS_URL` is correctly formatted
   - Verify Redis is running and accessible
   - Check network connectivity

2. **API Key Not Working**
   - Verify key exists in Redis with correct format
   - Check model ID mapping is valid
   - Ensure Redis events are enabled (`notify-keyspace-events` config)

3. **Model Not Found**
   - Verify model table entry exists in Redis
   - Check model ID matches between API key mapping and model table
   - Validate JSON structure in Redis

### Debug Commands

```bash
# Check Redis keys
redis-cli KEYS "api_key:*"
redis-cli KEYS "model_table:*"

# Get specific key value
redis-cli GET "api_key:your_key_here"

# Monitor Redis events
redis-cli PSUBSCRIBE "__keyevent@*__:*"
```

## Future Enhancements

1. **Metrics**: Track API key usage and model request patterns
2. **Rate Limiting**: Per-API-key rate limits stored in Redis
3. **Caching**: Cache model responses in Redis for common queries
4. **Audit Logging**: Log all API key usage for compliance
5. **Key Expiration**: Support automatic API key expiration