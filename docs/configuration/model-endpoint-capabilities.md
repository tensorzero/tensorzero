# Model Endpoint Capabilities

## Overview

TensorZero uses an endpoint capability system to explicitly declare which types of operations each model supports. This provides better type safety, clearer configuration, and prevents runtime errors from attempting unsupported operations.

## Endpoint Capabilities

Currently supported endpoint capabilities:

- `chat` - For chat completion/text generation endpoints
- `embedding` - For text embedding generation endpoints

Future capabilities may include:
- `completions` - For text completion endpoints
- `images` - For image generation endpoints
- `audio` - For audio processing endpoints

## Configuration

### Basic Model Configuration

Models declare their capabilities using the `endpoints` array:

```toml
# Chat-only model
[models.gpt-4]
routing = ["openai"]
endpoints = ["chat"]

[models.gpt-4.providers.openai]
type = "openai"
model_name = "gpt-4"

# Embedding-only model
[models.text-embedding-3-small]
routing = ["openai"]
endpoints = ["embedding"]

[models.text-embedding-3-small.providers.openai]
type = "openai"
model_name = "text-embedding-3-small"

# Multi-capability model (supports both chat and embeddings)
[models.voyage-3]
routing = ["voyage"]
endpoints = ["chat", "embedding"]

[models.voyage-3.providers.voyage]
type = "voyage"
model_name = "voyage-3"
```

### Default Behavior

If the `endpoints` field is omitted, the model defaults to supporting only `chat`:

```toml
# This model implicitly supports only "chat"
[models.claude-3]
routing = ["anthropic"]
# endpoints = ["chat"]  # This is the default

[models.claude-3.providers.anthropic]
type = "anthropic"
model_name = "claude-3-5-sonnet-latest"
```

## Migration from Legacy Configuration

### Old Configuration (Deprecated)

Previously, TensorZero used separate tables for different model types:

```toml
# Old style - separate tables
[models.gpt-4]
routing = ["openai"]
# ... provider config

[embedding_models.text-embedding-3-small]
routing = ["openai"]
# ... provider config
```

### New Configuration (Recommended)

The new unified configuration uses a single `models` table with explicit capabilities:

```toml
# New style - unified table with endpoints
[models.gpt-4]
routing = ["openai"]
endpoints = ["chat"]
# ... provider config

[models.text-embedding-3-small]
routing = ["openai"]
endpoints = ["embedding"]
# ... provider config
```

### Automatic Migration

TensorZero automatically migrates old configurations:
- Models in the `[models]` table receive `endpoints = ["chat"]`
- Models in the `[embedding_models]` table receive `endpoints = ["embedding"]`
- The `[embedding_models]` table is deprecated but still supported for backward compatibility

## Usage in Variants

When configuring variants, ensure the selected model supports the required endpoint:

### Chat Variant

```toml
[functions.chat_function.variants.my_variant]
type = "chat_completion"
model = "gpt-4"  # Must support "chat" endpoint
# ... other config
```

### Embedding Variant (DICL)

```toml
[functions.rag_function.variants.dicl_variant]
type = "experimental_dynamic_in_context_learning"
model = "gpt-4"  # Must support "chat" endpoint
embedding_model = "text-embedding-3-small"  # Must support "embedding" endpoint
# ... other config
```

## Error Handling

### Capability Mismatch Errors

If you attempt to use a model for an unsupported operation, you'll receive a clear error:

```
Error: CapabilityNotSupported {
    capability: "embedding",
    provider: "gpt-4"
}
```

This might occur when:
- Using a chat-only model for embeddings
- Using an embedding-only model for chat completion
- Configuring a DICL variant with incompatible models

### Common Issues and Solutions

1. **Model not found for embedding operations**
   ```
   Error: Config { message: "Model 'gpt-4' not found or does not support endpoint 'embedding'" }
   ```
   **Solution**: Ensure the model has `endpoints = ["embedding"]` or use a dedicated embedding model.

2. **DICL variant configuration error**
   ```
   Error: The embedding_model must support the 'embedding' endpoint
   ```
   **Solution**: Use a model with embedding capabilities for the `embedding_model` field.

## Best Practices

1. **Explicit Declaration**: Always explicitly declare endpoints for clarity:
   ```toml
   [models.my-model]
   endpoints = ["chat"]  # Be explicit even for chat-only models
   ```

2. **Validate Early**: The configuration parser validates endpoint capabilities at startup, catching errors before runtime.

3. **Use Appropriate Models**: Select models designed for their specific tasks:
   - Use dedicated embedding models for embedding tasks (better performance, lower cost)
   - Use chat models for text generation tasks

4. **Multi-Capability Models**: Some providers offer models that support multiple endpoints. Declare all supported capabilities:
   ```toml
   [models.versatile-model]
   endpoints = ["chat", "embedding"]
   ```

## Provider Support

Different providers support different endpoint capabilities:

| Provider | Chat | Embeddings | Notes |
|----------|------|------------|-------|
| OpenAI | ✅ | ✅ | Separate models for each capability |
| Anthropic | ✅ | ❌ | Chat only |
| Azure | ✅ | ✅ | Via OpenAI models |
| AWS Bedrock | ✅ | ✅ | Depends on model |
| vLLM | ✅ | ✅ | Depends on loaded model |
| Voyage | ❌ | ✅ | Embeddings only |

## API Integration

### Model Table API

The model table provides methods to query models by capability:

```rust
// Get all models supporting embeddings
let embedding_models = model_table.get_models_for_capability(EndpointCapability::Embedding);

// Get a specific model if it supports embeddings
let model = model_table.get_with_capability("text-embedding-3-small", EndpointCapability::Embedding).await?;
```

### Type Safety

The capability system provides compile-time and runtime safety:
- Configuration validation at startup
- Type-safe provider implementations
- Clear error messages for capability mismatches

## Future Extensions

The endpoint capability system is designed to be extensible:

1. **New Capabilities**: Additional capabilities can be added to the `EndpointCapability` enum
2. **Provider Evolution**: Providers can declare support for new capabilities as they become available
3. **Fine-Grained Control**: Future versions may support capability-specific configuration options

## Example Configurations

### Simple Chat Application

```toml
[models.gpt-4]
routing = ["openai"]
endpoints = ["chat"]

[models.gpt-4.providers.openai]
type = "openai"
model_name = "gpt-4"

[functions.assistant]
type = "chat"

[functions.assistant.variants.default]
type = "chat_completion"
model = "gpt-4"
```

### RAG Application with Embeddings

```toml
# Chat model for generation
[models.gpt-4]
routing = ["openai"]
endpoints = ["chat"]

[models.gpt-4.providers.openai]
type = "openai"
model_name = "gpt-4"

# Embedding model for retrieval
[models.text-embedding-3-small]
routing = ["openai"]
endpoints = ["embedding"]

[models.text-embedding-3-small.providers.openai]
type = "openai"
model_name = "text-embedding-3-small"

[functions.rag_query]
type = "chat"

[functions.rag_query.variants.dicl]
type = "experimental_dynamic_in_context_learning"
model = "gpt-4"
embedding_model = "text-embedding-3-small"
```

### Multi-Provider Setup

```toml
# OpenAI Chat
[models.gpt-4]
routing = ["openai"]
endpoints = ["chat"]

[models.gpt-4.providers.openai]
type = "openai"
model_name = "gpt-4"

# Anthropic Chat  
[models.claude-3]
routing = ["anthropic"]
endpoints = ["chat"]

[models.claude-3.providers.anthropic]
type = "anthropic"
model_name = "claude-3-5-sonnet-latest"

# OpenAI Embeddings
[models.text-embedding-3-small]
routing = ["openai"]
endpoints = ["embedding"]

[models.text-embedding-3-small.providers.openai]
type = "openai"
model_name = "text-embedding-3-small"

# Voyage Embeddings (alternative)
[models.voyage-3]
routing = ["voyage"]
endpoints = ["embedding"]

[models.voyage-3.providers.voyage]
type = "voyage"
model_name = "voyage-3"
```