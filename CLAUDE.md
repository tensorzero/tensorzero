# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Building
```bash
cargo build                     # Development build
cargo build --release          # Production build
cargo build --workspace        # Build all workspace members
```

### Testing
```bash
cargo test                     # Run all tests
cargo test --workspace         # Run tests for all workspace members
cargo test test_name           # Run specific test
cargo test --lib              # Run only library tests
cargo test --package tensorzero-internal  # Test specific package
```

### Running the Gateway
```bash
# Development mode with custom config
cargo run --bin gateway -- --config-file test_tensorzero.toml

# Production mode
cargo run --release --bin gateway -- --config-file tensorzero.toml

# With environment variables
TENSORZERO_CLICKHOUSE_URL=... cargo run --bin gateway -- --config-file tensorzero.toml
```

### Code Quality
```bash
cargo fmt                      # Format code
cargo clippy -- -D warnings    # Lint with all warnings as errors
cargo check                    # Quick compilation check
```

## Architecture Overview

### Unified Model System

TensorZero uses a unified model configuration system where all models (chat, embedding, moderation) are configured in a single `models` table with endpoint capabilities:

```toml
[models."gpt-4"]
routing = ["primary_provider", "fallback_provider"]
endpoints = ["chat"]  # Capabilities: chat, embedding, moderation

[models."gpt-4".providers.primary_provider]
type = "openai"
model_name = "gpt-4"
api_key_location = { env = "OPENAI_API_KEY" }

[models."text-embedding-ada-002"]
routing = ["openai"]
endpoints = ["embedding"]

[models."omni-moderation-latest"]
routing = ["openai"]
endpoints = ["moderation"]
```

### Endpoint Structure

All OpenAI-compatible endpoints are implemented in `tensorzero-internal/src/endpoints/openai_compatible.rs`:

1. **Handler Function**: Processes the HTTP request
2. **Parameter Conversion**: Converts OpenAI format to internal format
3. **Model Resolution**: Resolves model name based on authentication
4. **Capability Check**: Verifies model supports the required endpoint
5. **Provider Routing**: Routes to appropriate provider based on model config
6. **Response Formatting**: Converts internal response to OpenAI format

### Adding New Endpoints (e.g., Audio)

To add new endpoints like `/v1/audio/transcriptions` or `/v1/audio/speech`:

1. **Define Capability** in `tensorzero-internal/src/endpoints/capability.rs`:
```rust
pub enum EndpointCapability {
    Chat,
    Embedding,
    Moderation,
    AudioTranscription,  // New
    AudioSpeech,         // New
}
```

2. **Add Route** in `gateway/src/main.rs`:
```rust
let openai_routes = Router::new()
    .route("/v1/chat/completions", post(endpoints::openai_compatible::inference_handler))
    .route("/v1/audio/transcriptions", post(endpoints::openai_compatible::audio_transcription_handler))  // New
    .route("/v1/audio/speech", post(endpoints::openai_compatible::audio_speech_handler));  // New
```

3. **Implement Handler** in `tensorzero-internal/src/endpoints/openai_compatible.rs`:
```rust
pub async fn audio_transcription_handler(
    State(app_state): AppState,
    headers: HeaderMap,
    StructuredJson(params): StructuredJson<OpenAIAudioTranscriptionParams>,
) -> Result<Response<Body>, Error> {
    // 1. Resolve model name
    // 2. Check model has AudioTranscription capability
    // 3. Route to provider
    // 4. Return OpenAI-compatible response
}
```

4. **Add Provider Support**:
   - Add trait method to provider (e.g., `transcribe()` for OpenAI provider)
   - Implement for each supporting provider
   - Handle provider-specific request/response formats

5. **Update Model Config**:
```toml
[models."whisper-1"]
routing = ["openai"]
endpoints = ["audio_transcription"]
```

### Provider Integration Pattern

Providers follow a consistent pattern:

1. **Trait Definition**: Define capability trait (e.g., `AudioTranscriptionProvider`)
2. **Implementation**: Provider-specific implementation in `inference/providers/`
3. **Request/Response Types**: Provider-specific types with conversion to/from internal types
4. **Error Handling**: Convert provider errors to internal error types
5. **Authentication**: Handle API keys via `InferenceCredentials`

### Authentication System

- Controlled by `gateway.authentication` in config
- When enabled, requires API key validation via Redis
- OpenAI routes use authentication middleware
- Internal routes remain public

### Caching Considerations

- Chat/embedding requests support caching via ClickHouse
- Moderation explicitly disables caching (see `cache_options` in moderation handler)
- Audio endpoints should consider whether caching makes sense

### Error Handling

All errors flow through the unified `Error` type with `ErrorDetails` enum. New endpoint-specific errors should be added to `ErrorDetails`.

### Testing Strategy

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test endpoint handlers with mock providers
3. **E2E Tests**: Full request/response cycle with test configs
4. **Provider Tests**: Mock provider responses for consistent testing

### Recent Changes

The moderation system was recently unified under the model system:
- Removed separate `moderation_models` configuration
- Moderation is now an endpoint capability like chat/embedding
- Models declare moderation support via `endpoints = ["moderation"]`
- Simplified configuration and reduced code duplication

## Key Principles

1. **Type Safety**: Use Rust's type system to prevent errors at compile time
2. **Unified Configuration**: All models use the same configuration structure
3. **Provider Abstraction**: Providers implement traits for capabilities they support
4. **OpenAI Compatibility**: Maintain API compatibility while using internal types
5. **Performance**: Keep latency <1ms P99 for gateway operations
6. **Observability**: Comprehensive logging, metrics, and tracing

## Common Patterns

### Adding a New Model Type
1. Add endpoint capability
2. Define request/response types
3. Implement handler
4. Add provider trait and implementations
5. Update router
6. Add tests

### Debugging
- Enable debug mode: `gateway.debug = true` in config
- Check logs for request/response details
- Use `tracing::debug!` for development logging
- Verify model capabilities match endpoint requirements

### Performance Considerations
- Minimize allocations in hot paths
- Use `Arc` for shared immutable data
- Stream responses when possible
- Cache expensive computations