# AWS Bedrock Migration Plan: SDK to Direct HTTP

## Goal

Remove `aws_sdk_bedrockruntime` and call the Bedrock Converse API directly via HTTP using reqwest, similar to other providers.

---

## PoC Results (CONFIRMED)

### Non-streaming (`/converse`)

- **Endpoint**: `POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse`
- **Request Content-Type**: `application/json`
- **Response Content-Type**: `application/json`
- **Response**: Direct JSON, parse with serde

### Streaming (`/converse-stream`)

- **Endpoint**: `POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse-stream`
- **Request Content-Type**: `application/json`
- **Response Content-Type**: `application/vnd.amazon.eventstream` (binary, NOT SSE)
- **Accept header**: Doesn't matter - always returns binary event stream

### Binary Event Stream Format

Each frame has:

- Headers: `:event-type`, `:content-type`, `:message-type`
- Payload: JSON bytes

Parse with `aws_smithy_eventstream::frame::MessageFrameDecoder`:

```rust
let mut decoder = MessageFrameDecoder::new();
let mut buffer = BytesMut::from(data.as_ref());
while buffer.has_remaining() {
    match decoder.decode_frame(&mut buffer)? {
        DecodedFrame::Complete(message) => {
            // message.headers() -> Vec<Header>
            // message.payload() -> &Bytes (JSON)
        }
        DecodedFrame::Incomplete => break,
    }
}
```

Note: Header values are `StrBytes`, use `.as_str().to_owned()` to convert to String.

### Event Types & JSON Payloads

```
messageStart      → {"p":"...","role":"assistant"}
contentBlockDelta → {"contentBlockIndex":0,"delta":{"text":"chunk"},"p":"..."}
contentBlockStop  → {"contentBlockIndex":0,"p":"..."}
messageStop       → {"p":"...","stopReason":"end_turn"}
metadata          → {"metrics":{"latencyMs":N},"p":"...","usage":{...}}
```

Note: `p` field appears to be padding/nonce, ignore it.

### SigV4 Signing

```rust
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;

let signing_params = v4::SigningParams::builder()
    .identity(&identity)  // Identity from credentials.into()
    .region(region)
    .name("bedrock")      // Service name is "bedrock"
    .time(SystemTime::now())
    .settings(SigningSettings::default())
    .build()?;

let signable_request = SignableRequest::new(
    method,
    uri,
    headers.iter().map(|(k, v)| (k.as_str(), v.to_str().unwrap())),
    SignableBody::Bytes(&body),
)?;

let (instructions, _) = sign(signable_request, &signing_params.into())?.into_parts();
for (name, value) in instructions.headers() {
    request.headers_mut().insert(name, value);
}
```

### Request Body Structure

```json
{
  "modelId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
  "messages": [
    {
      "role": "user",
      "content": [{"text": "Hello"}]
    }
  ],
  "inferenceConfig": {
    "maxTokens": 100,
    "temperature": 0.7,
    "topP": 0.9,
    "stopSequences": ["END"]
  },
  "system": [{"text": "You are helpful"}],
  "toolConfig": {...}
}
```

### Response Body Structure (Non-streaming)

```json
{
  "output": {
    "message": {
      "role": "assistant",
      "content": [{ "text": "Hello!" }]
    }
  },
  "stopReason": "end_turn",
  "usage": {
    "inputTokens": 16,
    "outputTokens": 8,
    "totalTokens": 24,
    "cacheReadInputTokens": 0,
    "cacheWriteInputTokens": 0
  },
  "metrics": {
    "latencyMs": 500
  }
}
```

---

## Dependencies

**Keep:**

- `aws-config` - credential loading (IAM roles, profiles, env vars)
- `aws-sigv4` - SigV4 request signing
- `aws-credential-types` - credential types (with `hardcoded-credentials` feature)
- `aws-smithy-eventstream` - binary event stream parsing
- `aws-smithy-types` - shared AWS types (StrBytes, etc.)
- `aws-smithy-runtime-api` - Identity type

**Remove:**

- `aws_sdk_bedrockruntime` - replaced by direct HTTP

---

## Implementation Steps

### 1. Define Serde Types

Create `src/providers/aws_bedrock_types.rs`:

**Request types:**

- `BedrockConverseRequest`
- `BedrockMessage`, `BedrockContentBlock`
- `BedrockInferenceConfig`
- `BedrockToolConfig`, `BedrockTool`
- `BedrockSystemContent`

**Response types:**

- `BedrockConverseResponse`
- `BedrockOutput`, `BedrockUsage`, `BedrockMetrics`

**Streaming event types:**

- `BedrockStreamEvent` (enum)
- `MessageStartEvent`, `ContentBlockDeltaEvent`, `ContentBlockStopEvent`
- `MessageStopEvent`, `MetadataEvent`

### 2. Create Signing Module

Add to `aws_common.rs`:

```rust
pub async fn sign_request(
    request: &mut reqwest::Request,
    config: &SdkConfig,
    region: &str,
) -> Result<(), Error>
```

### 3. Update Provider Struct

```rust
pub struct AWSBedrockProvider {
    model_id: String,
    region: Region,
    base_url: String,  // Pre-computed endpoint URL
}

impl AWSBedrockProvider {
    pub async fn new(model_id: String, region: Option<Region>) -> Result<Self, Error> {
        let region = resolve_region(region).await?;
        let base_url = format!("https://bedrock-runtime.{}.amazonaws.com", region);
        Ok(Self { model_id, region, base_url })
    }
}
```

### 4. Implement `infer` (Non-streaming)

1. Convert `ModelInferenceRequest` → `BedrockConverseRequest`
2. Serialize to JSON
3. Build reqwest request with headers
4. Sign with SigV4
5. Send via `http_client`
6. Parse JSON response
7. Convert to `ProviderInferenceResponse`

### 5. Implement `infer_stream` (Streaming)

1. Convert request (same as above)
2. Sign and send to `/converse-stream`
3. Get response bytes stream
4. Create async stream that:
   - Accumulates bytes in `MessageFrameDecoder`
   - Yields `ProviderInferenceResponseChunk` for each event
5. Handle `metadata` event for final usage

### 6. Simplify `aws_common.rs`

- Remove `TensorZeroInterceptor` (no longer needed)
- Remove `InterceptorAndRawBody`
- Keep `config_with_region` for credential/region loading
- Add signing helper

### 7. Update Cargo.toml

```toml
# Remove
aws-sdk-bedrockruntime = ...

# Keep/Add
aws-config = { version = "1.8", features = ["behavior-version-latest"] }
aws-sigv4 = "1.3"
aws-credential-types = { version = "1.2", features = ["hardcoded-credentials"] }
aws-smithy-eventstream = "0.60"
aws-smithy-types = { version = "1.3", features = ["serde-deserialize", "serde-serialize"] }
aws-smithy-runtime-api = "1.7"
```

---

## Files to Modify

- `tensorzero-core/src/providers/aws_bedrock.rs` - Main rewrite
- `tensorzero-core/src/providers/aws_common.rs` - Simplify, add signing
- `tensorzero-core/src/providers/mod.rs` - Maybe add aws_bedrock_types
- `tensorzero-core/Cargo.toml` - Update deps

---

## Key Decisions

- **Credentials**: Fresh credentials for each request (handles rotation/expiration)
- **Extra fields**: Merge extra_body/extra_headers into request (same behavior, different mechanism)
- **SageMaker**: Leave alone - only migrate Bedrock
- **Thinking**: Preserve exact `additionalModelRequestFields` format
- **Streaming**: True incremental streaming (parse/yield as bytes arrive)

## Breaking Changes (Expected)

- `raw_request`: Will be the JSON request body (was interceptor-captured)
- `raw_response`: Will be full response body (was interceptor-captured)
- `raw_usage`: Extracted from response JSON

---

## Testing Strategy

1. Run existing tests without modification
2. Identify failures
3. Fix implementation where possible
4. Document intentional breaking changes

---

## Reference: PoC Code

See `tensorzero-core/src/bin/bedrock_poc.rs` for working implementation of:

- SigV4 signing
- Non-streaming request
- Streaming request with binary event stream parsing
