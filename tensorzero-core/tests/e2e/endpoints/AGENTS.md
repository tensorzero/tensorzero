# Endpoint E2E Tests

## Strongly typed requests and responses

Use the Rust types from `tensorzero_core` for request and response bodies instead of raw `json!()` macros. This catches type mismatches at compile time and keeps tests in sync with API changes.

Request types live in `tensorzero_core::endpoints::{module}::v1::types` (e.g. `CreateDatapointsRequest`, `CreateChatDatapointRequest`).
Response types are in the same module (e.g. `CreateDatapointsResponse`).
Input types like `Input`, `InputMessage`, `InputMessageContent`, `Text`, `Template`, `Arguments` are re-exported from `tensorzero_core::inference::types`.

Example:

```rust
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
    CreateDatapointsResponse,
};

let payload = CreateDatapointsRequest {
    datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
        function_name: "basic_test".to_string(),
        input: Input { ... },
        output: Some(vec![ContentBlockChatOutput::Text(Text { text: "...".to_string() })]),
        ..Default::default() // if available, otherwise set fields explicitly
    })],
};

let resp = client
    .post(get_gateway_endpoint("/v1/datasets/{name}/datapoints"))
    .json(&payload)
    .send()
    .await
    .unwrap();

let result: CreateDatapointsResponse = resp.json().await.unwrap();
```
