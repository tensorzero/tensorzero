# JavaScript Client for TensorZero

## Overview
This PR adds a JavaScript/TypeScript client for TensorZero, similar to the existing Python client. The client allows JavaScript/TypeScript developers to easily interact with the TensorZero gateway by providing a simple and intuitive API.

## Features
- Unified interface for connecting to the TensorZero gateway
- Support for both streaming and non-streaming inference
- Support for providing feedback on inferences
- Comprehensive TypeScript type definitions
- Error handling and proper stream processing
- Unit tests for core functionality
- Example code demonstrating usage

## Implementation Details
The JavaScript client is implemented using TypeScript and includes:
- A client class that handles communication with the gateway
- Type definitions for requests and responses
- Support for both Promise-based and async/await patterns
- SSE parsing for streaming responses
- Axios for HTTP requests

## Testing
Tests are implemented using Jest and cover the core functionality of the client.

## Example Usage
The client can be used as follows:

```typescript
import { TensorZeroGateway } from 'tensorzero';

const client = new TensorZeroGateway({
  gatewayUrl: 'http://localhost:3000',
});

// Non-streaming inference
const response = await client.inference({
  modelName: 'openai::gpt-4o-mini',
  input: {
    messages: [
      { role: 'user', content: 'What is the capital of Japan?' },
    ],
  },
});

// Streaming inference
const stream = await client.inferenceStream({
  modelName: 'openai::gpt-4o-mini',
  input: {
    messages: [
      { role: 'user', content: 'Tell me about the solar system.' },
    ],
  },
});

for await (const chunk of stream) {
  console.log(chunk.chunk.content);
}
```

## Next Steps
Future work could include:
- Adding automatic retries for failed requests
- Supporting batch inference
- Implementing middleware for custom request/response processing
- Adding browser-specific optimizations for web applications 