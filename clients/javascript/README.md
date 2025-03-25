# TensorZero JavaScript Client

**[Website](https://www.tensorzero.com/)** ·
**[Docs](https://www.tensorzero.com/docs)** ·
**[Twitter](https://www.x.com/tensorzero)** ·
**[Slack](https://www.tensorzero.com/slack)** ·
**[Discord](https://www.tensorzero.com/discord)**

**[Quick Start (5min)](https://www.tensorzero.com/docs/quickstart)** ·
**[Comprehensive Tutorial](https://www.tensorzero.com/docs/gateway/tutorial)** ·
**[Deployment Guide](https://www.tensorzero.com/docs/gateway/deployment)** ·
**[API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference)** ·
**[Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference)**

The `tensorzero` npm package provides a JavaScript client for the TensorZero Gateway.
This client allows you to easily make inference requests and assign feedback to them via the gateway.

See our **[API Reference](https://www.tensorzero.com/docs/gateway/api-reference)** for more information.

## Installation

```bash
npm install tensorzero
# or
yarn add tensorzero
# or
pnpm add tensorzero
```

## Basic Usage

### Initialization

The TensorZero client provides both Promise-based and async/await patterns for usage, and supports connecting to an external HTTP gateway.

#### HTTP Gateway Connection

```typescript
import { TensorZeroGateway } from 'tensorzero';

// Create a client instance
const client = new TensorZeroGateway({
  gatewayUrl: 'http://localhost:3000'
});

// Use the client
// ...
```

### Inference

#### Non-Streaming Inference

```typescript
import { TensorZeroGateway } from 'tensorzero';

const client = new TensorZeroGateway({
  gatewayUrl: 'http://localhost:3000'
});

// Using async/await
async function run() {
  const response = await client.inference({
    modelName: 'openai::gpt-4o-mini',
    input: {
      messages: [
        { role: 'user', content: 'What is the capital of Japan?' },
      ],
    },
  });

  console.log(response);
}

// Using promises
client.inference({
  modelName: 'openai::gpt-4o-mini',
  input: {
    messages: [
      { role: 'user', content: 'What is the capital of Japan?' },
    ],
  },
})
.then(response => {
  console.log(response);
})
.catch(error => {
  console.error(error);
});
```

#### Streaming Inference

```typescript
import { TensorZeroGateway } from 'tensorzero';

const client = new TensorZeroGateway({
  gatewayUrl: 'http://localhost:3000'
});

// Using async/await
async function run() {
  const stream = await client.inferenceStream({
    modelName: 'openai::gpt-4o-mini',
    input: {
      messages: [
        { role: 'user', content: 'What is the capital of Japan?' },
      ],
    },
  });

  for await (const chunk of stream) {
    console.log(chunk);
  }
}

// Using promise-based event handling
client.inferenceStream({
  modelName: 'openai::gpt-4o-mini',
  input: {
    messages: [
      { role: 'user', content: 'What is the capital of Japan?' },
    ],
  },
})
.then(stream => {
  stream.on('data', chunk => {
    console.log(chunk);
  });
  
  stream.on('end', () => {
    console.log('Stream ended');
  });
  
  stream.on('error', error => {
    console.error(error);
  });
});
```

### Feedback

```typescript
import { TensorZeroGateway } from 'tensorzero';

const client = new TensorZeroGateway({
  gatewayUrl: 'http://localhost:3000'
});

// Using async/await
async function sendFeedback() {
  const response = await client.feedback({
    metricName: 'thumbs_up',
    inferenceId: '00000000-0000-0000-0000-000000000000',
    value: true, // 👍
  });

  console.log(response);
}

// Using promises
client.feedback({
  metricName: 'thumbs_up',
  inferenceId: '00000000-0000-0000-0000-000000000000',
  value: true, // 👍
})
.then(response => {
  console.log(response);
})
.catch(error => {
  console.error(error);
});
``` 