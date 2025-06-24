# OpenAI-Compatible Chat Completions API

## Endpoint

```
POST /v1/chat/completions
```

## Description

This endpoint provides an OpenAI-compatible interface for chat completions.

## Authentication

```
Authorization: Bearer <API_KEY>
```

## Request Body

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model identifier. It could be your deployment name, adapter name, routing etc. |
| `messages` | array | Array of message objects forming the conversation |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float | Model default | Sampling temperature (0.0 to 2.0) |
| `max_tokens` | integer | Model default | Maximum tokens to generate |
| `max_completion_tokens` | integer | Model default | Alternative to max_tokens (OpenAI compatibility) |
| `top_p` | float | Model default | Nucleus sampling parameter |
| `frequency_penalty` | float | 0.0 | Penalize repeated tokens (-2.0 to 2.0) |
| `presence_penalty` | float | 0.0 | Penalize tokens based on presence (-2.0 to 2.0) |
| `seed` | integer | null | Random seed for reproducibility |
| `stream` | boolean | false | Enable streaming response |
| `stream_options` | object | null | Streaming configuration |
| `logprobs` | boolean | false | Return token log probabilities |
| `response_format` | object | null | Output format control |
| `tools` | array | null | Available tool/function definitions |
| `tool_choice` | string/object | "auto" | Tool selection strategy |
| `parallel_tool_calls` | boolean | true | Allow parallel tool calls |


### Additional fields

| Field | Type | Description |
|-------|------|-------------|
| `chat_template` | string | Custom chat template |
| `chat_template_kwargs` | object | Template parameters (e.g., `{"enable_thinking": true}`) |
| `mm_processor_kwargs` | object | Multi-modal processor parameters |
| `guided_json` | object | JSON schema for guided generation |
| `guided_regex` | string | Regex pattern for guided generation |
| `guided_choice` | array | List of allowed values |
| `guided_grammar` | string | Grammar for guided generation |
| `structural_tag` | string | Structural generation tag |
| `guided_decoding_backend` | string | Backend for guided decoding |
| `guided_whitespace_pattern` | string | Whitespace pattern for guided generation |

### Message Object Format

```json
{
  "role": "system" | "user" | "assistant" | "tool",
  "content": "string" | [{...}],  // String or array of content blocks
  "tool_calls": [...],             // For assistant messages
  "tool_call_id": "string"         // For tool messages
}
```

#### Content Block Types

**Text Content:**
```json
{
  "type": "text",
  "text": "Your message here"
}
```

**Image Content:**
```json
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/image.jpg"  // or data:image/jpeg;base64,...
  }
}
```

**File Content:**
```json
{
  "type": "file",
  "file": {
    "file_data": "data:application/pdf;base64,...",
    "filename": "document.pdf"
  }
}
```

## Response Format

### Standard Response

```json
{
  "id": "01977ed9-7492-7b70-8347-955764f97b3d",
  "episode_id": "01977ed9-7492-7b70-8347-9564aeb44a24",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Response text here",
        "tool_calls": [],
        "reasoning_content": "Optional reasoning/thinking content"
      },
      "logprobs": null
    }
  ],
  "created": 1750179872,
  "model": "qwen_3_4b",
  "system_fingerprint": "",
  "service_tier": "",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 133,
    "total_tokens": 145
  }
}
```

### Streaming Response

When `stream: true`, returns Server-Sent Events (SSE):

```
data: {"id":"...","object":"chat.completion.chunk","created":1750179872,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","created":1750179872,"model":"...","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","created":1750179872,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]
```

## Usage Examples

### Basic Chat Completion

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```


### With Tool/Function Calling

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "tensorzero::function_name::assistant",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather information",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

### With Guided JSON Generation

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "tensorzero::model_name::vllm_model",
    "messages": [
      {"role": "user", "content": "Extract person information"}
    ],
    "guided_json": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
      },
      "required": ["name", "age"]
    }
  }'
```

### With Streaming

```javascript
const response = await fetch('http://localhost:3000/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    model: 'qwen3-4b',
    messages: [{role: 'user', content: 'Tell me a story'}],
    stream: true,
    stream_options: {include_usage: true}
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') break;
      
      const parsed = JSON.parse(data);
      if (parsed.choices[0].delta.content) {
        process.stdout.write(parsed.choices[0].delta.content);
      }
    }
  }
}
```

### With Multi-modal Content

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "qwen2-7b-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            }
          }
        ]
      }
    ]
  }'
```

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique inference ID (UUID) |
| `episode_id` | string | Episode ID for grouped inferences |
| `choices` | array | Array of completion choices (usually 1) |
| `choices[].index` | integer | Choice index (always 0 for single completion) |
| `choices[].finish_reason` | string | Reason for completion: `stop`, `length`, `content_filter`, `tool_calls` |
| `choices[].message` | object | Generated message object |
| `choices[].message.role` | string | Always "assistant" |
| `choices[].message.content` | string/null | Generated text content |
| `choices[].message.tool_calls` | array | Tool/function calls if any |
| `choices[].message.reasoning_content` | string/null | Reasoning/thinking content |
| `choices[].logprobs` | object/null | Token log probabilities if requested |
| `created` | integer | Unix timestamp |
| `model` | string | Model identifier used |
| `usage` | object | Token usage statistics |
| `usage.prompt_tokens` | integer | Input token count |
| `usage.completion_tokens` | integer | Output token count |
| `usage.total_tokens` | integer | Total tokens used |

## Error Responses

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

Common error codes:
- `400` - Invalid request format or parameters
- `401` - Authentication failed
- `404` - Function or model not found
- `429` - Rate limit exceeded
- `500` - Internal server error
