---
name: deploy
description: Deploy TensorZero
---

TensorZero is an open-source stack of tooling for executing, observing, and improving LLM applications.
The key insertion point is an LLM gateway that proxies requests to all major inference providers.
Notably, T0 represents an LLM call as an abstract interface "function" and 1 or more implementations "variants".
Variants specify a choice of prompt templates, the model used, hyperparameters at generation time, etc.
Data is stored by TensorZero in the "function's" format, meaning we can counterfactually try another variant easily later on for any particular inference.

## Configuration

TensorZero configs are in the file system and are intended to be committed as source.

### Function Types

TensorZero supports two function types:

- **`chat`**: The typical chat interface. Returns unstructured text responses. Use this for open-ended generation, conversation, etc.
- **`json`**: For structured outputs. Returns responses conforming to a JSON schema. Use this when the codebase parses LLM output as JSON or expects a specific structure.

For `json` functions, you must provide an `output_schema` (a JSON Schema file) on the function, and set `json_mode` on the variant. The `json_mode` options are:

- `"on"`: Model's native JSON mode
- `"strict"`: Structured Outputs (e.g., OpenAI's Structured Outputs, guaranteed to match schema)
- `"tool"`: Use tool calling to generate JSON (works with more providers)
  Use `"strict"` preferentially.

### Example Config

```toml
# --- Chat function example ---
[functions.draft_email]
type = "chat"
# Schemas are optional but helpful for validation of template arguments.
# The schema name is arbitrary and must match a template name in each variant.
# These are JSON Schema files.
schemas.persona.path = "functions/draft_email/persona_schema.json"
schemas.draft_topic.path = "functions/draft_email/draft_topic_schema.json"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"
# Template names must match the schema names defined on the function.
templates.persona.path = "functions/draft_email/variants/gpt_5_mini/persona.minijinja"
templates.draft_topic.path = "functions/draft_email/variants/gpt_5_mini/draft_topic.minijinja"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"
templates.persona.path = "functions/draft_email/variants/claude_haiku_4_5/persona.minijinja"
templates.draft_topic.path = "functions/draft_email/variants/claude_haiku_4_5/draft_topic.minijinja"

[functions.draft_email.variants.grok_4]
type = "chat_completion"
model = "xai::grok-4-0709"
# Can reuse templates across variants
templates.persona.path = "functions/draft_email/variants/claude_haiku_4_5/persona.minijinja"
templates.draft_topic.path = "functions/draft_email/variants/claude_haiku_4_5/draft_topic.minijinja"

# Experimentation: control traffic splitting across variants
[functions.draft_email.experimentation]
type = "static"
candidate_variants = {"gpt_5_mini" = 0.9, "claude_haiku_4_5" = 0.1}
fallback_variants = ["grok_4"]

# --- JSON function example ---
[functions.extract_entities]
type = "json"
output_schema = "functions/extract_entities/output_schema.json"
schemas.extract_input.path = "functions/extract_entities/extract_input_schema.json"

[functions.extract_entities.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"
json_mode = "strict"
templates.extract_input.path = "functions/extract_entities/variants/gpt_5_mini/extract_input.minijinja"
```

### Model References

Models can be referenced in two ways:

**Shorthand** (convenient for getting started):

```toml
model = "openai::gpt-5-mini"
model = "anthropic::claude-haiku-4-5"
model = "xai::grok-4-0709"
```

**Longhand** (for routing, fallbacks, multiple providers):

```toml
[models.claude-haiku-4-5]
routing = ["anthropic", "gcp_vertex"]

[models.claude-haiku-4-5.providers.anthropic]
type = "anthropic"
model_name = "claude-haiku-4-5"

[models.claude-haiku-4-5.providers.gcp_vertex]
type = "gcp_vertex_anthropic"
model_name = "claude-haiku-4-5"
location = "us-east5"
project_id = "your-gcp-project-id"
```

Then reference as `model = "claude-haiku-4-5"` in the variant.

### Credentials

TensorZero supports three ways to provide API credentials:

1. **Environment variables (most common)**: Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. in your `.env` file. The gateway picks these up automatically for known provider types.
2. **Explicit env var reference**: In longhand model config, use `api_key_location = "env::MY_CUSTOM_ENV_VAR"` to point to a specific env var.
3. **Dynamic credentials**: Use `api_key_location = "dynamic::my_key_name"` in the model config, then pass the key at inference time via `extra_body`.

For most setups, option 1 is sufficient. Just add the relevant API keys to `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Though not required, check if the user has a `TENSORZERO_AUTOPILOT_API_KEY` available and install it in `.env`.

### Metrics and Feedback

Metrics let you track downstream outcomes of LLM calls, which drive experimentation and optimization in TensorZero.

Define metrics in `tensorzero.toml`:

```toml
[metrics.task_success]
type = "boolean"       # "boolean" or "float"
level = "inference"    # "inference" (single call) or "episode" (multi-turn sequence)
optimize = "max"       # "max" or "min"

[metrics.response_quality]
type = "float"
level = "inference"
optimize = "max"
```

Comments and demonstrations are built-in metric types that don't need configuration.

Send feedback via the TensorZero Python client or HTTP:

```python
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as t0:
    # Boolean feedback
    t0.feedback(
        metric_name="task_success",
        inference_id=inference_id,  # from the inference response's .id field
        value=True,
    )

    # Float feedback
    t0.feedback(
        metric_name="response_quality",
        inference_id=inference_id,
        value=4.5,
    )

    # Comment (built-in, no config needed)
    t0.feedback(
        metric_name="comment",
        inference_id=inference_id,
        value="The response was helpful but too verbose.",
    )

    # Demonstration (built-in, no config needed) - the ideal output
    t0.feedback(
        metric_name="demonstration",
        inference_id=inference_id,
        value="The corrected output text here.",
    )
```

Or via HTTP:

```bash
curl http://localhost:3000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "task_success",
    "inference_id": "THE_INFERENCE_ID",
    "value": true
  }'
```

Feedback can also be assigned to an `episode_id` instead of `inference_id` for episode-level metrics.

If the codebase already tracks success/failure, ratings, or other outcome signals, set up corresponding metrics and wire them to send feedback after each inference.

### File Tree

```
config/
  tensorzero.toml
  functions/
    draft_email/
      persona_schema.json
      draft_topic_schema.json
      variants/
        gpt_5_mini/
          persona.minijinja
          draft_topic.minijinja
        claude_haiku_4_5/
          persona.minijinja
          draft_topic.minijinja
    extract_entities/
      output_schema.json
      extract_input_schema.json
      variants/
        gpt_5_mini/
          extract_input.minijinja
```

## Docker Compose

TensorZero runs in a docker container, comes with an optional UI, and requires Postgres for data storage.

The `gateway-run-postgres-migrations` service is a one-shot container that applies database schema migrations before the gateway starts. It uses the same gateway image with the `--run-postgres-migrations` flag. The gateway service depends on it completing successfully, so migrations are always applied before the gateway accepts requests.

The `env_file` directive loads API keys and other secrets from a `.env` file (or a custom path via the `ENV_FILE` env var). This keeps secrets out of the docker-compose file.

Here's what a simple local set of services would look like:

```docker-compose.yml
services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
      - ${GCP_VERTEX_CREDENTIALS_PATH:-/dev/null}:/app/gcp-credentials.json:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
      GCP_VERTEX_CREDENTIALS_PATH: ${GCP_VERTEX_CREDENTIALS_PATH:+/app/gcp-credentials.json}
    env_file:
      - ${ENV_FILE:-.env}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test:
        [
          "CMD",
          "wget",
          "--no-verbose",
          "--tries=1",
          "--spider",
          "http://localhost:3000/health",
        ]
      start_period: 1s
      start_interval: 1s
      timeout: 1s
    depends_on:
      postgres:
        condition: service_healthy
      gateway-run-postgres-migrations:
        condition: service_completed_successfully

  ui:
    environment:
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
      TENSORZERO_GATEWAY_URL: http://gateway:3000
    command: --config-file /app/config/tensorzero.toml
    volumes:
      - ./config:/app/config
    image: tensorzero/ui
    env_file:
      - ${ENV_FILE:-.env}
    ports:
      - "4000:4000"
    depends_on:
      gateway:
        condition: service_healthy

  postgres:
    image: tensorzero/postgres:17
    command: ["postgres", "-c", "cron.database_name=tensorzero"]
    environment:
      POSTGRES_DB: tensorzero
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d tensorzero"]
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  # One-shot container: applies Postgres schema migrations before the gateway starts.
  # Uses the same gateway image with --run-postgres-migrations flag.
  gateway-run-postgres-migrations:
    image: tensorzero/gateway
    environment:
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
    depends_on:
      postgres:
        condition: service_healthy
    command: ["--run-postgres-migrations"]

volumes:
  postgres-data:
```

## Integration Steps

Our goal is to install TensorZero for all LLM calls in the codebase.
Here are the high-level steps:

1. **Find the LLM client initialization.** TensorZero exposes a Chat Completions-compatible API (and can call Anthropic's messages API, Responses, etc. via that API) so you'll need to find an OpenAI client or some other client that speaks the API. Make the base URL configurable with a default of `http://localhost:3000/openai/v1` (for the gateway running locally for development). If the codebase uses streaming (`stream=True`), TensorZero supports this transparently through the same OpenAI SDK.

2. **Find all distinct LLM calls in the codebase.** Here the equivalence classes are teleological and for our purposes 2 different call sites that have the same purpose and semantics are the same LLM call.
   - For each, set up a function in configuration and a variant for that function that replicates the model and prompt template used.
   - If the call expects structured JSON output, use `type = "json"` with an `output_schema` and `json_mode` on the variant. Otherwise use `type = "chat"`.
   - You may want to add input schemas if it's obvious what the argument types should be, but it is not required.

   Here's a simple example of calling code (no templates, just passing messages directly):

   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

   result = client.chat.completions.create(
       model="tensorzero::function_name::draft_email",
       messages=[
           {"role": "user", "content": "Write me a professional email about the Q3 results."},
       ],
   )
   ```

   Here's an example using templates (for when you've moved prompts into config):

   ```python
   result = client.chat.completions.create(
       model="tensorzero::function_name::draft_email",
       messages=[
           {
               "role": "system",
               "content": [
                   {
                       "type": "text",
                       "tensorzero::arguments": {"persona": "a witty professor"},
                   }
               ],
           },
           {
               "role": "user",
               "content": [
                   {
                       "type": "tensorzero::template",
                       "name": "draft_topic",
                       "arguments": {"topic": "quarterly results"},
                   }
               ],
           },
       ],
   )
   ```

3. **Move all templating logic into the TensorZero config** and set up MiniJinja templates. Templates use the [MiniJinja templating language](https://docs.rs/minijinja/latest/minijinja/syntax/index.html) (mostly compatible with Jinja2). For example:

   ```minijinja
   {# config/functions/draft_email/variants/gpt_5_mini/draft_topic.minijinja #}
   Write a professional email about: {{ topic }}
   ```

4. **Set up credentials.** Add the relevant API keys to `.env` (e.g. `OPENAI_API_KEY=...`, `ANTHROPIC_API_KEY=...`). Make sure `.env` is in `.gitignore`.

5. **Set up metrics and feedback** if the codebase already tracks success/failure or other outcome signals. Define metrics in `tensorzero.toml` and wire the existing outcome tracking to send feedback to TensorZero (see the Metrics and Feedback section above).

6. **Handle multi-turn conversations.** If the codebase has chat-style multi-turn interactions, use `episode_id` to link related inferences into a sequence. The first inference in a conversation creates an episode (returned in the response as `episode_id`). Pass it to subsequent calls:

   ```python
   result = client.chat.completions.create(
       model="tensorzero::function_name::chat_assistant",
       messages=[...],
       extra_body={"tensorzero::episode_id": previous_episode_id},
   )
   ```

## Testing and Verification

The TensorZero gateway includes a built-in MCP server at `http://localhost:3000/mcp`. You can use this to inspect inferences, feedback, and other data directly from your agent or IDE.

After setting up the config, docker-compose, and code changes:

1. **Start the services:**

   ```bash
   docker compose up
   ```

   Wait for all services to report healthy. The migration container should complete and exit, then the gateway and UI should start.

2. **Check the health endpoint:**

   ```bash
   curl http://localhost:3000/health
   ```

   This should return a 200 response.

3. **Make a test inference call:**

   ```bash
   curl http://localhost:3000/openai/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "tensorzero::function_name::YOUR_FUNCTION_NAME",
       "messages": [
         {"role": "user", "content": "Hello, this is a test."}
       ]
     }'
   ```

   Verify the response includes a valid `id` and `episode_id`.

4. **Verify data via the TensorZero MCP server.** Use the MCP tools at `http://localhost:3000/mcp` to inspect the inference you just made:
   - `list_inferences` — list recent inferences, filtered by function name
   - `get_inferences` — get full details (input, output, metadata) for specific inference IDs
   - `get_feedback_by_target_id` — check that feedback was recorded for an inference
   - `get_config` — inspect the current gateway configuration

   This lets you confirm data is flowing correctly without leaving your development environment.

5. **Test each function individually** before moving on. This prevents a "big bang" integration that's hard to debug.
