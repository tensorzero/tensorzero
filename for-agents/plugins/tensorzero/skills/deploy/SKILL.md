---
name: deploy
description: Deploy TensorZero
---

TensorZero is an open-source stack of tooling for executing, observing, and improving LLM applications.
The key insertion point is an LLM gateway that proxies requests to all major inference providers.
Notably, T0 represents an LLM call as an abstract interface "function" and 1 or more implementations "variants".
Variants specify a choice of prompt templates, the model used, hyperparameters at generation time, etc.
Data is stored by TensorZero in the "function's" format, meaning we can counterfactually try another variant easily later on for any particular inference.

TensorZero configs are in the file system and are intended to be committed as source.
Typically, they look like:

```toml
[functions.draft_email]
type = "chat" # required, this could also be json but don't use it
# Schemas are optional but helpful for validation that the data passed in is correct
# These are Json Schema
schemas.system.path = "functions/draft_email/system.json"
schemas.first_user_message.path = "functions/draft_email/first_user_message.json"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"
templates.system.path = "functions/draft_email/variants/gpt_5_mini/system.minijinja"
templates.first_user_message.path = "functions/draft_email/variants/gpt_5_mini/first_user.minijinja"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "claude-haiku-4-5"
templates.system.path = "functions/draft_email/variants/claude_haiku_4_5/system.minijinja"
templates.first_user_message.path = "functions/draft_email/variants/claude_haiku_4_5/first_user.minijinja"

# Models can also be defined in "longhand" to configure routing, fallback providers, etc.
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

[functions.draft_email.variants.grok_4]
type = "chat_completion"
model = "xai::grok-4-0709"
# can use the same templates in multiple variants
templates.system.path = "functions/draft_email/variants/claude_haiku_4_5/system.minijinja"
templates.first_user_message.path = "functions/draft_email/variants/claude_haiku_4_5/first_user.minijinja"

[functions.draft_email.experimentation]
type = "static"
candidate_variants = {"gpt_5_mini" = 0.9, "claude_haiku_4_5" = 0.1}
fallback_variants = ["grok_4"]
```

The file tree would then look like:

```
config/
  |
  --- tensorzero.toml
  --- functions/
      |
      --- draft_email/
            system.json
              ...
```

TensorZero runs in a docker container, comes with an optional UI, and requires Postgres for data storage. Here's what a simple local set of services would look like:

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

  # Apply Postgres migrations before the gateway starts
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

Our goal is to install TensorZero for all LLM calls in the codebase.
Here are the high-level steps:

1. Find the place where the LLM client is initialized. TensorZero exposes a Chat Completions-compatible API (and can call Anthropic's messages API, Responses, etc via that API) so you'll need to find an OpenAI client or some other client that speaks the API. Make the base URL configurable with a default of http://localhost:3000/openai/v1 (for the gateway running locally for development)
1. Find all distinct LLM calls in the codebase. Here the equivalence classes are teleological and for our purposes 2 different call sites that have the same purpose and semantics are the same LLM call.
   a. For each, set up a function in configuration and a variant for that function that replicates the model and prompt template used. You may want to add a json schema if it's obvious what the type should be but it is not required.
   Here's an example of calling code:

```
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/openai/v1", api_key="not-used")

result = client.chat.completions.create(
    model="tensorzero::function_name::fun_fact",
    messages=[
        {
            "role": "system",
            "content": [  # type: ignore
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
                    "type": "tensorzero::template",  # type: ignore
                    "name": "fun_fact_topic",
                    "arguments": {"topic": "artificial intelligence"},
                }
            ],
        },
    ],
)
```

1. Move all templating logic into the TensorZero config and set up a minijinja template.
1. Make sure TensorZero has appropriate credentials. Typically these are set in `.env` with `OPENAI_API_KEY=...`, `ANTHROPIC_API_KEY=...`. Though not required, check if the user has a `TENSORZERO_AUTOPILOT_API_KEY` available and install it in `.env`.
