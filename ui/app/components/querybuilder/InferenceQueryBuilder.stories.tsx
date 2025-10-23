import type { Meta, StoryObj } from "@storybook/react-vite";
import InferenceQueryBuilder from "./InferenceQueryBuilder";
import { ConfigProvider } from "~/context/config";
import type { FunctionConfig, Config } from "tensorzero-node";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { FILLED_INFERENCE_FILTER } from "./InferenceFilterBuilder.stories";

const meta = {
  title: "QueryBuilder/InferenceQueryBuilder",
  component: InferenceQueryBuilder,
  decorators: [
    (Story) => {
      return (
        <div className="border-border w-2xl rounded border p-4">
          <Story />
        </div>
      );
    },
  ],
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof InferenceQueryBuilder>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockFunctions: Record<string, FunctionConfig> = {
  [DEFAULT_FUNCTION]: {
    type: "chat",
    variants: {},
    schemas: {},
    tools: [],
    tool_choice: "auto",
    parallel_tool_calls: null,
    description: "Default chat function",
    experimentation: { type: "uniform" },
  },
  "chat-function": {
    type: "chat",
    variants: {},
    schemas: {},
    tools: ["calculator", "weather"],
    tool_choice: "auto",
    parallel_tool_calls: true,
    description: "Chat function with tools",
    experimentation: { type: "uniform" },
  },
  "json-extractor": {
    type: "json",
    variants: {},
    schemas: {},
    output_schema: {
      value: null,
    },
    implicit_tool_call_config: {
      tools_available: [],
      provider_tools: null,
      tool_choice: "auto",
      parallel_tool_calls: false,
    },
    description: "Extract structured data from text",
    experimentation: { type: "uniform" },
  },
};

const mockConfig: Config = {
  gateway: {
    disable_pseudonymous_usage_analytics: false,
    fetch_and_encode_input_files_before_inference: false,
    observability: {
      enabled: true,
      async_writes: false,
      batch_writes: {
        enabled: false,
        __force_allow_embedded_batch_writes: false,
        flush_interval_ms: 100n,
        max_rows: 1000,
      },
      disable_automatic_migrations: false,
    },
    export: {
      otlp: {
        traces: {
          enabled: false,
          format: "opentelemetry",
          extra_headers: {},
        },
      },
    },
    debug: false,
    template_filesystem_access: {
      enabled: false,
      base_path: null,
    },
    bind_address: "localhost:8080",
    base_path: "/",
    unstable_error_json: false,
    unstable_disable_feedback_target_validation: false,
  },
  object_store_info: { kind: { type: "disabled" } },
  provider_types: {
    anthropic: { defaults: { api_key_location: "" } },
    azure: { defaults: { api_key_location: "" } },
    deepseek: { defaults: { api_key_location: "" } },
    fireworks: { defaults: { api_key_location: "" } },
    gcp_vertex_gemini: { batch: null, defaults: { credential_location: "" } },
    gcp_vertex_anthropic: {
      batch: null,
      defaults: { credential_location: "" },
    },
    google_ai_studio_gemini: { defaults: { api_key_location: "" } },
    groq: { defaults: { api_key_location: "" } },
    hyperbolic: { defaults: { api_key_location: "" } },
    mistral: { defaults: { api_key_location: "" } },
    openai: { defaults: { api_key_location: "" } },
    openrouter: { defaults: { api_key_location: "" } },
    sglang: { defaults: { api_key_location: "" } },
    tgi: { defaults: { api_key_location: "" } },
    together: { defaults: { api_key_location: "" } },
    vllm: { defaults: { api_key_location: "" } },
    xai: { defaults: { api_key_location: "" } },
  },
  optimizers: {},
  models: { table: {} },
  embedding_models: { table: {} },
  functions: mockFunctions,
  metrics: {
    sentiment_score: {
      type: "float",
      optimize: "max",
      level: "inference",
    },
    toxicity: {
      type: "float",
      optimize: "min",
      level: "inference",
    },
    approved: {
      type: "boolean",
      optimize: "max",
      level: "inference",
    },
    episode_success: {
      type: "boolean",
      optimize: "max",
      level: "episode",
    },
  },
  tools: {},
  evaluations: {},
  postgres: {
    connection_pool_size: 10,
  },
  rate_limiting: {
    rules: [],
    enabled: true,
  },
};

export const Default: Story = {
  render: () => {
    return (
      <ConfigProvider value={mockConfig}>
        <InferenceQueryBuilder />
      </ConfigProvider>
    );
  },
};

export const Filled: Story = {
  render: () => {
    return (
      <ConfigProvider value={mockConfig}>
        <InferenceQueryBuilder
          defaultValues={{
            function: "chat-function",
            inferenceFilter: FILLED_INFERENCE_FILTER,
          }}
        />
      </ConfigProvider>
    );
  },
};

export const NoFunctions: Story = {
  render: () => {
    const emptyConfig: Config = {
      ...mockConfig,
      functions: {},
    };

    return (
      <ConfigProvider value={emptyConfig}>
        <InferenceQueryBuilder />
      </ConfigProvider>
    );
  },
};
