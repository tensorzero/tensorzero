import type { Meta, StoryObj } from "@storybook/react-vite";
import InferenceFilterBuilder from "./InferenceFilterBuilder";
import { ConfigProvider } from "~/context/config";
import type { Config } from "tensorzero-node";
import type { InferenceFilter } from "tensorzero-node";
import { FormProvider, useForm } from "react-hook-form";
import { useState } from "react";

const meta = {
  title: "QueryBuilder/InferenceFilterBuilder",
  component: InferenceFilterBuilder,
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
} satisfies Meta<typeof InferenceFilterBuilder>;

export default meta;
type Story = StoryObj<typeof meta>;

// Shared fixture for filled filter state
const FILLED_INFERENCE_FILTER: InferenceFilter = {
  type: "and",
  children: [
    {
      type: "float_metric",
      metric_name: "toxicity",
      comparison_operator: ">",
      value: 0.8,
    },
    {
      type: "or",
      children: [
        {
          type: "boolean_metric",
          metric_name: "episode_success",
          value: true,
        },
        {
          type: "tag",
          key: "user_id",
          value: "12345",
          comparison_operator: "=",
        },
      ],
    },
    {
      type: "tag",
      key: "device",
      value: "mobile",
      comparison_operator: "!=",
    },
  ],
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
  functions: {},
  metrics: {
    sentiment_score: {
      type: "float",
      optimize: "max",
      level: "inference",
    },
    confidence_with_a_very_very_very_very_very_very_very_very_long_metric_name:
      {
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
    factually_correct: {
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
  args: {
    inferenceFilter: undefined,
    setInferenceFilter: () => {},
  },
  render: function DefaultStory() {
    const [inferenceFilter, setInferenceFilter] = useState<
      InferenceFilter | undefined
    >(undefined);
    const form = useForm();

    return (
      <ConfigProvider value={mockConfig}>
        <FormProvider {...form}>
          <InferenceFilterBuilder
            inferenceFilter={inferenceFilter}
            setInferenceFilter={setInferenceFilter}
          />
          <div className="mt-4 rounded border border-blue-300 bg-blue-50 p-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="font-semibold text-blue-900">
                Debug: Inference Filters
              </h3>
            </div>
            <pre className="mt-2 overflow-auto rounded bg-white p-2 text-xs">
              {inferenceFilter
                ? JSON.stringify(inferenceFilter, null, 2)
                : "undefined"}
            </pre>
          </div>
        </FormProvider>
      </ConfigProvider>
    );
  },
};

export const Filled: Story = {
  args: {
    inferenceFilter: undefined,
    setInferenceFilter: () => {},
  },
  render: function FilledStory() {
    const [inferenceFilter, setInferenceFilter] = useState<
      InferenceFilter | undefined
    >(FILLED_INFERENCE_FILTER);
    const form = useForm();

    return (
      <ConfigProvider value={mockConfig}>
        <FormProvider {...form}>
          <InferenceFilterBuilder
            inferenceFilter={inferenceFilter}
            setInferenceFilter={setInferenceFilter}
          />
          <div className="mt-4 rounded border border-blue-300 bg-blue-50 p-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="font-semibold text-blue-900">
                Debug: Inference Filters
              </h3>
            </div>
            <pre className="mt-2 overflow-auto rounded bg-white p-2 text-xs">
              {inferenceFilter
                ? JSON.stringify(inferenceFilter, null, 2)
                : "undefined"}
            </pre>
          </div>
        </FormProvider>
      </ConfigProvider>
    );
  },
};

export const EmptyMetrics: Story = {
  args: {
    inferenceFilter: undefined,
    setInferenceFilter: () => {},
  },
  render: function EmptyMetricsStory() {
    const [inferenceFilter, setInferenceFilter] = useState<
      InferenceFilter | undefined
    >(undefined);
    const form = useForm();

    const emptyMetricsConfig: Config = {
      ...mockConfig,
      metrics: {},
    };

    return (
      <ConfigProvider value={emptyMetricsConfig}>
        <FormProvider {...form}>
          <InferenceFilterBuilder
            inferenceFilter={inferenceFilter}
            setInferenceFilter={setInferenceFilter}
          />
          <div className="mt-4 rounded border border-blue-300 bg-blue-50 p-4">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="font-semibold text-blue-900">
                Debug: Inference Filters
              </h3>
            </div>
            <pre className="mt-2 overflow-auto rounded bg-white p-2 text-xs">
              {inferenceFilter
                ? JSON.stringify(inferenceFilter, null, 2)
                : "undefined"}
            </pre>
          </div>
        </FormProvider>
      </ConfigProvider>
    );
  },
};
