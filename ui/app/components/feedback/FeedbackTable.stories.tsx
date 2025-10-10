import type { Meta, StoryObj } from "@storybook/react-vite";
import FeedbackTable from "./FeedbackTable";
import { ConfigProvider } from "~/context/config";
import type { Config } from "tensorzero-node";

// Helper function to generate a UUID-like string from a number that sorts correctly
// Higher numbers produce lexicographically larger UUIDs (for descending sort)
function makeOrderedUuid(num = 0): string {
  const hexNum = num.toString(16).padStart(8, "0");
  return `${hexNum}-0000-0000-0000-000000000000`;
}

const config: Config = {
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
    accuracy: {
      type: "float" as const,
      optimize: "max" as const,
      level: "inference" as const,
    },
    exact_match: {
      type: "boolean" as const,
      optimize: "max" as const,
      level: "episode" as const,
    },
    nsfw_detected: {
      type: "boolean" as const,
      optimize: "min" as const,
      level: "inference" as const,
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

const meta = {
  title: "FeedbackTable",
  component: FeedbackTable,
  render: (args) => (
    <StoryWrapper>
      <FeedbackTable
        feedback={args.feedback}
        latestCommentId={args.latestCommentId}
        latestDemonstrationId={args.latestDemonstrationId}
        latestFeedbackIdByMetric={args.latestFeedbackIdByMetric}
      />
    </StoryWrapper>
  ),
} satisfies Meta<typeof FeedbackTable>;

export default meta;
type Story = StoryObj<typeof meta>;

const TARGET_ID = makeOrderedUuid();

const StoryWrapper = ({ children }: { children: React.ReactNode }) => (
  <ConfigProvider value={config}>
    <div className="w-[80vw] p-4">{children}</div>
  </ConfigProvider>
);

export const Empty: Story = {
  args: {
    feedback: [],
    latestCommentId: makeOrderedUuid(0),
    latestDemonstrationId: makeOrderedUuid(0),
    latestFeedbackIdByMetric: {},
  },
};

export const WithData: Story = {
  args: {
    feedback: [
      {
        type: "float",
        id: makeOrderedUuid(10),
        target_id: TARGET_ID,
        metric_name: "accuracy",
        value: 0.95,
        tags: {},
        timestamp: "2024-03-20T10:00:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(9),
        target_id: TARGET_ID,
        metric_name: "exact_match",
        value: true,
        tags: {},
        timestamp: "2024-03-20T10:01:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(8),
        target_id: TARGET_ID,
        metric_name: "exact_match",
        value: false,
        tags: {},
        timestamp: "2024-03-20T10:01:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(7),
        target_id: TARGET_ID,
        metric_name: "nsfw_detected",
        value: true,
        tags: {},
        timestamp: "2024-03-20T10:02:00Z",
      },
      {
        type: "float",
        id: makeOrderedUuid(6),
        target_id: TARGET_ID,
        metric_name: "unknown_float_metric",
        value: 0.5,
        tags: {},
        timestamp: "2024-03-20T10:03:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(5),
        target_id: TARGET_ID,
        metric_name: "nsfw_detected",
        value: false,
        tags: {},
        timestamp: "2024-03-20T10:02:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(4),
        target_id: TARGET_ID,
        metric_name: "unknown_boolean_metric",
        value: true,
        tags: {},
        timestamp: "2024-03-20T10:04:00Z",
      },
      {
        type: "comment",
        id: makeOrderedUuid(3),
        target_id: TARGET_ID,
        target_type: "episode",
        value: "This is a comment.",
        tags: {},
        timestamp: "2024-03-20T10:05:00Z",
      },
      {
        type: "demonstration",
        id: makeOrderedUuid(2),
        inference_id: TARGET_ID,
        value: JSON.stringify([
          {
            type: "text",
            text: "This is a demonstration.",
          },
        ]),
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:06:00Z",
      },
    ],
    latestCommentId: makeOrderedUuid(3),
    latestDemonstrationId: makeOrderedUuid(2),
    latestFeedbackIdByMetric: {
      accuracy: makeOrderedUuid(10),
      exact_match: makeOrderedUuid(9),
      nsfw_detected: makeOrderedUuid(100000), // Both `nsfw_detected` feedback in table will show "Overwritten"
      unknown_float_metric: makeOrderedUuid(6),
      unknown_boolean_metric: makeOrderedUuid(4),
    },
  },
};

export const WithLongComment: Story = {
  args: {
    feedback: [
      {
        type: "comment",
        id: makeOrderedUuid(0),
        target_id: TARGET_ID,
        target_type: "episode",
        value:
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras dolor ex, posuere at libero sit amet, mollis varius neque. Aliquam et purus eu erat imperdiet faucibus a non sapien. Proin arcu sapien, auctor a malesuada vel, condimentum vitae mauris. Nam nec pellentesque eros, nec accumsan metus. Proin quis augue sagittis, aliquet dolor gravida, pulvinar ligula. Proin et interdum lorem. Etiam at enim sodales ligula molestie viverra. Quisque tincidunt eget dolor id tempus.\n\n" +
          "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Sed vitae lorem vel nisl vehicula tincidunt. Mauris vehicula massa in dolor tincidunt eleifend. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Donec ultrices tellus vitae diam sagittis venenatis. Nulla facilisi. Nullam lacinia tellus nec dui tempor, non tempus eros facilisis.\n\n" +
          "Fusce vehicula, tortor et congue tincidunt, dolor magna tempor nisl, et ultricies massa diam sit amet libero. Integer consectetur urna non ex sollicitudin, in tincidunt nisi hendrerit. Praesent in vehicula nisi. Suspendisse potenti. Sed fermentum magna vitae lectus venenatis, vel sagittis dolor dictum. Donec sed odio dui. Cras mattis consectetur purus sit amet fermentum.\n\n" +
          "Maecenas volutpat, quam id porttitor tincidunt, velit turpis vulputate justo, sed laoreet nulla risus nec velit. Suspendisse potenti. Nullam auctor pulvinar nisi, at tempor nisi hendrerit vitae. Sed consequat magna at velit fermentum, quis aliquam enim tempus. Morbi malesuada ligula a mauris tempor dignissim. Vivamus dictum purus sed purus fermentum pharetra. Duis ut libero nec ligula facilisis mattis.\n\n" +
          "Phasellus ullamcorper ipsum rutrum nunc. Nunc nonummy metus. Vestibulum volutpat pretium libero. Cras id dui. Aenean ut eros et nisl sagittis vestibulum. Nullam nulla eros, ultricies sit amet, nonummy id, imperdiet feugiat, pede. Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis. Etiam imperdiet imperdiet orci.",
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:00:00Z",
      },
    ],
    latestCommentId: makeOrderedUuid(0),
    latestDemonstrationId: makeOrderedUuid(0),
    latestFeedbackIdByMetric: {},
  },
};

export const WithLongDemonstration: Story = {
  args: {
    feedback: [
      {
        type: "demonstration",
        id: makeOrderedUuid(0),
        inference_id: TARGET_ID,
        value: JSON.stringify([
          {
            type: "text",
            text: "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\n",
          },
        ]),
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:00:00Z",
      },
    ],
    latestCommentId: undefined,
    latestDemonstrationId: makeOrderedUuid(0),
    latestFeedbackIdByMetric: {},
  },
};

export const WithHumanFeedback: Story = {
  args: {
    feedback: [
      // Long demonstration (latest)
      {
        type: "demonstration",
        id: makeOrderedUuid(6),
        inference_id: TARGET_ID,
        value: JSON.stringify([
          {
            type: "text",
            text: "Long demonstration: Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\n",
          },
        ]),
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:06:00Z",
      },
      // Boolean
      {
        type: "boolean",
        id: makeOrderedUuid(5),
        target_id: TARGET_ID,
        metric_name: "nsfw_detected",
        value: true,
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:04:00Z",
      },
      // Long float (revenue)
      {
        type: "float",
        id: makeOrderedUuid(4),
        target_id: TARGET_ID,
        metric_name: "revenue",
        value: 12345678901234567890, // eslint-disable-line no-loss-of-precision
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:03:00Z",
      },
      // Short float (accuracy)
      {
        type: "float",
        id: makeOrderedUuid(3),
        target_id: TARGET_ID,
        metric_name: "accuracy",
        value: 0.5,
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:03:00Z",
      },
      // Long comment
      {
        type: "comment",
        id: makeOrderedUuid(2),
        target_id: TARGET_ID,
        target_type: "episode",
        value:
          "This is a really long comment. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\n",
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:02:00Z",
      },
      // Short comment
      {
        type: "comment",
        id: makeOrderedUuid(1),
        target_id: TARGET_ID,
        target_type: "episode",
        value: "Nice job!",
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:01:00Z",
      },
      // Short demonstration (overwritten)
      {
        type: "demonstration",
        id: makeOrderedUuid(0),
        inference_id: TARGET_ID,
        value: JSON.stringify([
          {
            type: "text",
            text: "Short demonstration",
          },
        ]),
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:00:00Z",
      },
    ],
    latestCommentId: makeOrderedUuid(2),
    latestDemonstrationId: makeOrderedUuid(6),
    latestFeedbackIdByMetric: {
      accuracy: makeOrderedUuid(3),
      revenue: makeOrderedUuid(4),
      nsfw_detected: makeOrderedUuid(5),
    },
  },
};

export const WithVariousTags: Story = {
  args: {
    feedback: [
      {
        type: "float",
        id: makeOrderedUuid(10),
        target_id: TARGET_ID,
        metric_name: "accuracy",
        value: 0.95,
        tags: { user_id: "123", experiment: "A" },
        timestamp: "2024-03-20T10:00:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(9),
        target_id: TARGET_ID,
        metric_name: "exact_match",
        value: true,
        tags: {
          "tensorzero::human_feedback": "true",
          session_id: "abc-def-ghi",
        },
        timestamp: "2024-03-20T10:01:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(8),
        target_id: TARGET_ID,
        metric_name: "nsfw_detected",
        value: false,
        tags: {
          "tensorzero::evaluation_name": "safety_check",
          priority: "high",
          model: "gpt-4",
        },
        timestamp: "2024-03-20T10:02:00Z",
      },
      {
        type: "float",
        id: makeOrderedUuid(7),
        target_id: TARGET_ID,
        metric_name: "relevance",
        value: 0.87,
        tags: {
          very_long_tag_key_that_might_overflow:
            "very_long_tag_value_that_will_definitely_need_truncation_in_the_ui",
        },
        timestamp: "2024-03-20T10:03:00Z",
      },
      {
        type: "comment",
        id: makeOrderedUuid(6),
        target_id: TARGET_ID,
        target_type: "episode",
        value: "Great response!",
        tags: {
          multiple: "tags",
          showing: "various",
          lengths: "short",
          and_some_longer_ones: "like_this_one",
        },
        timestamp: "2024-03-20T10:04:00Z",
      },
      {
        type: "demonstration",
        id: makeOrderedUuid(5),
        inference_id: TARGET_ID,
        value: JSON.stringify([{ type: "text", text: "Perfect example" }]),
        tags: {
          "tensorzero::dataset_name": "training_set",
          "tensorzero::datapoint_id": "dp-123",
          version: "2.0",
        },
        timestamp: "2024-03-20T10:05:00Z",
      },
      {
        type: "boolean",
        id: makeOrderedUuid(4),
        target_id: TARGET_ID,
        metric_name: "hallucination",
        value: false,
        tags: {},
        timestamp: "2024-03-20T10:06:00Z",
      },
    ],
    latestCommentId: makeOrderedUuid(6),
    latestDemonstrationId: makeOrderedUuid(5),
    latestFeedbackIdByMetric: {
      accuracy: makeOrderedUuid(10),
      exact_match: makeOrderedUuid(9),
      nsfw_detected: makeOrderedUuid(8),
      relevance: makeOrderedUuid(7),
      hallucination: makeOrderedUuid(4),
    },
  },
};
