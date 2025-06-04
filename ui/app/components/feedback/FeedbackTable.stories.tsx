import type { Meta, StoryObj } from "@storybook/react-vite";
import FeedbackTable from "./FeedbackTable";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import { ConfigProvider } from "~/context/config";
import type { Config } from "~/utils/config";

const config: Config = {
  gateway: {
    observability: {
      enabled: true,
      async_writes: false,
    },
    export: {
      otlp: {
        traces: {
          enabled: false,
        },
      },
    },
    debug: false,
    enable_template_filesystem_access: false,
  },
  models: {},
  embedding_models: {},
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
    demonstration: {
      type: "demonstration" as const,
      level: "inference" as const,
    },
    comment: {
      type: "comment" as const,
    },
  },
  tools: {},
  evaluations: {},
};

const meta = {
  title: "FeedbackTable",
  component: FeedbackTable,
  render: (args) => (
    <StoryWrapper>
      <FeedbackTable feedback={args.feedback} />
    </StoryWrapper>
  ),
} satisfies Meta<typeof FeedbackTable>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockFeedback: FeedbackRow[] = [
  {
    type: "float",
    id: "00000000-0000-0000-0000-000000000000",
    target_id: "00000001-0000-0000-0000-000000000000",
    metric_name: "accuracy",
    value: 0.95,
    tags: {},
    timestamp: "2024-03-20T10:00:00Z",
  },
  {
    type: "boolean",
    id: "00000002-0000-0000-0000-000000000000",
    target_id: "00000003-0000-0000-0000-000000000000",
    metric_name: "exact_match",
    value: true,
    tags: {},
    timestamp: "2024-03-20T10:01:00Z",
  },
  {
    type: "boolean",
    id: "00000004-0000-0000-0000-000000000000",
    target_id: "00000005-0000-0000-0000-000000000000",
    metric_name: "exact_match",
    value: false,
    tags: {},
    timestamp: "2024-03-20T10:01:00Z",
  },
  {
    type: "boolean",
    id: "00000006-0000-0000-0000-000000000000",
    target_id: "00000007-0000-0000-0000-000000000000",
    metric_name: "nsfw_detected",
    value: true,
    tags: {},
    timestamp: "2024-03-20T10:02:00Z",
  },
  {
    type: "boolean",
    id: "00000008-0000-0000-0000-000000000000",
    target_id: "00000009-0000-0000-0000-000000000000",
    metric_name: "nsfw_detected",
    value: false,
    tags: {},
    timestamp: "2024-03-20T10:02:00Z",
  },
  {
    type: "float",
    id: "0000000a-0000-0000-0000-000000000000",
    target_id: "0000000b-0000-0000-0000-000000000000",
    metric_name: "unknown_float_metric",
    value: 0.5,
    tags: {},
    timestamp: "2024-03-20T10:03:00Z",
  },
  {
    type: "boolean",
    id: "0000000c-0000-0000-0000-000000000000",
    target_id: "0000000d-0000-0000-0000-000000000000",
    metric_name: "unknown_boolean_metric",
    value: true,
    tags: {},
    timestamp: "2024-03-20T10:04:00Z",
  },
  {
    type: "comment",
    id: "0000000e-0000-0000-0000-000000000000",
    target_id: "0000000f-0000-0000-0000-000000000000",
    target_type: "episode",
    value: "This is a comment.",
    tags: {},
    timestamp: "2024-03-20T10:05:00Z",
  },
  {
    type: "demonstration",
    id: "00000010-0000-0000-0000-000000000000",
    inference_id: "00000011-0000-0000-0000-000000000000",
    value: JSON.stringify([
      {
        type: "text",
        text: "This is a demonstration.",
      },
    ]),
    tags: { "tensorzero::human_feedback": "true" },
    timestamp: "2024-03-20T10:06:00Z",
  },
];

const StoryWrapper = ({ children }: { children: React.ReactNode }) => (
  <ConfigProvider value={config}>
    <div className="w-[80vw] p-4">{children}</div>
  </ConfigProvider>
);

export const Empty: Story = {
  args: {
    feedback: [],
  },
};

export const WithData: Story = {
  args: {
    feedback: mockFeedback,
  },
};

export const WithLongComment: Story = {
  args: {
    feedback: [
      {
        type: "comment",
        id: "00000000-0000-0000-0000-000000000000",
        target_id: "00000000-0000-0000-0000-000000000000",
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
  },
  render: (args) => (
    <StoryWrapper>
      <FeedbackTable feedback={args.feedback} />
    </StoryWrapper>
  ),
};

export const WithLongDemonstration: Story = {
  args: {
    feedback: [
      {
        type: "demonstration",
        id: "00000000-0000-0000-0000-000000000000",
        inference_id: "00000000-0000-0000-0000-000000000000",
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
  },
};

export const WithHumanFeedback: Story = {
  args: {
    feedback: [
      // Short demonstration
      {
        type: "demonstration",
        id: "00000000-0000-0000-0000-000000000000",
        inference_id: "00000000-0000-0000-0000-000000000000",
        value: JSON.stringify([
          {
            type: "text",
            text: "Short demonstration",
          },
        ]),
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:00:00Z",
      },
      // Long demonstration
      {
        type: "demonstration",
        id: "00000000-0000-0000-0000-000000000000",
        inference_id: "00000000-0000-0000-0000-000000000000",
        value: JSON.stringify([
          {
            type: "text",
            text: "Long demonstration: Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\n",
          },
        ]),
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:00:00Z",
      },
      // Short comment
      {
        type: "comment",
        id: "00000000-0000-0000-0000-000000000001",
        target_id: "00000000-0000-0000-0000-000000000001",
        target_type: "episode",
        value: "Nice job!",
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:01:00Z",
      },
      // Long comment
      {
        type: "comment",
        id: "00000000-0000-0000-0000-000000000002",
        target_id: "00000000-0000-0000-0000-000000000002",
        target_type: "episode",
        value:
          "This is a really long comment. Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n\n",
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:02:00Z",
      },
      // Short float
      {
        type: "float",
        id: "00000000-0000-0000-0000-000000000003",
        target_id: "00000000-0000-0000-0000-000000000003",
        metric_name: "accuracy",
        value: 0.5,
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:03:00Z",
      },
      // Long float
      {
        type: "float",
        id: "00000000-0000-0000-0000-000000000003",
        target_id: "00000000-0000-0000-0000-000000000003",
        metric_name: "revenue",
        value: 12345678901234567890, // eslint-disable-line no-loss-of-precision
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:03:00Z",
      },
      // Boolean
      {
        type: "boolean",
        id: "00000000-0000-0000-0000-000000000004",
        target_id: "00000000-0000-0000-0000-000000000004",
        metric_name: "nsfw_detected",
        value: true,
        tags: { "tensorzero::human_feedback": "true" },
        timestamp: "2024-03-20T10:04:00Z",
      },
    ],
  },
};
