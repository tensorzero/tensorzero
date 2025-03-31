import type { Meta, StoryObj } from "@storybook/react";
import FeedbackTable from "./FeedbackTable";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import { ConfigProvider } from "~/context/config";

const meta = {
  title: "FeedbackTable",
  component: FeedbackTable,
  decorators: [
    (Story) => (
      <ConfigProvider
        value={{
          gateway: {
            disable_observability: false,
          },
          models: {},
          embedding_models: {},
          functions: {},
          metrics: {
            accuracy: {
              type: "float",
              optimize: "max",
              level: "inference",
            },
            latency: {
              type: "float",
              optimize: "min",
              level: "inference",
            },
          },
          tools: {},
          evals: {},
        }}
      >
        <div className="w-[800px]">
          <Story />
        </div>
      </ConfigProvider>
    ),
  ],
} satisfies Meta<typeof FeedbackTable>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockFeedback: FeedbackRow[] = [
  {
    type: "float",
    id: "550e8400-e29b-41d4-a716-446655440000",
    target_id: "550e8400-e29b-41d4-a716-446655440001",
    metric_name: "accuracy",
    value: 0.95,
    tags: {},
    timestamp: "2024-03-20T10:00:00Z",
  },
  {
    type: "float",
    id: "550e8400-e29b-41d4-a716-446655440002",
    target_id: "550e8400-e29b-41d4-a716-446655440003",
    metric_name: "latency",
    value: 150,
    tags: {},
    timestamp: "2024-03-20T10:01:00Z",
  },
];

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
