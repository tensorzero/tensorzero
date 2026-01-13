import type { Meta, StoryObj } from "@storybook/react-vite";
import { ModelUsage } from "./ModelUsage";
import type { ModelUsageTimePoint } from "~/types/tensorzero";

// Mock data for model usage over time
const mockUsageData: ModelUsageTimePoint[] = [
  // Week 1
  {
    period_start: "2024-01-01T00:00:00Z",
    model_name: "gpt-4o",
    input_tokens: BigInt(125000),
    output_tokens: BigInt(45000),
    count: BigInt(523),
  },
  {
    period_start: "2024-01-01T00:00:00Z",
    model_name: "claude-3-5-sonnet",
    input_tokens: BigInt(98000),
    output_tokens: BigInt(38000),
    count: BigInt(412),
  },
  {
    period_start: "2024-01-01T00:00:00Z",
    model_name: "gpt-4o-mini",
    input_tokens: BigInt(210000),
    output_tokens: BigInt(82000),
    count: BigInt(1245),
  },
  // Week 2
  {
    period_start: "2024-01-08T00:00:00Z",
    model_name: "gpt-4o",
    input_tokens: BigInt(145000),
    output_tokens: BigInt(52000),
    count: BigInt(612),
  },
  {
    period_start: "2024-01-08T00:00:00Z",
    model_name: "claude-3-5-sonnet",
    input_tokens: BigInt(112000),
    output_tokens: BigInt(43000),
    count: BigInt(478),
  },
  {
    period_start: "2024-01-08T00:00:00Z",
    model_name: "gpt-4o-mini",
    input_tokens: BigInt(235000),
    output_tokens: BigInt(91000),
    count: BigInt(1389),
  },
  // Week 3
  {
    period_start: "2024-01-15T00:00:00Z",
    model_name: "gpt-4o",
    input_tokens: BigInt(168000),
    output_tokens: BigInt(61000),
    count: BigInt(723),
  },
  {
    period_start: "2024-01-15T00:00:00Z",
    model_name: "claude-3-5-sonnet",
    input_tokens: BigInt(134000),
    output_tokens: BigInt(51000),
    count: BigInt(567),
  },
  {
    period_start: "2024-01-15T00:00:00Z",
    model_name: "gpt-4o-mini",
    input_tokens: BigInt(278000),
    output_tokens: BigInt(108000),
    count: BigInt(1634),
  },
  // Week 4
  {
    period_start: "2024-01-22T00:00:00Z",
    model_name: "gpt-4o",
    input_tokens: BigInt(189000),
    output_tokens: BigInt(72000),
    count: BigInt(845),
  },
  {
    period_start: "2024-01-22T00:00:00Z",
    model_name: "claude-3-5-sonnet",
    input_tokens: BigInt(156000),
    output_tokens: BigInt(59000),
    count: BigInt(656),
  },
  {
    period_start: "2024-01-22T00:00:00Z",
    model_name: "gpt-4o-mini",
    input_tokens: BigInt(312000),
    output_tokens: BigInt(124000),
    count: BigInt(1856),
  },
];

const meta = {
  title: "Model/ModelUsage",
  component: ModelUsage,
  parameters: {
    layout: "padded",
  },
  decorators: [
    (Story) => (
      <div className="w-full max-w-4xl p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ModelUsage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Loading: Story = {
  args: {
    modelUsageDataPromise: new Promise(() => {}), // Never resolves - shows skeleton
  },
};

export const WithData: Story = {
  args: {
    modelUsageDataPromise: Promise.resolve(mockUsageData),
  },
};

export const WithError: Story = {
  args: {
    modelUsageDataPromise: Promise.reject(
      new Error("Failed to fetch usage data from ClickHouse"),
    ),
  },
};

export const Empty: Story = {
  args: {
    modelUsageDataPromise: Promise.resolve([]),
  },
};

export const SingleModel: Story = {
  args: {
    modelUsageDataPromise: Promise.resolve(
      mockUsageData.filter((d) => d.model_name === "gpt-4o"),
    ),
  },
};
