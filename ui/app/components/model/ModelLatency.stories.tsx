import type { Meta, StoryObj } from "@storybook/react-vite";
import { ModelLatency } from "./ModelLatency";
import type { ModelLatencyDatapoint } from "~/types/tensorzero";

// Sample quantiles (subset of the full list for testing)
const SAMPLE_QUANTILES = [
  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
];

// Mock data for two models
const mockLatencyData: ModelLatencyDatapoint[] = [
  {
    model_name: "gpt-4o",
    response_time_ms_quantiles: [
      120, 150, 180, 210, 250, 300, 380, 500, 750, 1200, 2500,
    ],
    ttft_ms_quantiles: [80, 95, 110, 130, 150, 180, 220, 280, 400, 650, 1200],
    count: BigInt(15234),
  },
  {
    model_name: "claude-3-5-sonnet",
    response_time_ms_quantiles: [
      100, 130, 160, 190, 230, 280, 350, 450, 680, 1100, 2200,
    ],
    ttft_ms_quantiles: [60, 75, 90, 110, 130, 160, 200, 260, 380, 600, 1100],
    count: BigInt(12456),
  },
  {
    model_name: "gpt-4o-mini",
    response_time_ms_quantiles: [
      80, 100, 120, 140, 170, 200, 250, 320, 480, 800, 1600,
    ],
    ttft_ms_quantiles: [40, 50, 60, 75, 90, 110, 140, 180, 280, 450, 850],
    count: BigInt(28901),
  },
];

const meta = {
  title: "Model/ModelLatency",
  component: ModelLatency,
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
} satisfies Meta<typeof ModelLatency>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Loading: Story = {
  args: {
    modelLatencyDataPromise: new Promise(() => {}), // Never resolves - shows skeleton
    quantiles: SAMPLE_QUANTILES,
  },
};

export const WithData: Story = {
  args: {
    modelLatencyDataPromise: Promise.resolve(mockLatencyData),
    quantiles: SAMPLE_QUANTILES,
  },
};

export const WithError: Story = {
  args: {
    modelLatencyDataPromise: Promise.reject(
      new Error("Connection timeout while fetching latency metrics"),
    ),
    quantiles: SAMPLE_QUANTILES,
  },
};

export const SingleModel: Story = {
  args: {
    modelLatencyDataPromise: Promise.resolve([mockLatencyData[0]]),
    quantiles: SAMPLE_QUANTILES,
  },
};

export const Empty: Story = {
  args: {
    modelLatencyDataPromise: Promise.resolve([]),
    quantiles: SAMPLE_QUANTILES,
  },
};
