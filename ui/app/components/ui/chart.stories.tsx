import type { Meta, StoryObj } from "@storybook/react-vite";
import { ChartAsyncErrorState } from "./chart";

const meta = {
  title: "DS/ChartAsyncErrorState",
  component: ChartAsyncErrorState,
  parameters: {
    layout: "centered",
  },
  decorators: [
    (Story) => (
      <div className="w-[600px]">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ChartAsyncErrorState>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    defaultMessage: "Failed to load chart data",
  },
};

export const CustomMessage: Story = {
  args: {
    defaultMessage: "Connection timeout while fetching latency metrics",
  },
};

export const LongMessage: Story = {
  args: {
    defaultMessage:
      "Unable to connect to ClickHouse database. Please check that the database is running and accessible.",
  },
};
