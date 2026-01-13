import type { Meta, StoryObj } from "@storybook/react-vite";
import { LineChartSkeleton } from "./chart";

const meta = {
  title: "Data Visualization/LineChartSkeleton",
  component: LineChartSkeleton,
  parameters: {
    layout: "padded",
  },
  decorators: [
    (Story) => (
      <div className="w-full max-w-4xl">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof LineChartSkeleton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  name: "Loading State",
};
