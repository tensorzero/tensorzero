import type { Meta, StoryObj } from "@storybook/react-vite";
import { LineChartSkeleton } from "./LineChartSkeleton";

const meta: Meta<typeof LineChartSkeleton> = {
  title: "Chart Skeletons/LineChartSkeleton",
  component: LineChartSkeleton,
  parameters: {
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof LineChartSkeleton>;

export const Default: Story = {
  args: {},
};

export const WithCustomWidth: Story = {
  args: {
    className: "w-[600px]",
  },
};
