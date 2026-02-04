import type { Meta, StoryObj } from "@storybook/react-vite";
import { BarChartSkeleton } from "./BarChartSkeleton";

const meta: Meta<typeof BarChartSkeleton> = {
  title: "Chart Skeletons/BarChartSkeleton",
  component: BarChartSkeleton,
  parameters: {
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof BarChartSkeleton>;

export const Default: Story = {
  args: {},
};

export const WithCustomWidth: Story = {
  args: {
    className: "w-[600px]",
  },
};
