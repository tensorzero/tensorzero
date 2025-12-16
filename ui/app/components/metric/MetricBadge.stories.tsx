import type { Meta, StoryObj } from "@storybook/react-vite";
import { MetricBadge } from "./MetricBadge";

const meta: Meta<typeof MetricBadge> = {
  title: "MetricBadge",
  component: MetricBadge,
  parameters: {
    layout: "centered",
  },
};

export default meta;
type Story = StoryObj<typeof MetricBadge>;

export const BooleanTrue: Story = {
  args: {
    value: true,
  },
};

export const BooleanFalse: Story = {
  args: {
    value: false,
  },
};

export const Number: Story = {
  args: {
    value: 0.875,
  },
};

export const String: Story = {
  args: {
    value: "excellent",
  },
};

export const NullValue: Story = {
  args: {
    value: null,
  },
};

export const WithLabel: Story = {
  args: {
    label: "accuracy",
    value: 0.95,
  },
};

export const Error: Story = {
  args: {
    label: "relevance",
    error: true,
  },
};

export const LongLabel: Story = {
  args: {
    label: "semantic_similarity_score",
    value: 0.823,
  },
};
