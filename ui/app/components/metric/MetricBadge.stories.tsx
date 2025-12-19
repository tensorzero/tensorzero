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

// Threshold examples - shows red when value fails threshold
export const NumberFailsMaxThreshold: Story = {
  args: {
    label: "accuracy",
    value: 0.5,
    optimize: "max",
    cutoff: 0.8,
  },
};

export const NumberPassesMaxThreshold: Story = {
  args: {
    label: "accuracy",
    value: 0.9,
    optimize: "max",
    cutoff: 0.8,
  },
};

export const NumberFailsMinThreshold: Story = {
  args: {
    label: "latency_ms",
    value: 500,
    optimize: "min",
    cutoff: 200,
  },
};

export const BooleanFailsMax: Story = {
  args: {
    label: "exact_match",
    value: false,
    optimize: "max",
  },
};

export const BooleanPassesMax: Story = {
  args: {
    label: "exact_match",
    value: true,
    optimize: "max",
  },
};
