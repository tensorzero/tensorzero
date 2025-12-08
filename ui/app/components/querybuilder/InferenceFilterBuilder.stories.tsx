import type { Meta, StoryObj } from "@storybook/react-vite";
import InferenceFilterBuilder from "./InferenceFilterBuilder";
import { ConfigProvider } from "~/context/config";
import type { InferenceFilter, UiConfig } from "~/types/tensorzero";
import { FormProvider, useForm } from "react-hook-form";
import { useState } from "react";
import { StoryDebugWrapper } from "~/components/.storybook/StoryDebugWrapper";

const meta = {
  title: "QueryBuilder/InferenceFilterBuilder",
  component: InferenceFilterBuilder,
  decorators: [
    (Story) => {
      return (
        <div className="w-2xl">
          <Story />
        </div>
      );
    },
  ],
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof InferenceFilterBuilder>;

export default meta;
type Story = StoryObj<typeof meta>;

// Shared fixture for filled filter state
const FILLED_INFERENCE_FILTER: InferenceFilter = {
  type: "and",
  children: [
    {
      type: "float_metric",
      metric_name: "toxicity",
      comparison_operator: ">",
      value: 0.8,
    },
    {
      type: "or",
      children: [
        {
          type: "boolean_metric",
          metric_name: "episode_success",
          value: true,
        },
        {
          type: "tag",
          key: "user_id",
          value: "12345",
          comparison_operator: "=",
        },
      ],
    },
    {
      type: "tag",
      key: "device",
      value: "mobile",
      comparison_operator: "!=",
    },
  ],
};

const mockConfig: UiConfig = {
  functions: {},
  metrics: {
    sentiment_score: {
      type: "float",
      optimize: "max",
      level: "inference",
    },
    confidence_with_a_very_very_very_very_very_very_very_very_long_metric_name:
      {
        type: "float",
        optimize: "max",
        level: "inference",
      },
    toxicity: {
      type: "float",
      optimize: "min",
      level: "inference",
    },
    approved: {
      type: "boolean",
      optimize: "max",
      level: "inference",
    },
    factually_correct: {
      type: "boolean",
      optimize: "max",
      level: "inference",
    },
    episode_success: {
      type: "boolean",
      optimize: "max",
      level: "episode",
    },
  },
  tools: {},
  evaluations: {},
  model_names: [],
  config_hash: "test-config-hash",
};

export const Default: Story = {
  args: {
    inferenceFilter: undefined,
    setInferenceFilter: () => {},
  },
  render: function DefaultStory() {
    const [inferenceFilter, setInferenceFilter] = useState<
      InferenceFilter | undefined
    >(undefined);
    const form = useForm();

    return (
      <ConfigProvider value={mockConfig}>
        <FormProvider {...form}>
          <StoryDebugWrapper
            debugLabel="inferenceFilter"
            debugData={inferenceFilter}
          >
            <InferenceFilterBuilder
              inferenceFilter={inferenceFilter}
              setInferenceFilter={setInferenceFilter}
            />
          </StoryDebugWrapper>
        </FormProvider>
      </ConfigProvider>
    );
  },
};

export const Filled: Story = {
  args: {
    inferenceFilter: undefined,
    setInferenceFilter: () => {},
  },
  render: function FilledStory() {
    const [inferenceFilter, setInferenceFilter] = useState<
      InferenceFilter | undefined
    >(FILLED_INFERENCE_FILTER);
    const form = useForm();

    return (
      <ConfigProvider value={mockConfig}>
        <FormProvider {...form}>
          <StoryDebugWrapper
            debugLabel="inferenceFilter"
            debugData={inferenceFilter}
          >
            <InferenceFilterBuilder
              inferenceFilter={inferenceFilter}
              setInferenceFilter={setInferenceFilter}
            />
          </StoryDebugWrapper>
        </FormProvider>
      </ConfigProvider>
    );
  },
};

export const EmptyMetrics: Story = {
  args: {
    inferenceFilter: undefined,
    setInferenceFilter: () => {},
  },
  render: function EmptyMetricsStory() {
    const [inferenceFilter, setInferenceFilter] = useState<
      InferenceFilter | undefined
    >(undefined);
    const form = useForm();

    const emptyMetricsConfig: UiConfig = {
      ...mockConfig,
      metrics: {},
    };

    return (
      <ConfigProvider value={emptyMetricsConfig}>
        <FormProvider {...form}>
          <StoryDebugWrapper
            debugLabel="inferenceFilter"
            debugData={inferenceFilter}
          >
            <InferenceFilterBuilder
              inferenceFilter={inferenceFilter}
              setInferenceFilter={setInferenceFilter}
            />
          </StoryDebugWrapper>
        </FormProvider>
      </ConfigProvider>
    );
  },
};
