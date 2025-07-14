import type { Meta, StoryObj } from "@storybook/react-vite";
import FineTuningStatus from "./FineTuningStatus";
import type { SFTJobStatus } from "~/utils/supervised_fine_tuning/common";
import { withRouter } from "storybook-addon-remix-react-router";

const meta = {
  title: "SFT/FineTuningStatus",
  component: FineTuningStatus,
  decorators: [withRouter],
  render: (args) => (
    <div className="w-[80vw] p-4">
      <FineTuningStatus {...args} />
    </div>
  ),
} satisfies Meta<typeof FineTuningStatus>;

export default meta;
type Story = StoryObj<typeof meta>;

const baseFormData = {
  function: "chat_completion",
  metric: "accuracy",
  model: {
    displayName: "gpt-4o-mini-2024-07-18",
    name: "gpt-4o-mini-2024-07-18",
    provider: "openai" as const,
  },
  variant: "default_chat_variant",
  validationSplitPercent: 20,
  maxSamples: 1000,
  threshold: 0.8,
  jobId: "01234567-89ab-cdef-0123-456789abcdef",
};

const baseRawData = {
  status: "ok" as const,
  info: {
    id: "ftjob-abc123xyz789",
    object: "fine_tuning.job",
    created_at: 1640995200,
    finished_at: null,
    model: "gpt-4o-mini-2024-07-18",
    organization_id: "org-123456789",
    result_files: [],
    status: "running",
    validation_file: "file-abc123xyz789",
    training_file: "file-def456uvw012",
  },
};

export const Idle: Story = {
  args: {
    status: { status: "idle" },
    result: null,
  },
};

export const Running: Story = {
  args: {
    status: {
      status: "running",
      modelProvider: "openai",
      formData: baseFormData,
      jobUrl: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      rawData: baseRawData,
    } satisfies SFTJobStatus,
    result: null,
  },
};

export const RunningWithEstimatedCompletion: Story = {
  args: {
    status: {
      status: "running",
      modelProvider: "openai",
      formData: baseFormData,
      jobUrl: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      rawData: baseRawData,
      estimatedCompletionTime: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
    } satisfies SFTJobStatus,

    result: null,
  },
};

export const Completed: Story = {
  args: {
    status: {
      status: "completed",
      modelProvider: "openai",
      formData: baseFormData,
      jobUrl: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      rawData: {
        status: "ok",
        info: {
          ...baseRawData.info,
          status: "succeeded",
          finished_at: 1640998800,
          fine_tuned_model:
            "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
        },
      },
      result: "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
    } satisfies SFTJobStatus,
    result: null,
  },
};

export const Error: Story = {
  args: {
    status: {
      status: "error",
      modelProvider: "openai",
      formData: baseFormData,
      jobUrl: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      rawData: {
        status: "error",
        message: "Training data validation failed: Invalid format in line 42",
      },
      error: "Training data validation failed: Invalid format in line 42",
    } satisfies SFTJobStatus,
    result: null,
  },
};

export const LongJobId: Story = {
  name: "Very long job ID",
  args: {
    status: {
      status: "completed",
      modelProvider: "openai",
      formData: {
        ...baseFormData,
        jobId:
          "01234567-89ab-cdef-0123-456789abcdef-very-long-suffix-that-wraps",
        function:
          "very_long_function_name_that_might_wrap_on_smaller_screens_or_might_not_but_hopefully_does_when_this_id_does_finally_terminate",
        variant:
          "extremely_long_variant_name_that_definitely_wraps_around_in_the_ui",
      },
      jobUrl: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      rawData: {
        status: "ok",
        info: {
          ...baseRawData.info,
          status: "succeeded",
          finished_at: 1640998800,
          fine_tuned_model:
            "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
        },
      },
      result: "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
    } satisfies SFTJobStatus,
    result: null,
  },
};
