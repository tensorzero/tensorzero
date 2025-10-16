import type { Meta, StoryObj } from "@storybook/react-vite";
import FineTuningStatus from "./FineTuningStatus";
import type {
  OptimizationJobInfo,
  OptimizationJobHandle,
} from "tensorzero-node";

const meta = {
  title: "SFT/FineTuningStatus",
  component: FineTuningStatus,
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

export const Running: Story = {
  args: {
    status: {
      status: "pending",
      message: "Fine-tuning job is currently running",
      estimated_finish: new Date(Date.now() + 3600000), // 1 hour from now
      trained_tokens: 150000n,
      error: null,
    } as OptimizationJobInfo,
    formData: baseFormData,
    result: null,
    jobHandle: {
      type: "openai_sft",
      job_id: "ftjob-abc123xyz789",
      job_url: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      job_api_url:
        "https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123xyz789",
      credential_location: null,
    } as OptimizationJobHandle,
  },
};

export const RunningWithEstimatedCompletion: Story = {
  args: {
    status: {
      status: "pending",
      message:
        "Fine-tuning job is currently running with estimated completion time",
      estimated_finish: new Date(Date.now() + 3600000), // 1 hour from now
      trained_tokens: 500000n,
      error: null,
    } as OptimizationJobInfo,
    formData: baseFormData,
    result: null,
    jobHandle: {
      type: "openai_sft",
      job_id: "ftjob-abc123xyz789",
      job_url: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      job_api_url:
        "https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123xyz789",
      credential_location: null,
    } as OptimizationJobHandle,
  },
};

export const Completed: Story = {
  args: {
    status: {
      status: "completed",
      output: {
        type: "model",
        content: {
          routing: ["ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123"],
          providers: {
            "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123": {
              type: "openai",
              model_name:
                "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
              api_base: null,
              timeouts: {
                non_streaming: { total_ms: null },
                streaming: { ttft_ms: null },
              },
              discard_unknown_chunks: false,
              api_key_location: null,
              api_type: "chat_completions",
              include_encrypted_reasoning: false,
              provider_tools: [],
            },
          },
          timeouts: {
            non_streaming: {
              total_ms: null,
            },
            streaming: {
              ttft_ms: null,
            },
          },
        },
      },
    },
    formData: baseFormData,
    result: "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
    jobHandle: {
      type: "openai_sft",
      job_id: "ftjob-abc123xyz789",
      job_url: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      job_api_url:
        "https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123xyz789",
      credential_location: null,
    } as OptimizationJobHandle,
  },
};

export const Error: Story = {
  args: {
    status: {
      status: "failed",
      message: "Training data validation failed: Invalid format in line 42",
      error: "Training data validation failed: Invalid format in line 42",
    } as OptimizationJobInfo,
    formData: baseFormData,
    result: null,
    jobHandle: {
      type: "openai_sft",
      job_id: "ftjob-abc123xyz789",
      job_url: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      job_api_url:
        "https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123xyz789",
      credential_location: null,
    } as OptimizationJobHandle,
  },
};

export const LongJobId: Story = {
  name: "Very long job ID",
  args: {
    status: {
      status: "completed",
      output: {
        type: "model",
        content: {
          routing: ["ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123"],
          providers: {
            "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123": {
              type: "openai",
              model_name:
                "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
              api_base: null,
              timeouts: {
                non_streaming: { total_ms: null },
                streaming: { ttft_ms: null },
              },
              discard_unknown_chunks: false,
              api_key_location: null,
              api_type: "chat_completions",
              include_encrypted_reasoning: false,
              provider_tools: [],
            },
          },
          timeouts: {
            non_streaming: {
              total_ms: 300000n,
            },
            streaming: {
              ttft_ms: 300000n,
            },
          },
        },
      },
    },
    formData: {
      ...baseFormData,
      jobId: "01234567-89ab-cdef-0123-456789abcdef-very-long-suffix-that-wraps",
      function:
        "very_long_function_name_that_might_wrap_on_smaller_screens_or_might_not_but_hopefully_does_when_this_id_does_finally_terminate",
      variant:
        "extremely_long_variant_name_that_definitely_wraps_around_in_the_ui",
    },
    result: "ft:gpt-4o-mini-2024-07-18:my-org:custom-suffix:abc123",
    jobHandle: {
      type: "openai_sft",
      job_id: "ftjob-abc123xyz789",
      job_url: "https://platform.openai.com/finetune/ftjob-abc123xyz789",
      job_api_url:
        "https://api.openai.com/v1/fine_tuning/jobs/ftjob-abc123xyz789",
      credential_location: null,
    } as OptimizationJobHandle,
  },
};
