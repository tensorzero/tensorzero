import type { Meta, StoryObj } from "@storybook/react-vite";
import { VariantAdvancedMetadata } from "./VariantAdvancedMetadata";
import type { VariantConfig } from "tensorzero-node";

const meta: Meta<typeof VariantAdvancedMetadata> = {
  title: "Playground/VariantAdvancedMetadata",
  component: VariantAdvancedMetadata,
  parameters: {
    layout: "padded",
  },
  decorators: [
    (Story) => (
      <div className="max-w-2xl rounded-lg bg-white p-6 shadow-sm">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof VariantAdvancedMetadata>;

// Chat Completion variant with advanced settings
const chatCompletionVariant: VariantConfig = {
  type: "chat_completion",
  weight: 0.7,
  model: "gpt-4o-mini-2024-07-18",
  system_template: null,
  user_template: null,
  assistant_template: null,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 1000,
  presence_penalty: 0.1,
  frequency_penalty: 0.2,
  seed: 12345,
  stop_sequences: ["\\n\\n", "END"],
  json_mode: "strict",
  retries: {
    num_retries: 3,
    max_delay_s: 10,
  },
  extra_body: {
    data: [{ pointer: "/response_format", value: { type: "json_object" } }],
  },
  extra_headers: {
    data: [{ name: "X-Custom-Header", value: "custom-value" } as const],
  },
};

export const ChatCompletion: Story = {
  args: {
    variant: chatCompletionVariant,
  },
};

// Best of N Sampling variant
const bestOfNVariant: VariantConfig = {
  type: "best_of_n_sampling",
  weight: 1.0,
  timeout_s: 30,
  candidates: ["variant_a", "variant_b", "variant_c"],
  evaluator: {
    weight: null,
    model: "gpt-4-turbo-2024-04-09",
    system_template: null,
    user_template: null,
    assistant_template: null,
    temperature: 0.0,
    top_p: 1.0,
    max_tokens: 100,
    presence_penalty: null,
    frequency_penalty: null,
    seed: 42,
    stop_sequences: null,
    json_mode: "on",
    retries: {
      num_retries: 2,
      max_delay_s: 5,
    },
    extra_body: null,
    extra_headers: null,
  },
};

export const BestOfNSampling: Story = {
  args: {
    variant: bestOfNVariant,
  },
};

// Mixture of N variant
const mixtureOfNVariant: VariantConfig = {
  type: "mixture_of_n",
  weight: 0.5,
  timeout_s: 45,
  candidates: ["expert_1", "expert_2", "expert_3", "expert_4"],
  fuser: {
    weight: null,
    model: "claude-3-5-sonnet-20241022",
    system_template: null,
    user_template: null,
    assistant_template: null,
    temperature: 0.3,
    top_p: 0.95,
    max_tokens: 2000,
    presence_penalty: null,
    frequency_penalty: null,
    seed: null,
    stop_sequences: ["</answer>"],
    json_mode: null,
    retries: {
      num_retries: 3,
      max_delay_s: 15,
    },
    extra_body: null,
    extra_headers: null,
  },
};

export const MixtureOfN: Story = {
  args: {
    variant: mixtureOfNVariant,
  },
};

// DICL variant
const diclVariant: VariantConfig = {
  type: "dicl",
  weight: 0.8,
  embedding_model: "text-embedding-3-small",
  k: 5,
  model: "gpt-4o-2024-08-06",
  system_instructions:
    "You are a helpful assistant that provides accurate information based on the provided context.",
  temperature: 0.5,
  top_p: 0.8,
  stop_sequences: null,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  max_tokens: 1500,
  seed: 99999,
  json_mode: "implicit_tool",
  extra_body: null,
  extra_headers: {
    data: [
      { name: "X-API-Version", value: "v2" } as const,
      { name: "X-Request-ID", value: "abc123" } as const,
    ],
  },
  retries: {
    num_retries: 5,
    max_delay_s: 30,
  },
};

export const DICL: Story = {
  args: {
    variant: diclVariant,
  },
};
