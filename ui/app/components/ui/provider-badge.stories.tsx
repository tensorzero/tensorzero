import type { Meta, StoryObj } from "@storybook/react";
import { ProviderBadge } from "./provider-badge";
import type { ProviderConfig } from "~/utils/config/models";

const meta: Meta<typeof ProviderBadge> = {
  title: "UI/ProviderBadge",
  component: ProviderBadge,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    showModelName: {
      control: "boolean",
    },
    compact: {
      control: "boolean",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Provider configs for stories
const openaiProvider: ProviderConfig = {
  type: "openai",
  model_name: "gpt-4",
};

const anthropicProvider: ProviderConfig = {
  type: "anthropic",
  model_name: "claude-3-sonnet-20240229",
};

const awsBedrockProvider: ProviderConfig = {
  type: "aws_bedrock",
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0",
  region: "us-east-1",
};

const azureProvider: ProviderConfig = {
  type: "azure",
  deployment_id: "gpt-4-deployment",
  endpoint: "https://example.openai.azure.com/",
};

const mistralProvider: ProviderConfig = {
  type: "mistral",
  model_name: "mistral-large-latest",
};

const googleProvider: ProviderConfig = {
  type: "google_ai_studio_gemini",
  model_name: "gemini-pro",
};

export const Default: Story = {
  args: {
    provider: openaiProvider,
  },
};

export const WithModelName: Story = {
  args: {
    provider: openaiProvider,
    showModelName: true,
  },
};

export const Compact: Story = {
  args: {
    provider: openaiProvider,
    compact: true,
  },
};

export const CompactWithModel: Story = {
  args: {
    provider: openaiProvider,
    compact: true,
    showModelName: true,
  },
};

export const AllProviders: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2 max-w-4xl">
      <ProviderBadge provider={openaiProvider} />
      <ProviderBadge provider={anthropicProvider} />
      <ProviderBadge provider={awsBedrockProvider} />
      <ProviderBadge provider={azureProvider} />
      <ProviderBadge provider={mistralProvider} />
      <ProviderBadge provider={googleProvider} />
      <ProviderBadge provider={{ type: "groq", model_name: "llama-3-70b" }} />
      <ProviderBadge provider={{ type: "together", model_name: "meta-llama/Llama-2-70b-chat-hf" }} />
      <ProviderBadge provider={{ type: "fireworks", model_name: "accounts/fireworks/models/llama-v2-70b-chat" }} />
      <ProviderBadge provider={{ type: "deepseek", model_name: "deepseek-chat" }} />
      <ProviderBadge provider={{ type: "xai", model_name: "grok-beta" }} />
      <ProviderBadge provider={{ type: "hyperbolic", model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct" }} />
      <ProviderBadge provider={{ type: "vllm", model_name: "mistral-7b", api_base: "http://localhost:8000" }} />
      <ProviderBadge provider={{ type: "dummy", model_name: "test-model" }} />
    </div>
  ),
};

export const AllProvidersWithModelNames: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2 max-w-4xl">
      <ProviderBadge provider={openaiProvider} showModelName />
      <ProviderBadge provider={anthropicProvider} showModelName />
      <ProviderBadge provider={awsBedrockProvider} showModelName />
      <ProviderBadge provider={azureProvider} showModelName />
      <ProviderBadge provider={mistralProvider} showModelName />
      <ProviderBadge provider={googleProvider} showModelName />
      <ProviderBadge provider={{ type: "groq", model_name: "llama-3-70b" }} showModelName />
      <ProviderBadge provider={{ type: "together", model_name: "meta-llama/Llama-2-70b-chat-hf" }} showModelName />
      <ProviderBadge provider={{ type: "fireworks", model_name: "accounts/fireworks/models/llama-v2-70b-chat" }} showModelName />
      <ProviderBadge provider={{ type: "deepseek", model_name: "deepseek-chat" }} showModelName />
      <ProviderBadge provider={{ type: "xai", model_name: "grok-beta" }} showModelName />
      <ProviderBadge provider={{ type: "hyperbolic", model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct" }} showModelName />
      <ProviderBadge provider={{ type: "vllm", model_name: "mistral-7b", api_base: "http://localhost:8000" }} showModelName />
      <ProviderBadge provider={{ type: "dummy", model_name: "test-model" }} showModelName />
    </div>
  ),
};

export const CompactMode: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2 max-w-4xl">
      <ProviderBadge provider={openaiProvider} compact />
      <ProviderBadge provider={anthropicProvider} compact />
      <ProviderBadge provider={awsBedrockProvider} compact />
      <ProviderBadge provider={azureProvider} compact />
      <ProviderBadge provider={mistralProvider} compact />
      <ProviderBadge provider={googleProvider} compact />
      <ProviderBadge provider={{ type: "groq", model_name: "llama-3-70b" }} compact />
      <ProviderBadge provider={{ type: "together", model_name: "meta-llama/Llama-2-70b-chat-hf" }} compact />
      <ProviderBadge provider={{ type: "fireworks", model_name: "accounts/fireworks/models/llama-v2-70b-chat" }} compact />
      <ProviderBadge provider={{ type: "deepseek", model_name: "deepseek-chat" }} compact />
      <ProviderBadge provider={{ type: "xai", model_name: "grok-beta" }} compact />
      <ProviderBadge provider={{ type: "hyperbolic", model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct" }} compact />
      <ProviderBadge provider={{ type: "vllm", model_name: "mistral-7b", api_base: "http://localhost:8000" }} compact />
      <ProviderBadge provider={{ type: "dummy", model_name: "test-model" }} compact />
    </div>
  ),
};