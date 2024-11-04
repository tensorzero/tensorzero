import { z } from "zod";

// Base provider configs
const AnthropicProviderConfig = z.object({
  type: z.literal("anthropic"),
  model_name: z.string(),
});

const AWSBedrockProviderConfig = z.object({
  type: z.literal("aws_bedrock"),
  model_id: z.string(),
  region: z.string().optional(),
});

const AzureProviderConfig = z.object({
  type: z.literal("azure"),
  deployment_id: z.string(),
  endpoint: z.string().url(),
});

const FireworksProviderConfig = z.object({
  type: z.literal("fireworks"),
  model_name: z.string(),
});

const GCPVertexAnthropicProviderConfig = z.object({
  type: z.literal("gcp_vertex_anthropic"),
  model_id: z.string(),
  location: z.string(),
  project_id: z.string(),
});

const GCPVertexGeminiProviderConfig = z.object({
  type: z.literal("gcp_vertex_gemini"),
  model_id: z.string(),
  location: z.string(),
  project_id: z.string(),
});

const GoogleAIStudioGeminiProviderConfig = z.object({
  type: z.literal("google_ai_studio_gemini"),
  model_name: z.string(),
});

const MistralProviderConfig = z.object({
  type: z.literal("mistral"),
  model_name: z.string(),
});

const OpenAIProviderConfig = z.object({
  type: z.literal("openai"),
  model_name: z.string(),
  api_base: z.string().url().optional(),
});

const TogetherProviderConfig = z.object({
  type: z.literal("together"),
  model_name: z.string(),
});

const VLLMProviderConfig = z.object({
  type: z.literal("vllm"),
  model_name: z.string(),
  api_base: z.string().url(),
});

// Union of all provider configs
export const ProviderConfig = z.discriminatedUnion("type", [
  AnthropicProviderConfig,
  AWSBedrockProviderConfig,
  AzureProviderConfig,
  FireworksProviderConfig,
  GCPVertexAnthropicProviderConfig,
  GCPVertexGeminiProviderConfig,
  GoogleAIStudioGeminiProviderConfig,
  MistralProviderConfig,
  OpenAIProviderConfig,
  TogetherProviderConfig,
  VLLMProviderConfig,
]);

export type ProviderConfig = z.infer<typeof ProviderConfig>;

// Model config schema
export const ModelConfig = z.object({
  // Array of provider names for routing/fallback order
  routing: z.array(z.string()),
  // Map of provider name to provider config
  providers: z.record(z.string(), ProviderConfig),
});

export type ModelConfig = z.infer<typeof ModelConfig>;

// Embedding provider config schema
export const EmbeddingProviderConfig = z.discriminatedUnion("type", [
  OpenAIProviderConfig,
]);

export type EmbeddingProviderConfig = z.infer<typeof EmbeddingProviderConfig>;

// Embedding model config schema
export const EmbeddingModelConfig = z.object({
  // Array of provider names for routing/fallback order
  routing: z.array(z.string()),
  // Map of provider name to provider config
  providers: z.record(z.string(), EmbeddingProviderConfig),
});

export type EmbeddingModelConfig = z.infer<typeof EmbeddingModelConfig>;
