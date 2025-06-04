import { z } from "zod";
import { stringify } from "smol-toml";

// Base provider configs
export const AnthropicProviderConfigSchema = z.object({
  type: z.literal("anthropic"),
  model_name: z.string(),
});
export type AnthropicProviderConfig = z.infer<
  typeof AnthropicProviderConfigSchema
>;

export const AWSBedrockProviderConfigSchema = z.object({
  type: z.literal("aws_bedrock"),
  model_id: z.string(),
  region: z.string().optional(),
});
export type AWSBedrockProviderConfig = z.infer<
  typeof AWSBedrockProviderConfigSchema
>;

export const AWSSagemakerProviderConfigSchema = z.object({
  type: z.literal("aws_sagemaker"),
  endpoint_name: z.string(),
  model_name: z.string(),
  hosted_provider: z.string(),
  region: z.string().optional(),
});
export type AWSSagemakerProviderConfig = z.infer<
  typeof AWSSagemakerProviderConfigSchema
>;

export const AzureProviderConfigSchema = z.object({
  type: z.literal("azure"),
  deployment_id: z.string(),
  endpoint: z.string().url(),
});
export type AzureProviderConfig = z.infer<typeof AzureProviderConfigSchema>;

export const DeepSeekProviderConfigSchema = z.object({
  type: z.literal("deepseek"),
  model_name: z.string(),
});
export type DeepSeekProviderConfig = z.infer<
  typeof DeepSeekProviderConfigSchema
>;

export const DummyProviderConfigSchema = z.object({
  type: z.literal("dummy"),
  model_name: z.string(),
});
export type DummyProviderConfig = z.infer<typeof DummyProviderConfigSchema>;

export const FireworksProviderConfigSchema = z.object({
  type: z.literal("fireworks"),
  model_name: z.string(),
});
export type FireworksProviderConfig = z.infer<
  typeof FireworksProviderConfigSchema
>;

export const GCPVertexAnthropicProviderConfigSchema = z.object({
  type: z.literal("gcp_vertex_anthropic"),
  model_id: z.string(),
  location: z.string(),
  project_id: z.string(),
});
export type GCPVertexAnthropicProviderConfig = z.infer<
  typeof GCPVertexAnthropicProviderConfigSchema
>;

export const GCPVertexGeminiProviderConfigSchema = z.object({
  type: z.literal("gcp_vertex_gemini"),
  model_id: z.string().optional(), // Exactly one of model_id or endpoint_id must be provided
  endpoint_id: z.string().optional(),
  location: z.string(),
  project_id: z.string(),
});
export type GCPVertexGeminiProviderConfig = z.infer<
  typeof GCPVertexGeminiProviderConfigSchema
>;

export const GoogleAIStudioGeminiProviderConfigSchema = z.object({
  type: z.literal("google_ai_studio_gemini"),
  model_name: z.string(),
});
export type GoogleAIStudioGeminiProviderConfig = z.infer<
  typeof GoogleAIStudioGeminiProviderConfigSchema
>;

export const GroqProviderConfigSchema = z.object({
  type: z.literal("groq"),
  model_name: z.string(),
});
export type GroqProviderConfig = z.infer<typeof GroqProviderConfigSchema>;

export const HyperbolicProviderConfigSchema = z.object({
  type: z.literal("hyperbolic"),
  model_name: z.string(),
});
export type HyperbolicProviderConfig = z.infer<
  typeof HyperbolicProviderConfigSchema
>;

export const MistralProviderConfigSchema = z.object({
  type: z.literal("mistral"),
  model_name: z.string(),
});
export type MistralProviderConfig = z.infer<typeof MistralProviderConfigSchema>;

export const OpenAIProviderConfigSchema = z.object({
  type: z.literal("openai"),
  model_name: z.string(),
  api_base: z.string().url().optional(),
});
export type OpenAIProviderConfig = z.infer<typeof OpenAIProviderConfigSchema>;

export const OpenRouterProviderConfigSchema = z.object({
  type: z.literal("openrouter"),
  model_name: z.string(),
  api_base: z.string().url().optional(),
});
export type OpenRouterProviderConfig = z.infer<
  typeof OpenRouterProviderConfigSchema
>;

export const TGIProviderConfigSchema = z.object({
  type: z.literal("tgi"),
  api_base: z.string().url(),
});
export type TGIProviderConfig = z.infer<typeof TGIProviderConfigSchema>;

export const SGLangProviderConfigSchema = z.object({
  type: z.literal("sglang"),
  model_name: z.string(),
  api_base: z.string().url(),
});
export type SGLangProviderConfig = z.infer<typeof SGLangProviderConfigSchema>;

export const TogetherProviderConfigSchema = z.object({
  type: z.literal("together"),
  model_name: z.string(),
});
export type TogetherProviderConfig = z.infer<
  typeof TogetherProviderConfigSchema
>;

export const VLLMProviderConfigSchema = z.object({
  type: z.literal("vllm"),
  model_name: z.string(),
  api_base: z.string().url(),
});
export type VLLMProviderConfig = z.infer<typeof VLLMProviderConfigSchema>;

export const XAIProviderConfigSchema = z.object({
  type: z.literal("xai"),
  model_name: z.string(),
});
export type XAIProviderConfig = z.infer<typeof XAIProviderConfigSchema>;

// Union of all provider configs
export const ProviderConfigSchema = z.discriminatedUnion("type", [
  AnthropicProviderConfigSchema,
  AWSBedrockProviderConfigSchema,
  AWSSagemakerProviderConfigSchema,
  AzureProviderConfigSchema,
  DeepSeekProviderConfigSchema,
  DummyProviderConfigSchema,
  FireworksProviderConfigSchema,
  GCPVertexAnthropicProviderConfigSchema,
  GCPVertexGeminiProviderConfigSchema,
  GoogleAIStudioGeminiProviderConfigSchema,
  GroqProviderConfigSchema,
  HyperbolicProviderConfigSchema,
  MistralProviderConfigSchema,
  OpenAIProviderConfigSchema,
  OpenRouterProviderConfigSchema,
  SGLangProviderConfigSchema,
  TGIProviderConfigSchema,
  TogetherProviderConfigSchema,
  VLLMProviderConfigSchema,
  XAIProviderConfigSchema,
]);

export type ProviderConfig = z.infer<typeof ProviderConfigSchema>;
export type ProviderType = ProviderConfig["type"];

export function createProviderConfig(
  type: ProviderType,
  model_name: string,
): ProviderConfig {
  switch (type) {
    case "anthropic":
    case "dummy":
    case "fireworks":
    case "mistral":
    case "together":
    case "google_ai_studio_gemini":
    case "openai":
    case "openrouter":
      return { type, model_name };
    case "aws_bedrock":
      return { type, model_id: model_name };
    default:
      throw new Error(`Provider ${type} requires additional configuration`);
  }
}

// Model config schema
export const ModelConfigSchema = z.object({
  // Array of provider names for routing/fallback order
  routing: z.array(z.string()),
  // Map of provider name to provider config
  providers: z.record(z.string(), ProviderConfigSchema),
});

export type ModelConfig = z.infer<typeof ModelConfigSchema>;

// Embedding provider config schema
export const EmbeddingProviderConfigSchema = z.discriminatedUnion("type", [
  OpenAIProviderConfigSchema,
]);

export type EmbeddingProviderConfig = z.infer<
  typeof EmbeddingProviderConfigSchema
>;

// Embedding model config schema
export const EmbeddingModelConfigSchema = z.object({
  // Array of provider names for routing/fallback order
  routing: z.array(z.string()),
  // Map of provider name to provider config
  providers: z.record(z.string(), EmbeddingProviderConfigSchema),
});

export type EmbeddingModelConfig = z.infer<typeof EmbeddingModelConfigSchema>;

// Helper functions for model config
export type FullyQualifiedModelConfig = {
  models: {
    [key: string]: ModelConfig;
  };
};

export function get_fine_tuned_model_config(
  model_name: string,
  model_provider_type: ProviderType,
) {
  const providerConfig: ProviderConfig = createProviderConfig(
    model_provider_type,
    model_name,
  );
  const modelConfig: ModelConfig = {
    routing: [model_name],
    providers: {
      [model_name]: providerConfig,
    },
  };
  const fullyQualifiedModelConfig: FullyQualifiedModelConfig = {
    models: {
      [model_name]: modelConfig,
    },
  };
  return fullyQualifiedModelConfig;
}

export function dump_model_config(modelConfig: FullyQualifiedModelConfig) {
  const rawSerializedModelConfig = stringify(modelConfig);
  const lines = rawSerializedModelConfig.split("\n");
  const linesWithoutFirst = lines.slice(1);
  linesWithoutFirst.splice(3, 1);
  const trimmedSerializedModelConfig = linesWithoutFirst.join("\n");
  return trimmedSerializedModelConfig;
}
