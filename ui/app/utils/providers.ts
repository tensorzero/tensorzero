import type { ProviderConfig } from "~/types/tensorzero";

export function formatProvider(provider: ProviderConfig["type"]): {
  name: string;
  className: string;
} {
  switch (provider) {
    case "anthropic":
      return {
        name: "Anthropic",
        className:
          "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
      };
    case "aws_bedrock":
      return {
        name: "AWS Bedrock",
        className:
          "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
      };
    case "aws_sagemaker":
      return {
        name: "AWS Sagemaker",
        className:
          "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
      };
    case "azure":
      return {
        name: "Azure",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "deepseek":
      return {
        name: "DeepSeek",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "dummy":
      return {
        name: "Dummy",
        className:
          "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300",
      };
    case "fireworks":
      return {
        name: "Fireworks",
        className:
          "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300",
      };
    case "gcp_vertex_anthropic":
      return {
        name: "GCP Vertex AI (Anthropic)",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "gcp_vertex_gemini":
      return {
        name: "GCP Vertex AI (Gemini)",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "google_ai_studio_gemini":
      return {
        name: "Google AI Studio",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "groq":
      return {
        name: "Groq",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "hyperbolic":
      return {
        name: "Hyperbolic",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "mistral":
      return {
        name: "Mistral",
        className:
          "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300",
      };
    case "openai":
      return {
        name: "OpenAI",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "openrouter":
      return {
        name: "OpenRouter",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "sglang":
      return {
        name: "SGLang",
        className:
          "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-300",
      };
    case "tensorzero_relay":
      return {
        name: "TensorZero Relay",
        className:
          "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-300",
      };
    case "tgi":
      return {
        name: "TGI",
        className:
          "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-300",
      };
    case "together":
      return {
        name: "Together",
        className:
          "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300",
      };
    case "vllm":
      return {
        name: "vLLM",
        className:
          "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300",
      };
    case "xai":
      return {
        name: "xAI",
        className:
          "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-300",
      };
  }
}
