/**
 * Models for fine-tuning and Model token limits for training examples
 * Reference: https://platform.openai.com/docs/guides/fine-tuning/token-counting#which-models-can-be-fine-tuned
 * Reference:https://platform.openai.com/docs/guides/fine-tuning/token-counting#token-limits
 * As of 20250305
 */
export const CURRENT_MODEL_VERSIONS = [
  "gpt-4-0613",
  "gpt-4o-2024-08-06",
  "gpt-4o-mini-2024-07-18",
  "gpt-3.5-turbo-0125",
  "gpt-3.5-turbo-1106",
  // "gpt-3.5-turbo-0613",  //https://platform.openai.com/docs/deprecations#2023-11-06-chat-model-updates
] as const;
export const MODEL_TOKEN_LIMITS: Record<string, number> = {
  "gpt-4-0613": 8192,
  "gpt-4o-2024-08-06": 65536,
  "gpt-4o-mini-2024-07-18": 65536,
  "gpt-3.5-turbo-0125": 16385,
  "gpt-3.5-turbo-1106": 16385,
  // "gpt-3.5-turbo-0613": 4096,
};

export const OPENAI_ROLES = [
  "system",
  "user",
  "assistant",
  "tool",
  "function",
] as const;
export const REQUIRED_ROLES = ["assistant"] as const;
export const RECOMMENDED_ROLES = ["system", "user"] as const;

export const VALID_MESSAGE_KEYS = [
  "role",
  "content",
  "name",
  "function_call",
  "weight",
  "tool_calls",
  "tool_call_id",
] as const;
