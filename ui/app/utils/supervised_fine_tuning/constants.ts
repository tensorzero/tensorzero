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

/**
 * Model token limits for training examples
 * Reference: https://platform.openai.com/docs/guides/fine-tuning/token-counting
 * As of 20250305
 */
export const MODEL_TOKEN_LIMITS: Record<string, number> = {
  "gpt-4o-2024-08-06": 65536,
  "gpt-4o-mini-2024-07-18": 65536,
  "gpt-4-0613": 8192,
  "gpt-3.5-turbo-0125": 16385,
  "gpt-3.5-turbo-1106": 16385,
  "gpt-3.5-turbo-0613": 4096,
};
