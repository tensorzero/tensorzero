import type { TiktokenModel } from "tiktoken";
import { MODEL_TOKEN_LIMITS } from "./constants";

/**
 * Converts model name to tiktoken model name
 * @param model Model name to convert
 * @returns Corresponding tiktoken model name
 */
export function convertToTiktokenModel(model: string): TiktokenModel {
  if (model.startsWith("gpt-4o-") || model.startsWith("gpt-4-")) {
    return "gpt-4";
  }
  if (model.startsWith("gpt-3.5-turbo")) {
    return "gpt-3.5-turbo";
  }

  throw new Error(
    `Unsupported model: ${model}. Supported models are: gpt-4o-*, gpt-4-*, gpt-3.5-turbo*`,
  );
}

/**
 * Gets token limit for a given model
 * @param model Model name
 * @returns Token limit for the model
 */
// export function getModelTokenLimit(model: string): number {
//   if (model in MODEL_TOKEN_LIMITS) {
//     return MODEL_TOKEN_LIMITS[model];
//   }

//   if (model.startsWith("gpt-4o-")) {
//     return MODEL_TOKEN_LIMITS["gpt-4"];
//   }
//   if (model.startsWith("gpt-4-")) {
//     return MODEL_TOKEN_LIMITS["gpt-4"];
//   }
//   if (model.startsWith("gpt-3.5-turbo")) {
//     return MODEL_TOKEN_LIMITS["gpt-3.5-turbo"];
//   }

//   return MODEL_TOKEN_LIMITS["gpt-3.5-turbo"];
// }

export function getModelTokenLimit(model: string): number {
  if (model in MODEL_TOKEN_LIMITS) {
    return MODEL_TOKEN_LIMITS[model];
  }

  if (model.startsWith("gpt-4o-")) {
    return 65536;
  }
  if (model.startsWith("gpt-4-")) {
    return 8192;
  }
  if (model.startsWith("gpt-3.5-turbo")) {
    return 16385;
  }
  return 4096;
}
