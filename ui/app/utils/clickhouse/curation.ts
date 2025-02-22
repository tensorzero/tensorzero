import type { MetricConfigLevel } from "../config/metric";
import {
  contentBlockOutputSchema,
  InferenceJoinKey,
  inputSchema,
  jsonInferenceOutputSchema,
} from "./common";
import { z } from "zod";
export function getInferenceJoinKey(
  level: MetricConfigLevel,
): InferenceJoinKey {
  switch (level) {
    case "inference":
      return InferenceJoinKey.ID;
    case "episode":
      return InferenceJoinKey.EPISODE_ID;
  }
}

export const inferenceExampleSchema = z
  .object({
    variant_name: z.string(),
    input: z.string(),
    output: z.string(),
    episode_id: z.string(),
  })
  .strict();
export type InferenceExample = z.infer<typeof inferenceExampleSchema>;

export const parsedChatExampleSchema = inferenceExampleSchema
  .omit({
    input: true,
    output: true,
  })
  .extend({
    input: inputSchema,
    output: z.array(contentBlockOutputSchema),
  })
  .strict();
export type ParsedChatInferenceExample = z.infer<
  typeof parsedChatExampleSchema
>;

export const parsedJsonInferenceExampleSchema = inferenceExampleSchema
  .omit({
    input: true,
    output: true,
  })
  .extend({
    input: inputSchema,
    output: jsonInferenceOutputSchema,
  })
  .strict();
export type ParsedJsonInferenceExample = z.infer<
  typeof parsedJsonInferenceExampleSchema
>;

export type ParsedInferenceExample =
  | ParsedChatInferenceExample
  | ParsedJsonInferenceExample;
