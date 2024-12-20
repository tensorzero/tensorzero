import { z } from "zod";

export const ModelOptionSchema = z.object({
  displayName: z.string(),
  name: z.string(),
  provider: z.enum(["openai", "fireworks", "mistral"]),
});

export type ModelOption = z.infer<typeof ModelOptionSchema>;

// TODO: make a type per model provider containing
// what information is needed in order to start a fine-tuning job.

export const models: ModelOption[] = [
  {
    displayName: "gpt-4o",
    name: "gpt-4o",
    provider: "openai",
  },
  {
    displayName: "gpt-4o-mini",
    name: "gpt-4o-mini",
    provider: "openai",
  },
  {
    displayName: "gpt-4o-large",
    name: "gpt-4o-large",
    provider: "openai",
  },
  {
    displayName: "gpt-4o-turbo",
    name: "gpt-4o-turbo",
    provider: "openai",
  },
  {
    displayName: "gpt-4o-mini-2024-07-18",
    name: "gpt-4o-mini-2024-07-18",
    provider: "openai",
  },
  {
    displayName: "llama-3.1-8b-instruct",
    name: "accounts/fireworks/models/llama-v3p1-8b-instruct",
    provider: "fireworks",
  },
];
