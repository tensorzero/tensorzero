import { z } from "zod";

export const ModelOptionSchema = z.object({
  displayName: z.string().nonempty("Model name is required"),
  name: z.string().nonempty("Model name is required"),
  provider: z.enum(["openai", "fireworks", "mistral"]),
});

export type ModelOption = z.infer<typeof ModelOptionSchema>;

export const models: ModelOption[] = [
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
