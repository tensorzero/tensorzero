import { z } from "zod";

export const ModelOptionSchema = z.object({
  displayName: z.string().nonempty("Model name is required"),
  name: z.string().nonempty("Model name is required"),
  provider: z.enum(["openai", "fireworks", "gcp_vertex_gemini"]),
});

export type ModelOption = z.infer<typeof ModelOptionSchema>;

export const models: ModelOption[] = [
  // OpenAI
  {
    displayName: "gpt-4.1-2025-04-14",
    name: "gpt-4.1-2025-04-14",
    provider: "openai",
  },
  {
    displayName: "gpt-4.1-mini-2025-04-14",
    name: "gpt-4.1-mini-2025-04-14",
    provider: "openai",
  },
  {
    displayName: "gpt-4o-2024-08-06",
    name: "gpt-4o-2024-08-06",
    provider: "openai",
  },
  {
    displayName: "gpt-4o-mini-2024-07-18",
    name: "gpt-4o-mini-2024-07-18",
    provider: "openai",
  },
  {
    displayName: "gpt-3.5-turbo-0125",
    name: "gpt-3.5-turbo-0125",
    provider: "openai",
  },
  {
    displayName: "gpt-3.5-turbo-1106",
    name: "gpt-3.5-turbo-1106",
    provider: "openai",
  },

  // Fireworks
  {
    displayName: "llama-3.1-8b-instruct",
    name: "accounts/fireworks/models/llama-v3p1-8b-instruct",
    provider: "fireworks",
  },
  {
    displayName: "llama-3.1-70b-instruct",
    name: "accounts/fireworks/models/llama-v3p1-70b-instruct",
    provider: "fireworks",
  },
  {
    displayName: "llama-3.2-3b-instruct",
    name: "accounts/fireworks/models/llama-v3p2-3b-instruct",
    provider: "fireworks",
  },

  // GCP Vertex AI Gemini
  {
    displayName: "gemini-2.5-flash-lite",
    name: "gemini-2.5-flash-lite",
    provider: "gcp_vertex_gemini",
  },
  {
    displayName: "gemini-2.5-pro",
    name: "gemini-2.5-pro",
    provider: "gcp_vertex_gemini",
  },
  {
    displayName: "gemini-2.5-flash",
    name: "gemini-2.5-flash",
    provider: "gcp_vertex_gemini",
  },
  {
    displayName: "gemini-2.0-flash",
    name: "gemini-2.0-flash",
    provider: "gcp_vertex_gemini",
  },
  {
    displayName: "gemini-2.0-flash-lite",
    name: "gemini-2.0-flash-lite",
    provider: "gcp_vertex_gemini",
  },
];
