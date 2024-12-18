export type ModelOption = {
  displayName: string;
  name: string;
  provider: "openai" | "fireworks" | "mistral";
};

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
