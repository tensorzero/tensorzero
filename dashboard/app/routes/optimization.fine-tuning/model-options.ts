export type ModelOption = {
  name: string;
  provider: "openai" | "fireworks" | "mistral";
};

export const models: ModelOption[] = [
  {
    name: "gpt-4o",
    provider: "openai",
  },
  {
    name: "gpt-4o-mini",
    provider: "openai",
  },
  {
    name: "gpt-4o-large",
    provider: "openai",
  },
  {
    name: "gpt-4o-turbo",
    provider: "openai",
  },
  {
    name: "gpt-4o-mini-2024-07-18",
    provider: "openai",
  },
];
