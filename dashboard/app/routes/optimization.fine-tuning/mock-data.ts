import type { FunctionConfig } from "~/../../utils/config/function";
import type { MetricConfig } from "~/../../utils/config/metric";
import type { ModelConfig } from "~/../../utils/config/models";
import type { VariantConfig } from "~/../../utils/config/variant";

export const functions: Record<string, FunctionConfig> = {
  "sentiment-analysis": {
    type: "json",
    variants: {},
    output_schema: "schema.json",
  },
  "customer-support": {
    type: "chat",
    variants: {},
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
  },
  "code-review": {
    type: "chat",
    variants: {},
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
  },
  "data-extraction": {
    type: "json",
    variants: {},
    output_schema: "schema.json",
  },
  "meeting-summarizer": {
    type: "chat",
    variants: {},
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
  },
  "entity-recognition": {
    type: "json",
    variants: {},
    output_schema: "schema.json",
  },
  "translation-helper": {
    type: "chat",
    variants: {},
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
  },
  "json-validator": {
    type: "json",
    variants: {},
    output_schema: "schema.json",
  },
  "tutor-assistant": {
    type: "chat",
    variants: {},
    tools: [],
    tool_choice: "none",
    parallel_tool_calls: false,
  },
  "log-parser": {
    type: "json",
    variants: {},
    output_schema: "schema.json",
  },
};

export const metrics: Record<string, MetricConfig> = {
  accuracy: {
    type: "float",
    optimize: "max",
    level: "inference",
  },
  latency: {
    type: "float",
    optimize: "min",
    level: "inference",
  },
  success_rate: {
    type: "boolean",
    optimize: "max",
    level: "episode",
  },
  token_efficiency: {
    type: "float",
    optimize: "max",
    level: "inference",
  },
  completion_rate: {
    type: "boolean",
    optimize: "max",
    level: "episode",
  },
};

export type ModelOption = {
  name: string;
  provider: "openai" | "anthropic" | "mistral";
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
];

export const promptTemplates: Record<string, VariantConfig> = {
  variant1: {
    type: "chat_completion",
    model: "gpt-4o",
    user_template: "user_template.txt",
    json_mode: "on",
  },
  variant2: {
    type: "chat_completion",
    model: "gpt-4o-mini",
    user_template: "user_template.txt",
    json_mode: "on",
  },
  variant3: {
    type: "chat_completion",
    model: "gpt-4o-large",
    user_template: "user_template.txt",
    json_mode: "on",
  },
  variant4: {
    type: "chat_completion",
    model: "gpt-4o-turbo",
    user_template: "user_template.txt",
    json_mode: "on",
  },
  variant5: {
    type: "chat_completion",
    model: "gpt-4o",
    user_template: "user_template.txt",
    json_mode: "on",
  },
};

export const promptTemplateDetails = {
  system: `You are an expert AI assistant specializing in {{ domain }}. Your responses should be detailed, accurate, and tailored to the user's needs. You have extensive knowledge in {{ domain }} and related fields, allowing you to provide comprehensive explanations and practical solutions. When answering questions, consider both theoretical foundations and real-world applications to deliver the most valuable insights to users.`,
  user: `I'm working on a project in {{ domain }} and need assistance with {{ specific_task }}. The main challenge I'm facing is {{ challenge }}, and I've already tried {{ previous_attempts }}. My goal is to {{ goal }}, and I'm particularly interested in understanding how this relates to {{ related_topic }}. Could you provide a detailed explanation and suggest some practical approaches?`,
  assistant: null,
};
