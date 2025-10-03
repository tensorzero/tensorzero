import type { Meta, StoryObj } from "@storybook/react-vite";
import { FunctionSelector } from "./FunctionSelector";
import type { FunctionConfig } from "tensorzero-node";
import { useState } from "react";
import { DEFAULT_FUNCTION } from "~/utils/constants";

const meta = {
  title: "UI/FunctionSelector",
  component: FunctionSelector,
  decorators: [
    (Story) => (
      <div className="w-sm p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof FunctionSelector>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockFunctions: Record<string, FunctionConfig> = {
  [DEFAULT_FUNCTION]: {
    type: "chat",
    variants: {},
    schemas: {},
    tools: [],
    tool_choice: "auto",
    parallel_tool_calls: null,
    description: "Default chat function",
    experimentation: { type: "uniform" },
  },
  "chat-function": {
    type: "chat",
    variants: {},
    schemas: {},
    tools: ["calculator", "weather"],
    tool_choice: "auto",
    parallel_tool_calls: true,
    description: "Chat function with tools",
    experimentation: { type: "uniform" },
  },
  "json-extractor": {
    type: "json",
    variants: {},
    schemas: {},
    output_schema: {
      value: null,
    },
    implicit_tool_call_config: {
      tools_available: [],
      tool_choice: "auto",
      parallel_tool_calls: false,
    },
    description: "Extract structured data from text",
    experimentation: { type: "uniform" },
  },
  "sentiment-analyzer": {
    type: "json",
    variants: {},
    schemas: {},
    output_schema: {
      value: null,
    },
    implicit_tool_call_config: {
      tools_available: [],
      tool_choice: "auto",
      parallel_tool_calls: false,
    },
    description: "Analyze sentiment of text",
    experimentation: { type: "uniform" },
  },
};

export const Default: Story = {
  args: {
    selected: null,
    functions: mockFunctions,
    hideDefaultFunction: false,
  },
  render: function DefaultStory(args) {
    const [selected, setSelected] = useState<string | null>(args.selected);

    return (
      <FunctionSelector {...args} selected={selected} onSelect={setSelected} />
    );
  },
};

export const WithDefaultHidden: Story = {
  args: {
    selected: "chat-function",
    functions: mockFunctions,
    hideDefaultFunction: true,
  },
  render: function WithSelectionStory(args) {
    const [selected, setSelected] = useState<string | null>(args.selected);

    return (
      <FunctionSelector {...args} selected={selected} onSelect={setSelected} />
    );
  },
};

export const EmptyFunctions: Story = {
  args: {
    selected: null,
    functions: {},
    hideDefaultFunction: false,
  },
  render: function EmptyFunctionsStory(args) {
    const [selected, setSelected] = useState<string | null>(args.selected);

    return (
      <FunctionSelector {...args} selected={selected} onSelect={setSelected} />
    );
  },
};
