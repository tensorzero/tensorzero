import type { Meta, StoryObj } from "@storybook/react-vite";
import { ToolParametersSection } from "./ToolParametersSection";

const meta: Meta<typeof ToolParametersSection> = {
  title: "Inference/ToolParametersSection",
  component: ToolParametersSection,
  parameters: {
    layout: "centered",
  },
  decorators: [
    (Story) => (
      <div className="w-[600px] p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    provider_tools: [],
  },
};

export const WithAllowedTools: Story = {
  args: {
    allowed_tools: ["search_wikipedia", "get_weather", "calculate"],
    provider_tools: [],
  },
};

export const WithToolChoice: Story = {
  args: {
    tool_choice: "auto",
    provider_tools: [],
  },
};

export const WithToolChoiceRequired: Story = {
  args: {
    tool_choice: "required",
    provider_tools: [],
  },
};

export const WithToolChoiceNone: Story = {
  args: {
    tool_choice: "none",
    provider_tools: [],
  },
};

export const WithToolChoiceSpecific: Story = {
  args: {
    tool_choice: { specific: "search_wikipedia" },
    provider_tools: [],
  },
};

export const WithParallelToolCallsEnabled: Story = {
  args: {
    parallel_tool_calls: true,
    provider_tools: [],
  },
};

export const WithParallelToolCallsDisabled: Story = {
  args: {
    parallel_tool_calls: false,
    provider_tools: [],
  },
};

export const WithSingleFunctionTool: Story = {
  args: {
    additional_tools: [
      {
        type: "function",
        name: "search_wikipedia",
        description:
          "Search Wikipedia for pages that match the query. Returns a list of page titles.",
        parameters: {
          $schema: "http://json-schema.org/draft-07/schema#",
          type: "object",
          properties: {
            query: {
              type: "string",
              description:
                'The query to search Wikipedia for (e.g. "machine learning").',
            },
          },
          required: ["query"],
          additionalProperties: false,
        },
        strict: true,
      },
    ],
    provider_tools: [],
  },
};

export const WithMultipleFunctionTools: Story = {
  args: {
    additional_tools: [
      {
        type: "function",
        name: "search_wikipedia",
        description:
          "Search Wikipedia for pages that match the query. Returns a list of page titles.",
        parameters: {
          $schema: "http://json-schema.org/draft-07/schema#",
          type: "object",
          properties: {
            query: {
              type: "string",
              description:
                'The query to search Wikipedia for (e.g. "machine learning").',
            },
          },
          required: ["query"],
          additionalProperties: false,
        },
        strict: true,
      },
      {
        type: "function",
        name: "get_weather",
        description: "Get the current weather for a location.",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and country, e.g. San Francisco, CA",
            },
            unit: {
              type: "string",
              enum: ["celsius", "fahrenheit"],
              description: "The temperature unit to use",
            },
          },
          required: ["location"],
        },
        strict: false,
      },
      {
        type: "function",
        name: "calculate",
        description: "Perform a mathematical calculation.",
        parameters: {
          type: "object",
          properties: {
            expression: {
              type: "string",
              description: "The mathematical expression to evaluate",
            },
          },
          required: ["expression"],
        },
        strict: false,
      },
    ],
    provider_tools: [],
  },
};

export const WithOpenAICustomTool: Story = {
  args: {
    additional_tools: [
      {
        type: "openai_custom",
        name: "code_interpreter",
        description: "Executes Python code and returns the result.",
        format: {
          type: "text",
        },
      },
    ],
    provider_tools: [],
  },
};

export const WithSingleProviderTool: Story = {
  args: {
    provider_tools: [
      {
        scope: null,
        tool: {
          type: "web_search_preview",
          search_context_size: "medium",
        },
      },
    ],
  },
};

export const WithMultipleProviderTools: Story = {
  args: {
    provider_tools: [
      {
        scope: { model_name: "gpt-4o", provider_name: "openai" },
        tool: {
          type: "web_search_preview",
          search_context_size: "medium",
        },
      },
      {
        scope: { model_name: "claude-3-5-sonnet", provider_name: "anthropic" },
        tool: {
          type: "computer_use",
          display_width: 1920,
          display_height: 1080,
        },
      },
    ],
  },
};

export const WithProviderToolScoped: Story = {
  args: {
    provider_tools: [
      {
        scope: { model_name: "gpt-4o" },
        tool: {
          type: "file_search",
          vector_store_ids: ["vs_123", "vs_456"],
        },
      },
    ],
  },
};

export const Complete: Story = {
  args: {
    allowed_tools: ["search_wikipedia", "get_weather"],
    additional_tools: [
      {
        type: "function",
        name: "calculate",
        description: "Perform a mathematical calculation.",
        parameters: {
          type: "object",
          properties: {
            expression: {
              type: "string",
              description: "The mathematical expression to evaluate",
            },
          },
          required: ["expression"],
        },
        strict: true,
      },
    ],
    tool_choice: "auto",
    parallel_tool_calls: true,
    provider_tools: [
      {
        scope: null,
        tool: {
          type: "web_search_preview",
          search_context_size: "medium",
        },
      },
    ],
  },
};
