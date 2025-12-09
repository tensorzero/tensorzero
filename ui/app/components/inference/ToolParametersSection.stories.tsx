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
    providerTools: [],
  },
};

export const WithAllowedTools: Story = {
  args: {
    allowedTools: ["search_wikipedia", "get_weather", "calculate"],
    providerTools: [],
  },
};

export const WithToolChoice: Story = {
  args: {
    toolChoice: "auto",
    providerTools: [],
  },
};

export const WithToolChoiceRequired: Story = {
  args: {
    toolChoice: "required",
    providerTools: [],
  },
};

export const WithToolChoiceNone: Story = {
  args: {
    toolChoice: "none",
    providerTools: [],
  },
};

export const WithToolChoiceSpecific: Story = {
  args: {
    toolChoice: { specific: "search_wikipedia" },
    providerTools: [],
  },
};

export const WithParallelToolCallsEnabled: Story = {
  args: {
    parallelToolCalls: true,
    providerTools: [],
  },
};

export const WithParallelToolCallsDisabled: Story = {
  args: {
    parallelToolCalls: false,
    providerTools: [],
  },
};

export const WithSingleFunctionTool: Story = {
  args: {
    additionalTools: [
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
    providerTools: [],
  },
};

export const WithMultipleFunctionTools: Story = {
  args: {
    additionalTools: [
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
    providerTools: [],
  },
};

export const WithOpenAICustomTool: Story = {
  args: {
    additionalTools: [
      {
        type: "openai_custom",
        name: "code_interpreter",
        description: "Executes Python code and returns the result.",
        format: {
          type: "text",
        },
      },
    ],
    providerTools: [],
  },
};

export const WithSingleProviderTool: Story = {
  args: {
    providerTools: [
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
    providerTools: [
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
    providerTools: [
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
    allowedTools: ["search_wikipedia", "get_weather"],
    additionalTools: [
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
    toolChoice: "auto",
    parallelToolCalls: true,
    providerTools: [
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
