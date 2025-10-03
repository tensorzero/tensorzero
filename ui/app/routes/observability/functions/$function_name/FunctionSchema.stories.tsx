import type { Meta, StoryObj } from "@storybook/react-vite";
import FunctionSchema from "./FunctionSchema";
import type { FunctionConfig } from "tensorzero-node";

const meta = {
  title: "Function Detail Page/FunctionSchema",
  component: FunctionSchema,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof FunctionSchema>;

export default meta;
type Story = StoryObj<typeof meta>;

export const CustomNamedSchemas: Story = {
  args: {
    functionConfig: {
      type: "chat",
      description: "Test function with custom-named schemas",
      tools: [],
      tool_choice: null,
      parallel_tool_calls: null,
      schemas: {
        greeting_template: {
          value: {
            type: "object",
            properties: {
              name: { type: "string" },
              place: { type: "string" },
              day_of_week: { type: "string" },
            },
            required: ["name", "place", "day_of_week"],
          },
        },
        analysis_prompt: {
          value: {
            type: "object",
            properties: {
              data: { type: "string" },
              aspects: {
                type: "array",
                items: { type: "string" },
              },
            },
            required: ["data", "aspects"],
          },
        },
        fun_fact_topic: {
          value: {
            type: "object",
            properties: {
              topic: { type: "string" },
            },
            required: ["topic"],
          },
        },
      },
      variants: {},
    } as unknown as FunctionConfig,
  },
};

export const LegacySchemas: Story = {
  args: {
    functionConfig: {
      type: "chat",
      description:
        "Test function with legacy schema names (system/user/assistant)",
      schemas: {
        system: {
          value: {
            type: "object",
            properties: {
              context: { type: "string" },
              language: { type: "string" },
            },
            required: ["context"],
          },
        },
        user: {
          value: {
            type: "object",
            properties: {
              query: { type: "string" },
              max_results: { type: "number" },
            },
            required: ["query"],
          },
        },
        assistant: {
          value: {
            type: "object",
            properties: {
              response: { type: "string" },
              confidence: { type: "number" },
            },
            required: ["response"],
          },
        },
      },
      variants: {},
    } as unknown as FunctionConfig,
  },
};

export const JsonFunctionWithOutput: Story = {
  args: {
    functionConfig: {
      type: "json",
      description: "JSON function with output schema",
      tools: [],
      tool_choice: null,
      parallel_tool_calls: null,
      implicit_tool_call_config: null,
      schemas: {
        system: {
          value: {
            type: "object",
            properties: {
              instructions: { type: "string" },
            },
          },
        },
      },
      output_schema: {
        value: {
          type: "object",
          properties: {
            entities: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  name: { type: "string" },
                  type: { type: "string" },
                  confidence: { type: "number" },
                },
                required: ["name", "type"],
              },
            },
          },
          required: ["entities"],
        },
      },
      variants: {},
    } as unknown as FunctionConfig,
  },
};

export const EmptyState: Story = {
  args: {
    functionConfig: {
      type: "chat",
      description: "Function with no schemas",
      schemas: {},
      tools: null,
      tool_choice: null,
      parallel_tool_calls: null,
      variants: {},
    } as unknown as FunctionConfig,
  },
};
