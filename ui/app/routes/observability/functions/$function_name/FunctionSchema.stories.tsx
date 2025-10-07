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
          schema: {
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
          legacy_definition: false,
        },
        analysis_prompt: {
          schema: {
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
          legacy_definition: false,
        },
        fun_fact_topic: {
          schema: {
            value: {
              type: "object",
              properties: {
                topic: { type: "string" },
              },
              required: ["topic"],
            },
          },
          legacy_definition: false,
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
        "Test function with legacy schema definitions (marked with legacy_definition: true)",
      schemas: {
        system: {
          schema: {
            value: {
              type: "object",
              properties: {
                context: { type: "string" },
                language: { type: "string" },
              },
              required: ["context"],
            },
          },
          legacy_definition: true,
        },
        user: {
          schema: {
            value: {
              type: "object",
              properties: {
                query: { type: "string" },
                max_results: { type: "number" },
              },
              required: ["query"],
            },
          },
          legacy_definition: true,
        },
        assistant: {
          schema: {
            value: {
              type: "object",
              properties: {
                response: { type: "string" },
                confidence: { type: "number" },
              },
              required: ["response"],
            },
          },
          legacy_definition: true,
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
          schema: {
            value: {
              type: "object",
              properties: {
                instructions: { type: "string" },
              },
            },
          },
          legacy_definition: false,
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

export const NonLegacySchemas: Story = {
  args: {
    functionConfig: {
      type: "chat",
      description:
        "Test function with system/user/assistant names but NOT legacy (legacy_definition: false)",
      schemas: {
        system: {
          schema: {
            value: {
              type: "object",
              properties: {
                context: { type: "string" },
              },
            },
          },
          legacy_definition: false,
        },
        user: {
          schema: {
            value: {
              type: "object",
              properties: {
                query: { type: "string" },
              },
            },
          },
          legacy_definition: false,
        },
        assistant: {
          schema: {
            value: {
              type: "object",
              properties: {
                response: { type: "string" },
              },
            },
          },
          legacy_definition: false,
        },
      },
      variants: {},
    } as unknown as FunctionConfig,
  },
};

export const MixedLegacyAndCustom: Story = {
  args: {
    functionConfig: {
      type: "chat",
      description:
        "Test function with both legacy and custom schemas to show badges only on legacy",
      schemas: {
        system: {
          schema: {
            value: {
              type: "object",
              properties: {
                instructions: { type: "string" },
              },
            },
          },
          legacy_definition: true,
        },
        greeting_template: {
          schema: {
            value: {
              type: "object",
              properties: {
                name: { type: "string" },
              },
            },
          },
          legacy_definition: false,
        },
        analysis_prompt: {
          schema: {
            value: {
              type: "object",
              properties: {
                data: { type: "string" },
              },
            },
          },
          legacy_definition: false,
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
