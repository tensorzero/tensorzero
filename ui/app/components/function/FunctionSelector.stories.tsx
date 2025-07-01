import type { Meta, StoryObj } from "@storybook/react-vite";
import { FunctionSelector } from "./FunctionSelector";
import type { FunctionSelectorProps } from "./FunctionSelector";
import { useForm, FormProvider } from "react-hook-form";
import type { Config } from "~/utils/config";
import type { Decorator } from "@storybook/react-vite";

// Custom args interface for our stories
interface StoryArgs
  extends Omit<
    FunctionSelectorProps<Record<string, unknown>>,
    "control" | "name"
  > {
  defaultValue?: string;
}

const mockConfig: Config = {
  gateway: {
    observability: {
      enabled: true,
      async_writes: false,
    },
    export: {
      otlp: {
        traces: {
          enabled: false,
        },
      },
    },
    debug: false,
    enable_template_filesystem_access: false,
  },
  models: {},
  embedding_models: {},
  functions: {
    "tensorzero::default": {
      type: "chat",
      variants: {
        default: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.7,
          json_mode: "off",
          weight: 1,
        },
      },
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
    },
    text_generation: {
      type: "chat",
      variants: {
        v1: {
          type: "chat_completion",
          model: "gpt-3.5-turbo",
          temperature: 0.5,
          json_mode: "off",
          weight: 1,
        },
        v2: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.3,
          json_mode: "off",
          weight: 1,
        },
      },
      tools: [],
      tool_choice: "none",
      parallel_tool_calls: false,
      description: "Generate text based on user prompts",
    },
    json_parser: {
      type: "json",
      variants: {
        default: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.1,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Parse and validate JSON structures",
    },
    code_assistant: {
      type: "chat",
      variants: {
        basic: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.2,
          json_mode: "off",
          weight: 1,
        },
        advanced: {
          type: "chat_completion",
          model: "gpt-4-turbo",
          temperature: 0.1,
          json_mode: "off",
          weight: 1,
        },
      },
      tools: ["code_execution", "file_search"],
      tool_choice: "auto",
      parallel_tool_calls: true,
      description: "Assist with coding tasks and debugging",
    },
    data_analyzer: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
    data_analyzer2: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
    data_analyzer3: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
    data_analyzer4: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
    data_analyzer5: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
    data_analyzer6: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
    data_analyzer7: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
    data_analyzer8: {
      type: "json",
      variants: {
        statistical: {
          type: "chat_completion",
          model: "gpt-4",
          temperature: 0.0,
          json_mode: "off",
          weight: 1,
        },
      },
      description: "Analyze data and return structured insights",
    },
  },
  metrics: {},
  tools: {},
  evaluations: {},
};

const FormProviderDecorator: Decorator<StoryArgs> = (_, context) => {
  const form = useForm({
    defaultValues: {
      function_name: context.args.defaultValue || "",
    },
  });

  return (
    <div className="w-96 p-4">
      <FormProvider {...form}>
        <FunctionSelector
          control={form.control}
          name="function_name"
          config={context.args.config}
          inferenceCount={context.args.inferenceCount}
          hide_default_function={context.args.hide_default_function}
        />
      </FormProvider>
    </div>
  );
};

const meta = {
  title: "FunctionSelector",
  component: FunctionSelector,
  decorators: [FormProviderDecorator],
  args: {
    inferenceCount: null,
    hide_default_function: false,
  },
} satisfies Meta<StoryArgs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    config: mockConfig,
  },
};

export const PreSelected: Story = {
  args: {
    config: mockConfig,
    defaultValue: "text_generation",
  },
};

export const Empty: Story = {
  args: {
    config: {
      ...mockConfig,
      functions: {},
    },
  },
};
