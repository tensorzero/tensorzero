import type { Meta, StoryObj } from "@storybook/react-vite";
import VariantTemplate from "./VariantTemplate";
import type { VariantConfig } from "~/types/tensorzero";

const meta = {
  title: "Variant Detail Page/VariantTemplate",
  component: VariantTemplate,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof VariantTemplate>;

export default meta;
type Story = StoryObj<typeof meta>;

export const CustomNamedTemplates: Story = {
  args: {
    variantConfig: {
      type: "chat_completion",
      weight: 1.0,
      model: "gpt-4o-mini",
      templates: {
        greeting_template: {
          template: {
            path: "greeting.minijinja",
            contents: "Hello {{ name }}! Welcome to {{ place }}.",
          },
          schema: null,
          legacy_definition: false,
        },
        analysis_prompt: {
          template: {
            path: "analysis.minijinja",
            contents: `Analyze the following data:
{{ data }}

Focus on:
{% for aspect in aspects %}
- {{ aspect }}
{% endfor %}`,
          },
          schema: null,
          legacy_definition: false,
        },
        fun_fact_topic: {
          template: {
            path: "fun_fact.minijinja",
            contents: "Share a fun fact about: {{ topic }}",
          },
          schema: null,
          legacy_definition: false,
        },
      },
      temperature: 0.7,
      max_tokens: 1024,
      retries: {
        num_retries: 3,
        max_delay_s: 10,
      },
    } as VariantConfig,
  },
};

export const LegacyTemplates: Story = {
  args: {
    variantConfig: {
      type: "chat_completion",
      weight: 1.0,
      model: "gpt-4o-mini",
      templates: {
        system: {
          template: {
            path: "system.minijinja",
            contents:
              "You are a helpful AI assistant. Today's date is {{ date }}.",
          },
          schema: null,
          legacy_definition: true,
        },
        user: {
          template: {
            path: "user.minijinja",
            contents: "User query: {{ query }}\n\nContext: {{ context }}",
          },
          schema: null,
          legacy_definition: true,
        },
        assistant: {
          template: {
            path: "assistant.minijinja",
            contents:
              "Based on the context, here's my response: {{ response }}",
          },
          schema: null,
          legacy_definition: true,
        },
      },
      temperature: 0.7,
      max_tokens: 2048,
      retries: {
        num_retries: 3,
        max_delay_s: 10,
      },
    } as VariantConfig,
  },
};

export const NonLegacyTemplates: Story = {
  args: {
    variantConfig: {
      type: "chat_completion",
      weight: 1.0,
      model: "gpt-4o-mini",
      templates: {
        system: {
          template: {
            path: "system.minijinja",
            contents:
              "You are a helpful AI assistant. Today's date is {{ date }}.",
          },
          schema: null,
          legacy_definition: false,
        },
        user: {
          template: {
            path: "user.minijinja",
            contents: "User query: {{ query }}\n\nContext: {{ context }}",
          },
          schema: null,
          legacy_definition: false,
        },
        assistant: {
          template: {
            path: "assistant.minijinja",
            contents:
              "Based on the context, here's my response: {{ response }}",
          },
          schema: null,
          legacy_definition: false,
        },
      },
      temperature: 0.7,
      max_tokens: 2048,
      retries: {
        num_retries: 3,
        max_delay_s: 10,
      },
    } as VariantConfig,
  },
};

export const EmptyState: Story = {
  args: {
    variantConfig: {
      type: "chat_completion",
      weight: 1.0,
      model: "gpt-4o-mini",
      templates: {},
      retries: {
        num_retries: 1,
        max_delay_s: 10,
      },
    } as VariantConfig,
  },
};

export const MixedPopulatedAndEmpty: Story = {
  args: {
    variantConfig: {
      type: "chat_completion",
      weight: 1.0,
      model: "gpt-4o-mini",
      templates: {
        greeting_template: {
          template: {
            path: "greeting.minijinja",
            contents: "Hello {{ name }}!",
          },
          schema: null,
          legacy_definition: false,
        },
        empty_template: {
          template: {
            path: "empty.minijinja",
            contents: "",
          },
          schema: null,
          legacy_definition: false,
        },
        analysis_prompt: {
          template: {
            path: "analysis.minijinja",
            contents: "Analyze: {{ data }}",
          },
          schema: null,
          legacy_definition: false,
        },
      },
      retries: {
        num_retries: 1,
        max_delay_s: 10,
      },
    } as VariantConfig,
  },
};

export const MixedLegacyAndCustom: Story = {
  args: {
    variantConfig: {
      type: "chat_completion",
      weight: 1.0,
      model: "gpt-4o-mini",
      templates: {
        system: {
          template: {
            path: "system.minijinja",
            contents: "You are a helpful AI assistant.",
          },
          schema: null,
          legacy_definition: true,
        },
        greeting_template: {
          template: {
            path: "greeting.minijinja",
            contents: "Hello {{ name }}!",
          },
          schema: null,
          legacy_definition: false,
        },
        analysis_prompt: {
          template: {
            path: "analysis.minijinja",
            contents: "Analyze: {{ data }}",
          },
          schema: null,
          legacy_definition: false,
        },
      },
      temperature: 0.7,
      max_tokens: 1024,
      retries: {
        num_retries: 3,
        max_delay_s: 10,
      },
    } as VariantConfig,
  },
};

export const DICLVariant: Story = {
  args: {
    variantConfig: {
      type: "dicl",
      weight: 1.0,
      embedding_model: "text-embedding-3-small",
      k: 10,
      model: "gpt-4o-mini",
      system_instructions: {
        __tensorzero_remapped_path: "system_instructions.minijinja",
        __data: "You are a helpful assistant for entity extraction.",
      },
      json_mode: "strict",
      retries: {
        num_retries: 3,
        max_delay_s: 10,
      },
    } as VariantConfig,
  },
};
