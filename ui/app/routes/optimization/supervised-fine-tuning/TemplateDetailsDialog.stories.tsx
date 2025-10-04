import type { Meta, StoryObj } from "@storybook/react-vite";
import { TemplateDetailsDialog } from "./TemplateDetailsDialog";
import type { ChatCompletionConfig } from "tensorzero-node";

const meta = {
  title: "Optimization/TemplateDetailsDialog",
  component: TemplateDetailsDialog,
  parameters: {
    layout: "centered",
  },
} satisfies Meta<typeof TemplateDetailsDialog>;

export default meta;
type Story = StoryObj<typeof meta>;

const customTemplateVariants: Record<string, ChatCompletionConfig> = {
  baseline: {
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
    top_p: null,
    max_tokens: 1024,
    presence_penalty: null,
    frequency_penalty: null,
    seed: null,
    stop_sequences: null,
    json_mode: null,
    retries: {
      num_retries: 3,
      max_delay_s: 10,
    },
  },
};

const legacyTemplateVariants: Record<string, ChatCompletionConfig> = {
  baseline: {
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
          contents: "Based on the context, here's my response: {{ response }}",
        },
        schema: null,
        legacy_definition: true,
      },
    },
    temperature: 0.7,
    top_p: null,
    max_tokens: 2048,
    presence_penalty: null,
    frequency_penalty: null,
    seed: null,
    stop_sequences: null,
    json_mode: null,
    retries: {
      num_retries: 3,
      max_delay_s: 10,
    },
  },
};

const nonLegacyTemplateVariants: Record<string, ChatCompletionConfig> = {
  baseline: {
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
          contents: "Based on the context, here's my response: {{ response }}",
        },
        schema: null,
        legacy_definition: false,
      },
    },
    temperature: 0.7,
    top_p: null,
    max_tokens: 2048,
    presence_penalty: null,
    frequency_penalty: null,
    seed: null,
    stop_sequences: null,
    json_mode: null,
    retries: {
      num_retries: 3,
      max_delay_s: 10,
    },
  },
};

const emptyTemplateVariants: Record<string, ChatCompletionConfig> = {
  baseline: {
    weight: 1.0,
    model: "gpt-4o-mini",
    templates: {},
    temperature: null,
    top_p: null,
    max_tokens: null,
    presence_penalty: null,
    frequency_penalty: null,
    seed: null,
    stop_sequences: null,
    json_mode: null,
    retries: {
      num_retries: 1,
      max_delay_s: 10,
    },
  },
};

export const CustomTemplates: Story = {
  args: {
    variant: "baseline",
    disabled: false,
    chatCompletionVariants: customTemplateVariants,
  },
};

export const LegacyTemplates: Story = {
  args: {
    variant: "baseline",
    disabled: false,
    chatCompletionVariants: legacyTemplateVariants,
  },
};

export const NonLegacyTemplates: Story = {
  args: {
    variant: "baseline",
    disabled: false,
    chatCompletionVariants: nonLegacyTemplateVariants,
  },
};

export const EmptyState: Story = {
  args: {
    variant: "baseline",
    disabled: false,
    chatCompletionVariants: emptyTemplateVariants,
  },
};

export const Disabled: Story = {
  args: {
    variant: "baseline",
    disabled: true,
    chatCompletionVariants: customTemplateVariants,
  },
};

export const LongTemplates: Story = {
  args: {
    variant: "baseline",
    disabled: false,
    chatCompletionVariants: {
      baseline: {
        weight: 1.0,
        model: "gpt-4o-mini",
        templates: {
          system: {
            template: {
              path: "system.minijinja",
              contents: `You are an advanced AI assistant specialized in data analysis and reporting.

Your capabilities include:
- Statistical analysis of datasets
- Data visualization recommendations
- Trend identification
- Anomaly detection
- Predictive modeling suggestions

When analyzing data, always:
1. Start with descriptive statistics
2. Look for patterns and trends
3. Identify outliers or anomalies
4. Consider temporal relationships
5. Suggest appropriate visualizations
6. Provide actionable insights

Format your responses clearly with headings, bullet points, and numbered lists where appropriate.`,
            },
            schema: null,
            legacy_definition: false,
          },
          user: {
            template: {
              path: "user.minijinja",
              contents: `Dataset: {{ dataset_name }}

Columns:
{% for column in columns %}
- {{ column.name }} ({{ column.type }}): {{ column.description }}
{% endfor %}

Sample data (first {{ sample_size }} rows):
{{ sample_data }}

Analysis request:
{{ analysis_request }}

Please provide a comprehensive analysis including:
1. Summary statistics
2. Key findings
3. Visualizations to create
4. Recommendations`,
            },
            schema: null,
            legacy_definition: false,
          },
        },
        temperature: 0.5,
        top_p: null,
        max_tokens: 4096,
        presence_penalty: null,
        frequency_penalty: null,
        seed: null,
        stop_sequences: null,
        json_mode: null,
        retries: {
          num_retries: 3,
          max_delay_s: 10,
        },
      },
    },
  },
};

export const MixedLegacyAndCustom: Story = {
  args: {
    variant: "baseline",
    disabled: false,
    chatCompletionVariants: {
      baseline: {
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
              contents: "Hello {{ name }}! Welcome to {{ place }}.",
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
        top_p: null,
        max_tokens: 1024,
        presence_penalty: null,
        frequency_penalty: null,
        seed: null,
        stop_sequences: null,
        json_mode: null,
        retries: {
          num_retries: 3,
          max_delay_s: 10,
        },
      },
    },
  },
};
