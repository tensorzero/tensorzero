import type { Meta, StoryObj } from "@storybook/react";
import { VariantEditor } from "./VariantEditor";
import { useState } from "react";
import type { VariantInfo } from "tensorzero-node";
import { Button } from "~/components/ui/button";
import { safeStringify } from "~/utils/serialization";

const meta = {
  title: "Components/Function/Variant/VariantEditor",
  component: VariantEditor,
  parameters: {
    layout: "fullscreen",
  },
} satisfies Meta<typeof VariantEditor>;

export default meta;
type Story = StoryObj<typeof meta>;

const VariantEditorWrapper = ({
  initialInfo,
}: {
  initialInfo: VariantInfo;
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [info, setInfo] = useState<VariantInfo>(initialInfo);

  return (
    <div className="p-8">
      <div className="space-y-4">
        <h2 className="text-2xl font-bold">Variant Editor Demo</h2>
        <p className="text-muted-foreground">
          Click the button below to open the variant editor
        </p>
        <Button onClick={() => setIsOpen(true)}>Open Variant Editor</Button>

        <div className="mt-8">
          <h3 className="mb-2 text-lg font-semibold">Current Configuration:</h3>
          <pre className="bg-muted overflow-auto rounded-md p-4 text-sm">
            {safeStringify(info)}
          </pre>
        </div>
      </div>

      <VariantEditor
        variantInfo={info}
        confirmVariantInfo={(newInfo) => {
          setInfo(newInfo);
          console.log("Updated info:", newInfo);
        }}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        variantName="example-variant"
      />
    </div>
  );
};

export const FullyPopulated: Story = {
  args: {
    variantInfo: {} as VariantInfo,
    confirmVariantInfo: () => {},
    isOpen: false,
    onClose: () => {},
    variantName: "fully-populated",
  },
  render: () => (
    <VariantEditorWrapper
      initialInfo={{
        inner: {
          type: "chat_completion",
          weight: 1.0,
          model: "gpt-4-turbo-preview",
          templates: {
            system: {
              template: {
                path: "templates/system.jinja2",
                contents:
                  "You are a helpful AI assistant. Today's date is {{ date }}.",
              },
              schema: null,
              legacy_definition: false,
            },
            user: {
              template: {
                path: "templates/user.jinja2",
                contents: "User query: {{ query }}\\n\\nContext: {{ context }}",
              },
              schema: null,
              legacy_definition: false,
            },
            assistant: {
              template: {
                path: "templates/assistant.jinja2",
                contents:
                  "Based on the context, here's my response: {{ response }}",
              },
              schema: null,
              legacy_definition: false,
            },
          },
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 2048,
          presence_penalty: 0.1,
          frequency_penalty: 0.2,
          seed: 42,
          stop_sequences: ["END", "STOP", "\\\\n\\\\n"],
          json_mode: "strict",
          retries: {
            num_retries: 3,
            max_delay_s: 60,
          },
        },
        timeouts: {
          non_streaming: {
            total_ms: 30000n,
          },
          streaming: {
            ttft_ms: 5000n,
          },
        },
      }}
    />
  ),
};

export const MinimalConfig: Story = {
  args: {
    variantInfo: {} as VariantInfo,
    confirmVariantInfo: () => {},
    isOpen: false,
    onClose: () => {},
    variantName: "minimal-config",
  },
  render: () => (
    <VariantEditorWrapper
      initialInfo={{
        inner: {
          type: "chat_completion",
          weight: null,
          model: "gpt-4.1-mini",
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
        timeouts: {
          non_streaming: {
            total_ms: null,
          },
          streaming: {
            ttft_ms: null,
          },
        },
      }}
    />
  ),
};

export const WithJsonMode: Story = {
  args: {
    variantInfo: {} as VariantInfo,
    confirmVariantInfo: () => {},
    isOpen: false,
    onClose: () => {},
    variantName: "json-mode-variant",
  },
  render: () => (
    <VariantEditorWrapper
      initialInfo={{
        inner: {
          type: "chat_completion",
          weight: 0.8,
          model: "claude-3-opus-20240229",
          templates: {
            system: {
              template: {
                path: "templates/json_system.jinja2",
                contents:
                  "You are a JSON API that always responds with valid JSON.",
              },
              schema: null,
              legacy_definition: false,
            },
            user: {
              template: {
                path: "templates/json_user.jinja2",
                contents:
                  "Extract the following information from the text: {{ text }}",
              },
              schema: null,
              legacy_definition: false,
            },
          },
          temperature: 0.3,
          top_p: 0.95,
          max_tokens: 1024,
          presence_penalty: null,
          frequency_penalty: null,
          seed: 12345,
          stop_sequences: ["}"],
          json_mode: "on",
          retries: {
            num_retries: 2,
            max_delay_s: 30,
          },
        },
        timeouts: {
          non_streaming: {
            total_ms: 15000n,
          },
          streaming: {
            ttft_ms: 3000n,
          },
        },
      }}
    />
  ),
};

export const WithTemplatesOnly: Story = {
  args: {
    variantInfo: {} as VariantInfo,
    confirmVariantInfo: () => {},
    isOpen: false,
    onClose: () => {},
    variantName: "templates-only",
  },
  render: () => (
    <VariantEditorWrapper
      initialInfo={{
        inner: {
          type: "chat_completion",
          weight: null,
          model: "mixtral-8x7b",
          templates: {
            system: {
              template: {
                path: "templates/creative_system.jinja2",
                contents: `You are a creative writing assistant specialized in {{ genre }} stories.
    Your tone should be {{ tone }} and engaging.`,
              },
              schema: null,
              legacy_definition: false,
            },
            user: {
              template: {
                path: "templates/creative_user.jinja2",
                contents: `Write a story about: {{ prompt }}

    Requirements:
    - Length: {{ word_count }} words
    - Include these elements: {{ elements | join(", ") }}`,
              },
              schema: null,
              legacy_definition: false,
            },
            assistant: {
              template: {
                path: "templates/creative_assistant.jinja2",
                contents: `Title: {{ title }}

    {{ story }}

    ---
    The End`,
              },
              schema: null,
              legacy_definition: false,
            },
          },
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
        timeouts: {
          non_streaming: {
            total_ms: 60000n,
          },
          streaming: {
            ttft_ms: 10000n,
          },
        },
      }}
    />
  ),
};

export const WithCustomNamedTemplates: Story = {
  args: {
    variantInfo: {} as VariantInfo,
    confirmVariantInfo: () => {},
    isOpen: false,
    onClose: () => {},
    variantName: "custom-named-templates",
  },
  render: () => (
    <VariantEditorWrapper
      initialInfo={{
        inner: {
          type: "chat_completion",
          weight: null,
          model: "gpt-5-mini",
          templates: {
            fun_fact_topic: {
              template: {
                path: "templates/fun_fact_topic.jinja2",
                contents: "Share a fun fact about: {{ topic }}",
              },
              schema: null,
              legacy_definition: false,
            },
            greeting_template: {
              template: {
                path: "templates/greeting.jinja2",
                contents:
                  "Hello {{ name }}! Welcome to {{ place }}. Today is {{ day_of_week }}.",
              },
              schema: null,
              legacy_definition: false,
            },
            analysis_prompt: {
              template: {
                path: "templates/analysis.jinja2",
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
          },
          temperature: 0.8,
          top_p: null,
          max_tokens: 512,
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
        timeouts: {
          non_streaming: {
            total_ms: 30000n,
          },
          streaming: {
            ttft_ms: 5000n,
          },
        },
      }}
    />
  ),
};

export const UnsupportedVariantType: Story = {
  args: {
    variantInfo: {} as VariantInfo,
    confirmVariantInfo: () => {},
    isOpen: false,
    onClose: () => {},
    variantName: "unsupported-type",
  },
  render: () => (
    <VariantEditorWrapper
      initialInfo={{
        inner: {
          type: "best_of_n_sampling",
          candidates: [
            {
              type: "chat_completion",
              weight: 1.0,
              model: "gpt-4",
              system_template: null,
              user_template: null,
              assistant_template: null,
              temperature: 0.7,
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
          ],
          evaluator: {
            type: "llm",
            model: "gpt-4",
            system_template: {
              path: "evaluator.jinja2",
              contents: "Evaluate the response quality.",
            },
            user_template: {
              path: "evaluator_user.jinja2",
              contents: "Rate this response: {{ response }}",
            },
            assistant_template: null,
            temperature: 0.0,
            top_p: null,
            max_tokens: 10,
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
        } as unknown as VariantInfo["inner"],
        timeouts: {
          non_streaming: {
            total_ms: null,
          },
          streaming: {
            ttft_ms: null,
          },
        },
      }}
    />
  ),
};
