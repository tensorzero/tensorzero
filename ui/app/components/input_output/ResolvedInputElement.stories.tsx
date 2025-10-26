import { type ReactNode, useState } from "react";
import ResolvedInputElement from "./ResolvedInputElement";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { ResolvedInput, Role } from "~/types/tensorzero";

function StoryWrapper({
  children,
  input,
}: {
  children: ReactNode;
  input: ResolvedInput;
}) {
  return (
    <div className="w-[80vw] bg-orange-100 p-8">
      <div className="bg-white p-4">{children}</div>
      <div className="mt-4 rounded border border-blue-300 bg-blue-50 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="font-semibold text-blue-900">
            Debug:{" "}
            <span className="text-md font-mono font-semibold">input</span>
          </h3>
        </div>
        <pre className="mt-2 overflow-auto rounded bg-white p-2 text-xs">
          {JSON.stringify(input, null, 2)}
        </pre>
      </div>
    </div>
  );
}

const meta = {
  title: "Input Output/ResolvedInputElement",
  component: ResolvedInputElement,
} satisfies Meta<typeof ResolvedInputElement>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    input: {
      messages: [],
    },
    isEditing: false,
  },
  render: function EmptyStory(args) {
    return (
      <StoryWrapper input={args.input}>
        <ResolvedInputElement {...args} />
      </StoryWrapper>
    );
  },
};

export const EmptyEditing: Story = {
  name: "Empty (Editing)",
  args: {
    input: {
      messages: [],
    },
    isEditing: true,
    onSystemChange: () => {},
    onMessagesChange: () => {},
  },
  render: function EmptyEditingStory() {
    const [input, setInput] = useState<ResolvedInput>({
      messages: [],
    });
    return (
      <StoryWrapper input={input}>
        <ResolvedInputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryWrapper>
    );
  },
};

export const SystemText: Story = {
  name: "System Text",
  args: {
    input: {
      system: "You are a helpful assistant that answers questions concisely.",
      messages: [],
    },
    isEditing: false,
  },
  render: function SystemTextStory(args) {
    return (
      <StoryWrapper input={args.input}>
        <ResolvedInputElement {...args} />
      </StoryWrapper>
    );
  },
};

export const SystemTextEditing: Story = {
  name: "System Text (Editing)",
  args: {
    input: {
      system: "You are a helpful assistant that answers questions concisely.",
      messages: [],
    },
    isEditing: true,
    onSystemChange: () => {},
    onMessagesChange: () => {},
  },
  render: function SystemTextEditingStory() {
    const [input, setInput] = useState<ResolvedInput>({
      system: "You are a helpful assistant that answers questions concisely.",
      messages: [],
    });
    return (
      <StoryWrapper input={input}>
        <ResolvedInputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryWrapper>
    );
  },
};

export const SystemTemplate: Story = {
  name: "System Template",
  args: {
    input: {
      system: {
        template_name: "customer_support_system",
        parameters: {
          company_name: "Acme Corp",
          tone: "professional",
          max_response_length: 200,
        },
      },
      messages: [],
    },
    isEditing: false,
  },
  render: function SystemTemplateStory(args) {
    return (
      <StoryWrapper input={args.input}>
        <ResolvedInputElement {...args} />
      </StoryWrapper>
    );
  },
};

export const SystemTemplateEditing: Story = {
  name: "System Template (Editing)",
  args: {
    input: {
      system: {
        template_name: "customer_support_system",
        parameters: {
          company_name: "Acme Corp",
          tone: "professional",
          max_response_length: 200,
        },
      },
      messages: [],
    },
    isEditing: true,
    onSystemChange: () => {},
    onMessagesChange: () => {},
  },
  render: function SystemTemplateEditingStory() {
    const [input, setInput] = useState<ResolvedInput>({
      system: {
        template_name: "customer_support_system",
        parameters: {
          company_name: "Acme Corp",
          tone: "professional",
          max_response_length: 200,
        },
      },
      messages: [],
    });
    return (
      <StoryWrapper input={input}>
        <ResolvedInputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryWrapper>
    );
  },
};

export const UserMessage: Story = {
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "What is the capital of France?" }],
        },
      ],
    },
    isEditing: false,
  },
  render: function UserMessageStory(args) {
    return (
      <StoryWrapper input={args.input}>
        <ResolvedInputElement {...args} />
      </StoryWrapper>
    );
  },
};

export const UserMessageEditing: Story = {
  name: "User Message (Editing)",
  args: {
    input: {
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "What is the capital of France?" }],
        },
      ],
    },
    isEditing: true,
    onSystemChange: () => {},
    onMessagesChange: () => {},
  },
  render: function UserMessageEditingStory() {
    const [input, setInput] = useState<ResolvedInput>({
      messages: [
        {
          role: "user" as Role,
          content: [{ type: "text", text: "What is the capital of France?" }],
        },
      ],
    });
    return (
      <StoryWrapper input={input}>
        <ResolvedInputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryWrapper>
    );
  },
};

export const AssistantMessage: Story = {
  args: {
    input: {
      messages: [
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "The capital of France is Paris.",
            },
          ],
        },
      ],
    },
    isEditing: false,
  },
  render: function AssistantMessageStory(args) {
    return (
      <StoryWrapper input={args.input}>
        <ResolvedInputElement {...args} />
      </StoryWrapper>
    );
  },
};

export const AssistantMessageEditing: Story = {
  name: "Assistant Message (Editing)",
  args: {
    input: {
      messages: [
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "The capital of France is Paris.",
            },
          ],
        },
      ],
    },
    isEditing: true,
    onSystemChange: () => {},
    onMessagesChange: () => {},
  },
  render: function AssistantMessageEditingStory() {
    const [input, setInput] = useState<ResolvedInput>({
      messages: [
        {
          role: "assistant" as Role,
          content: [
            {
              type: "text",
              text: "The capital of France is Paris.",
            },
          ],
        },
      ],
    });
    return (
      <StoryWrapper input={input}>
        <ResolvedInputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryWrapper>
    );
  },
};

const COMPLEX_INPUT: ResolvedInput = {
  system: "You are a helpful AI assistant with access to tools.",
  messages: [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: "Can you analyze this image and get me weather data for Paris?",
        },
        {
          type: "file",
          file: {
            data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            mime_type: "image/png",
            url: "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
          },
          storage_path: {
            kind: {
              type: "s3_compatible",
              bucket_name: "images",
              region: "us-east-1",
              endpoint: null,
              allow_http: null,
            },
            path: "user-uploads/abc123.png",
          },
        },
        {
          type: "template",
          name: "location_context",
          arguments: {
            city: "Paris",
            country: "France",
            timezone: "Europe/Paris",
          },
        },
      ],
    },
    {
      role: "assistant",
      content: [
        {
          type: "thought",
          text: "I need to first analyze the image, then fetch weather data using the weather tool.",
          signature: "abc123".repeat(1000),
          summary: [
            {
              type: "summary_text",
              text: "Planning to analyze image and fetch weather",
            },
          ],
        },
        {
          type: "text",
          text: "I can see the image shows a cityscape. Let me fetch the weather data for Paris.",
        },
        {
          type: "tool_call",
          name: "get_weather",
          arguments:
            '{"city": "Paris", "country": "France", "units": "metric"}',
          id: "call_abc123",
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool_result",
          name: "get_weather",
          result:
            '{"temperature": 18, "conditions": "partly cloudy", "humidity": 65}',
          id: "call_abc123",
        },
      ],
    },
    {
      role: "assistant",
      content: [
        {
          type: "raw_text",
          value: "Based on the weather data:",
        },
        {
          type: "text",
          text: "The current weather in Paris is 18°C with partly cloudy conditions and 65% humidity.",
        },
        {
          type: "unknown",
          data: {
            custom_field: "This is a custom content block",
            provider_specific: true,
          },
          model_provider_name: "custom_provider",
        },
      ],
    },
  ],
};

export const Complex: Story = {
  args: {
    input: COMPLEX_INPUT,
    isEditing: false,
  },
  render: function ComplexStory(args) {
    return (
      <StoryWrapper input={args.input}>
        <ResolvedInputElement {...args} />
      </StoryWrapper>
    );
  },
};

export const ComplexEditing: Story = {
  name: "Complex (Editing)",
  args: {
    input: COMPLEX_INPUT,
    isEditing: true,
    onSystemChange: () => {},
    onMessagesChange: () => {},
  },
  render: function ComplexEditingStory() {
    const [input, setInput] = useState<ResolvedInput>(COMPLEX_INPUT);
    return (
      <StoryWrapper input={input}>
        <ResolvedInputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryWrapper>
    );
  },
};
