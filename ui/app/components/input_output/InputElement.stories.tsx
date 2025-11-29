import { useState } from "react";
import { InputElement } from "./InputElement";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { Input, Role } from "~/types/tensorzero";
import { StoryDebugWrapper } from "~/components/.storybook/StoryDebugWrapper";
import { getBase64File } from "./content_blocks/FileContentBlock.stories";
import { GlobalToastProvider } from "~/providers/global-toast-provider";

const meta = {
  title: "Input Output/InputElement",
  component: InputElement,
  decorators: [
    (Story) => (
      <GlobalToastProvider>
        <Story />
      </GlobalToastProvider>
    ),
  ],
} satisfies Meta<typeof InputElement>;

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
      <StoryDebugWrapper debugLabel="input" debugData={args.input}>
        <InputElement {...args} />
      </StoryDebugWrapper>
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
    const [input, setInput] = useState<Input>({
      messages: [],
    });
    return (
      <StoryDebugWrapper debugLabel="input" debugData={input}>
        <InputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryDebugWrapper>
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
      <StoryDebugWrapper debugLabel="input" debugData={args.input}>
        <InputElement {...args} />
      </StoryDebugWrapper>
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
    const [input, setInput] = useState<Input>({
      system: "You are a helpful assistant that answers questions concisely.",
      messages: [],
    });
    return (
      <StoryDebugWrapper debugLabel="input" debugData={input}>
        <InputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryDebugWrapper>
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
      <StoryDebugWrapper debugLabel="input" debugData={args.input}>
        <InputElement {...args} />
      </StoryDebugWrapper>
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
    const [input, setInput] = useState<Input>({
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
      <StoryDebugWrapper debugLabel="input" debugData={input}>
        <InputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryDebugWrapper>
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
      <StoryDebugWrapper debugLabel="input" debugData={args.input}>
        <InputElement {...args} />
      </StoryDebugWrapper>
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
    const [input, setInput] = useState<Input>({
      messages: [
        {
          role: "user" as Role,
          content: [{ type: "text", text: "What is the capital of France?" }],
        },
      ],
    });
    return (
      <StoryDebugWrapper debugLabel="input" debugData={input}>
        <InputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryDebugWrapper>
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
      <StoryDebugWrapper debugLabel="input" debugData={args.input}>
        <InputElement {...args} />
      </StoryDebugWrapper>
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
    const [input, setInput] = useState<Input>({
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
      <StoryDebugWrapper debugLabel="input" debugData={input}>
        <InputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryDebugWrapper>
    );
  },
};

const COMPLEX_INPUT: Input = {
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
          ...(await getBase64File({
            source_url:
              "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
            mime_type: "image/png",
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
          })),
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
          arguments: { city: "Paris", country: "France", units: "metric" },
          raw_name: "get_weather",
          raw_arguments:
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
      <StoryDebugWrapper debugLabel="input" debugData={args.input}>
        <InputElement {...args} />
      </StoryDebugWrapper>
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
    const [input, setInput] = useState<Input>(COMPLEX_INPUT);
    return (
      <StoryDebugWrapper debugLabel="input" debugData={input}>
        <InputElement
          input={input}
          isEditing={true}
          onSystemChange={(system) => setInput({ ...input, system })}
          onMessagesChange={(messages) => setInput({ ...input, messages })}
        />
      </StoryDebugWrapper>
    );
  },
};
