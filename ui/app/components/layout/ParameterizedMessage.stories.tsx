import type { Meta, StoryObj } from "@storybook/react-vite";
import { ParameterizedMessage } from "./SnippetContent";

const meta = {
  title: "UI/Message Blocks/ParameterizedMessage",
  component: ParameterizedMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof ParameterizedMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  args: {
    parameters: {
      topic: "AI",
      style: "conversational",
    },
  },
};

export const NestedObject: Story = {
  args: {
    parameters: {
      user: {
        name: "John Doe",
        preferences: {
          language: "English",
          theme: "dark",
        },
      },
      settings: {
        notifications: true,
        frequency: "daily",
      },
    },
  },
};

export const WithArrays: Story = {
  args: {
    parameters: {
      topics: ["Machine Learning", "Neural Networks", "Deep Learning"],
      config: {
        models: ["GPT-4", "Claude", "Gemini"],
        maxTokens: 1000,
      },
    },
  },
};

export const ComplexStructure: Story = {
  args: {
    parameters: {
      query: "SELECT * FROM users WHERE active = true",
      filters: {
        department: "engineering",
        roles: ["senior", "lead", "principal"],
        dateRange: {
          start: "2024-01-01",
          end: "2024-12-31",
        },
      },
      options: {
        includeMetadata: true,
        format: "json",
        compression: null,
      },
    },
  },
};

export const WithNullAndUndefined: Story = {
  args: {
    parameters: {
      requiredField: "value",
      optionalField: null,
      missingField: undefined,
      booleanField: false,
      numberField: 0,
    },
  },
};
