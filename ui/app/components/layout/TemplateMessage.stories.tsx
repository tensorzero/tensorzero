import type { Meta, StoryObj } from "@storybook/react-vite";
import { TemplateMessage } from "./SnippetContent";

const meta = {
  title: "UI/Message Blocks/TemplateMessage",
  component: TemplateMessage,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof TemplateMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  args: {
    templateName: "greeting",
    arguments: {
      name: "Alice",
      language: "English",
    },
  },
};

export const EmptyArguments: Story = {
  args: {
    templateName: "empty_template",
    arguments: {},
  },
};

export const NestedObject: Story = {
  args: {
    templateName: "user_profile",
    arguments: {
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
    templateName: "content_generator",
    arguments: {
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
    templateName: "query_builder",
    arguments: {
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
    templateName: "conditional_template",
    arguments: {
      requiredField: "value",
      optionalField: null,
      missingField: undefined,
      booleanField: false,
      numberField: 0,
    },
  },
};

export const LongTemplateName: Story = {
  args: {
    templateName:
      "very_long_template_name_that_describes_what_the_template_does",
    arguments: {
      key: "value",
    },
  },
};
