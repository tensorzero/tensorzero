import type { Meta, StoryObj } from "@storybook/react-vite";
import { TextMessage } from "./SnippetContent";

const meta = {
  title: "UI/Message Blocks/TextMessage",
  component: TextMessage,
  parameters: {
    layout: "padded",
  },
  argTypes: {
    label: {
      control: "text",
      description: "Optional label for the message",
    },
    content: {
      control: "text",
      description: "The text content to display",
    },
    emptyMessage: {
      control: "text",
      description: "Message to show when content is empty",
    },
  },
} satisfies Meta<typeof TextMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  args: {
    content:
      "This is a simple text message that demonstrates how content is displayed.",
  },
};

export const WithLabel: Story = {
  args: {
    label: "User Input",
    content:
      "This text message includes a label to provide context about the content.",
  },
};

export const LongContent: Story = {
  args: {
    label: "Long Content",
    content:
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
  },
};

export const Empty: Story = {
  args: {
    label: "Empty Content",
    content: "",
    emptyMessage: "No text content provided",
  },
};

export const JSONContent: Story = {
  args: {
    label: "JSON Content",
    content: `{
  "topic": "Machine Learning",
  "style": "academic",
  "length": "detailed",
  "include_examples": true
}`,
  },
};

export const MultilineContent: Story = {
  args: {
    label: "Instructions",
    content: `Step 1: Initialize the project
Step 2: Install dependencies
Step 3: Configure the environment
Step 4: Run the development server
Step 5: Test the application`,
  },
};

export const MarkdownContent: Story = {
  args: {
    label: "Documentation",
    content: `# Project Overview

This is a **markdown** example with various formatting options.

## Features
- Bullet point one
- Bullet point two
- Bullet point three

### Code Example
\`\`\`javascript
function hello() {
  console.log("Hello, world!");
}
\`\`\`

### Links and Formatting
Visit [TensorZero](https://tensorzero.com) for more information.

*Italic text* and **bold text** are supported.

> This is a blockquote with important information.

1. Ordered list item
2. Another item
3. Final item`,
  },
};
