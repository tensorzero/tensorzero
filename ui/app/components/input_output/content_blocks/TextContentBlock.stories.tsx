import { useState } from "react";
import TextContentBlock from "./TextContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
  title: "Input Output/Content Blocks/TextContentBlock",
  component: TextContentBlock,
  decorators: [
    (Story) => (
      <div className="w-[80vw] bg-orange-100 p-8">
        <div className="bg-white p-4">
          <Story />
        </div>
      </div>
    ),
  ],
} satisfies Meta<typeof TextContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ShortText: Story = {
  name: "Short Text",
  args: {
    label: "Text",
    text: "Hello, world!",
    isEditing: false,
  },
};

export const ShortTextEditing: Story = {
  name: "Short Text (Editing)",
  args: {
    label: "Text",
    text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    isEditing: true,
    onChange: () => {},
  },
  render: function ShortTextEditingStory() {
    const [text, setText] = useState(
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    );
    return (
      <TextContentBlock
        label="Text"
        text={text}
        isEditing={true}
        onChange={setText}
      />
    );
  },
};

export const LongLine: Story = {
  name: "Long Line",
  args: {
    label: "Text",
    text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent quis orci turpis. Phasellus tempor metus sed enim congue consectetur. Donec commodo sollicitudin libero, quis mollis sapien pulvinar sit amet. Suspendisse potenti. Morbi vestibulum, justo id iaculis imperdiet, sem ipsum dignissim metus, ac viverra massa ex sit amet velit. Maecenas lobortis velit diam, nec finibus lacus blandit in. Morbi sed ullamcorper lectus, id maximus magna.",
    isEditing: false,
  },
};

export const LongText: Story = {
  name: "Long Text",
  args: {
    label: "Text",
    text: Array.from({ length: 100 }, (_, i) => `Line ${i + 1}`).join("\n"),
    isEditing: false,
  },
};

export const Json: Story = {
  name: "JSON",
  args: {
    label: "Text",
    text: '{"name": "John Doe", "age": 30, "city": "New York"}',
    isEditing: false,
  },
};

export const JsonEditing: Story = {
  name: "JSON (Editing)",
  args: {
    label: "Text",
    text: '{"name": "John Doe", "age": 30, "city": "New York"}',
    isEditing: true,
    onChange: () => {},
  },
  render: function JsonEditingStory() {
    const [text, setText] = useState(
      '{"name": "John Doe", "age": 30, "city": "New York"}',
    );
    return (
      <TextContentBlock
        label="Text"
        text={text}
        isEditing={true}
        onChange={setText}
      />
    );
  },
};

export const Empty: Story = {
  args: {
    label: "Text",
    text: "",
    isEditing: false,
  },
};

export const Markdown: Story = {
  args: {
    label: "Text",
    text: `# Heading 1

## Heading 2

This is a paragraph with **bold** and *italic* text.

- Bullet point 1
- Bullet point 2
- Bullet point 3

1. Numbered item 1
2. Numbered item 2
3. Numbered item 3

\`\`\`javascript
function example() {
  console.log("Hello, world!");
}
\`\`\`

> This is a blockquote

[Link text](https://example.com)`,
    isEditing: false,
  },
};
