import type { Meta, StoryObj } from "@storybook/react-vite";
import { Markdown } from "./markdown";

const meta = {
  title: "UI/Markdown",
  component: Markdown,
  parameters: {
    layout: "padded",
  },
  render: (args) => (
    <div className="max-w-2xl">
      <Markdown {...args} />
    </div>
  ),
} satisfies Meta<typeof Markdown>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  args: {
    children: "This is a simple paragraph with **bold** and *italic* text.",
  },
};

export const Headings: Story = {
  args: {
    children: `# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6`,
  },
};

export const Lists: Story = {
  args: {
    children: `## Unordered List
- Item one
- Item two
- Item three

## Ordered List
1. First item
2. Second item
3. Third item`,
  },
};

export const CodeBlocks: Story = {
  args: {
    children: `Here is some inline code: \`console.log("hello")\`

And here is a code block:

\`\`\`javascript
function greet(name) {
  return \`Hello, \${name}!\`;
}

greet("World");
\`\`\`

And another in Python:

\`\`\`python
def greet(name):
    return f"Hello, {name}!"

greet("World")
\`\`\``,
  },
};

export const Blockquotes: Story = {
  args: {
    children: `> This is a blockquote with some important information.
> It can span multiple lines.

Regular text after the blockquote.`,
  },
};

export const Links: Story = {
  args: {
    children: `Check out [TensorZero](https://tensorzero.com) for more information.

You can also visit [GitHub](https://github.com) or [Google](https://google.com).`,
  },
};

export const Tables: Story = {
  args: {
    children: `| Feature | Status | Notes |
|---------|--------|-------|
| Markdown | Done | Basic support |
| Tables | Done | With headers |
| Code | Done | Inline and blocks |`,
  },
};

export const FullDocument: Story = {
  args: {
    children: `# Project Overview

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
3. Final item

---

## Summary

| Feature | Supported |
|---------|-----------|
| Headings | Yes |
| Lists | Yes |
| Code | Yes |
| Tables | Yes |`,
  },
};
