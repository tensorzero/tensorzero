import type { Meta, ReactRenderer, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { CodeEditor, type CodeEditorProps } from "./code-editor";
import type { ArgsStoryFn } from "storybook/internal/csf";
import { GlobalToastProvider } from "~/providers/global-toast-provider";

const meta: Meta<typeof CodeEditor> = {
  title: "UI/CodeEditor",
  component: CodeEditor,
  parameters: {
    layout: "padded",
  },
  argTypes: {
    allowedLanguages: {
      control: { type: "check" },
      options: ["json", "markdown", "jinja2", "text"],
    },
  },
  decorators: [
    (Story) => (
      <GlobalToastProvider>
        <div className="max-w-md">
          <Story />
        </div>
      </GlobalToastProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof CodeEditor>;

// Sample content for different languages
const SAMPLE_JSON = `{
  "name": "CodeEditor",
  "version": "1.0.0",
  "features": [
    "syntax highlighting",
    "line numbers",
    "word wrap",
    "read-only mode"
  ],
  "config": {
    "theme": "github",
    "autoDetect": true
  }
}`;

const SAMPLE_MARKDOWN = `# CodeEditor Component

A **CodeMirror-based** code editor for React applications.

## Features

- Syntax highlighting for multiple languages
- Auto-detection of language based on content
- Customizable themes (light/dark)
- Read-only mode support

### Supported Languages

1. JSON
2. Markdown
3. Jinja2
4. Plain text

\`\`\`javascript
// Example usage
<CodeEditor
  value={code}
  onChange={setCode}
  language="json"
/>
\`\`\`

> **Note**: The editor supports both controlled and uncontrolled modes.
`;

const SAMPLE_JINJA2 = `<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
</head>
<body>
    <h1>Hello, {{ user.name }}!</h1>

    {% if user.is_admin %}
        <p>Welcome, admin!</p>
    {% else %}
        <p>Welcome, {{ user.role }}!</p>
    {% endif %}

    <ul>
    {% for item in items %}
        <li>{{ item.name }} - ${"{{ item.price }}"}</li>
    {% endfor %}
    </ul>

    {# This is a comment #}
    <footer>
        <p>&copy; {"{{ current_year }}"} My Company</p>
    </footer>
</body>
</html>`;

const InteractiveTemplate: ArgsStoryFn<ReactRenderer, CodeEditorProps> = (
  args,
) => {
  const [value, setValue] = useState(args.value || "");

  return (
    <div className="space-y-4">
      <CodeEditor {...args} value={value} onChange={setValue} />
      <div className="text-muted-foreground text-xs">
        Language detection:{" "}
        <strong>{args.autoDetectLanguage ? "enabled" : "disabled"}</strong> |
        Available languages:{" "}
        <strong>
          {args.allowedLanguages?.join(", ") || "text, markdown, json, jinja2"}
        </strong>
      </div>
    </div>
  );
};

export const Default: Story = {
  args: {
    value: SAMPLE_JSON,
    placeholder: "Enter your code here...",
    showLineNumbers: true,
    readOnly: false,
  },
  render: InteractiveTemplate,
};

export const JsonEditor: Story = {
  name: "JSON Editor",
  args: {
    value: SAMPLE_JSON,
    allowedLanguages: ["json"],
    autoDetectLanguage: false,
    showLineNumbers: true,
  },
  render: InteractiveTemplate,
};

export const MarkdownEditor: Story = {
  args: {
    value: SAMPLE_MARKDOWN,
    allowedLanguages: ["markdown"],
    autoDetectLanguage: false,
    showLineNumbers: true,
  },
  render: InteractiveTemplate,
};

export const Jinja2Editor: Story = {
  name: "Jinja2 Editor",
  args: {
    value: SAMPLE_JINJA2,
    allowedLanguages: ["jinja2"],
    autoDetectLanguage: false,
    showLineNumbers: true,
  },
  render: InteractiveTemplate,
};

export const ReadOnlyModeText: Story = {
  name: "Read-only mode (Text)",
  args: {
    value:
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    allowedLanguages: ["text"],
    autoDetectLanguage: false,
    readOnly: true,
    showLineNumbers: true,
  },
};

export const ReadOnlyMode: Story = {
  name: "Read-only mode",
  args: {
    value: SAMPLE_JSON,
    allowedLanguages: ["json"],
    autoDetectLanguage: false,
    readOnly: true,
    showLineNumbers: true,
  },
};

export const NoLineNumbers: Story = {
  name: "No line numbers",
  args: {
    value: SAMPLE_JSON,
    allowedLanguages: ["json"],
    autoDetectLanguage: false,
    showLineNumbers: false,
  },
  render: InteractiveTemplate,
};

export const AutoDetection: Story = {
  name: "Auto detection",
  args: {
    value: "",
    autoDetectLanguage: true,
    placeholder:
      "Paste some JSON, Markdown, or Jinja2 code to see auto-detection in action...",
  },
  render: InteractiveTemplate,
};

export const WrapLongLinesText: Story = {
  name: "Wide Input (Text)",
  args: {
    value:
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\nLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
  },
  render: InteractiveTemplate,
};

const jsonValue = JSON.stringify(
  {
    lorem:
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    nestedLorem: {
      lorem:
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
      lorem2:
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    },
    arrayLorem: [
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    ],
  },
  null,
  2,
);

export const WrapLongLinesJSON: Story = {
  name: "Wide Input (JSON)",
  args: {
    value: jsonValue,
  },
  render: InteractiveTemplate,
};

export const NoWrapLongLinesJSON: Story = {
  name: "Wide Input, no wrap (JSON)",
  play: async ({ canvas, userEvent }) => {
    const wordWrap = canvas.getByTitle("Toggle word wrap");
    await userEvent.click(wordWrap);
  },
  args: {
    value: jsonValue,
  },
  render: InteractiveTemplate,
};
