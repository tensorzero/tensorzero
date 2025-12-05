import { useState } from "react";
import { ChatOutputElement } from "./ChatOutputElement";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { ContentBlockChatOutput } from "~/types/tensorzero";
import { StoryDebugWrapper } from "~/components/.storybook/StoryDebugWrapper";
import { GlobalToastProvider } from "~/providers/global-toast-provider";

const meta = {
  title: "Input Output/ChatOutputElement",
  component: ChatOutputElement,
  decorators: [
    (Story) => (
      <GlobalToastProvider>
        <Story />
      </GlobalToastProvider>
    ),
  ],
} satisfies Meta<typeof ChatOutputElement>;

export default meta;
type Story = StoryObj<typeof meta>;

export const None: Story = {
  args: {
    output: undefined,
    isEditing: false,
  },
  render: function EmptyStory(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <ChatOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const NoneEditing: Story = {
  name: "None (Editing)",
  args: {
    output: undefined,
    isEditing: true,
  },
  render: function NoneEditingStory(args) {
    const [output, setOutput] = useState<ContentBlockChatOutput[] | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="input" debugData={output}>
        <ChatOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Empty: Story = {
  args: {
    output: [],
    isEditing: false,
  },
  render: function EmptyStory(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <ChatOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const EmptyEditing: Story = {
  name: "Empty (Editing)",
  args: {
    output: [],
    isEditing: true,
  },
  render: function EmptyEditingStory(args) {
    const [output, setOutput] = useState<ContentBlockChatOutput[] | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="input" debugData={output}>
        <ChatOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Text: Story = {
  args: {
    output: [
      {
        type: "text",
        text: "This is an output content block with text.\n\nMore text here.\n\n\n\nMegumin cast explosion!",
      },
    ],
    isEditing: false,
  },
  render: function EmptyStory(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <ChatOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const TextEditing: Story = {
  name: "Text (Editing)",
  args: {
    output: [
      {
        type: "text",
        text: "This is an output content block with text.\n\nMore text here.\n\n\n\nMegumin cast explosion!",
      },
    ],
    isEditing: true,
  },
  render: function TextEditingStory(args) {
    const [output, setOutput] = useState<ContentBlockChatOutput[] | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="input" debugData={output}>
        <ChatOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};

export const BadTool: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tool_call_1234567890",
        name: null,
        raw_name: "get_temperature",
        arguments: null,
        raw_arguments: '{"location": 123',
      },
    ],
    isEditing: false,
  },
  render: function EmptyStory(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <ChatOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const BadToolEditing: Story = {
  name: "Bad Tool (Editing)",
  args: {
    output: [
      {
        type: "tool_call",
        id: "tool_call_1234567890",
        name: null,
        raw_name: "get_temperature",
        arguments: null,
        raw_arguments: '{"location": 123',
      },
    ],
    isEditing: true,
  },
  render: function ComplexEditingStory(args) {
    const [output, setOutput] = useState<ContentBlockChatOutput[] | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="input" debugData={output}>
        <ChatOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};

const COMPLEX_OUTPUT: ContentBlockChatOutput[] = [
  {
    type: "thought",
    text: "Let me think carefully..." + "\n\nHmm...".repeat(10),
  },
  {
    type: "text",
    text: "Let's call a tool!",
  },
  {
    type: "tool_call",
    id: "tool_call_1234567890",
    name: "get_temperature",
    raw_name: "get_temperature",
    arguments: {
      location: "New York",
      unit: "celsius",
    },
    raw_arguments: JSON.stringify({
      location: "New York",
      unit: "celsius",
    }),
  },
  {
    type: "unknown",
    data: "{}",
  },
];

export const Complex: Story = {
  args: {
    output: COMPLEX_OUTPUT,
    isEditing: false,
  },
  render: function EmptyStory(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <ChatOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const ComplexEditing: Story = {
  name: "Complex (Editing)",
  args: {
    output: COMPLEX_OUTPUT,
    isEditing: true,
  },
  render: function ComplexEditingStory(args) {
    const [output, setOutput] = useState<ContentBlockChatOutput[] | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="input" debugData={output}>
        <ChatOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};
