import { useState } from "react";
import ThoughtContentBlock from "./ThoughtContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { Thought } from "~/types/tensorzero";
import { StoryDebugWrapper } from "~/components/.storybook/StoryDebugWrapper";

const meta = {
  title: "Input Output/Content Blocks/ThoughtContentBlock",
  component: ThoughtContentBlock,
} satisfies Meta<typeof ThoughtContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Text: Story = {
  args: {
    block: {
      text: "I need to analyze the user's request and determine the best approach.",
    },
    isEditing: false,
  },
  render: function TextStory(args) {
    return (
      <StoryDebugWrapper debugLabel="block" debugData={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const TextEditing: Story = {
  name: "Text (Editing)",
  args: {
    block: {
      text: "I need to analyze the user's request and determine the best approach.",
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function SimpleEditingStory() {
    const [block, setBlock] = useState<Thought>({
      text: "I need to analyze the user's request and determine the best approach.",
    });
    return (
      <StoryDebugWrapper debugLabel="block" debugData={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Signature: Story = {
  args: {
    block: {
      signature: "abc123".repeat(1000),
    },
    isEditing: false,
  },
  render: function SignatureStory(args) {
    return (
      <StoryDebugWrapper debugLabel="block" debugData={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const SignatureEditing: Story = {
  name: "Signature (Editing)",
  args: {
    block: {
      signature: "abc123".repeat(1000),
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function SignatureEditingStory() {
    const [block, setBlock] = useState<Thought>({
      signature: "abc123".repeat(1000),
    });
    return (
      <StoryDebugWrapper debugLabel="block" debugData={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Summary: Story = {
  args: {
    block: {
      summary: [
        {
          type: "summary_text",
          text: "Analysis and planning phase",
        },
      ],
    },
    isEditing: false,
  },
  render: function SummaryStory(args) {
    return (
      <StoryDebugWrapper debugLabel="block" debugData={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const SummaryEditing: Story = {
  name: "Summary (Editing)",
  args: {
    block: {
      summary: [
        {
          type: "summary_text",
          text: "Analysis and planning phase",
        },
      ],
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function SummaryEditingStory() {
    const [block, setBlock] = useState<Thought>({
      summary: [
        {
          type: "summary_text" as const,
          text: "Analysis and planning phase",
        },
      ],
    });
    return (
      <StoryDebugWrapper debugLabel="block" debugData={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Complete: Story = {
  name: "Complete",
  args: {
    block: {
      text: "I need to first analyze the image, then fetch weather data using the weather tool.",
      signature: "abc123".repeat(1000),
      summary: [
        {
          type: "summary_text",
          text: "Planning to analyze image and fetch weather",
        },
      ],
    },
    isEditing: false,
  },
  render: function CompleteStory(args) {
    return (
      <StoryDebugWrapper debugLabel="block" debugData={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const CompleteEditing: Story = {
  name: "Complete (Editing)",
  args: {
    block: {
      text: "I need to first analyze the image, then fetch weather data using the weather tool.",
      signature: "abc123".repeat(1000),
      summary: [
        {
          type: "summary_text",
          text: "Planning to analyze image and fetch weather",
        },
      ],
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function CompleteEditingStory() {
    const [block, setBlock] = useState<Thought>({
      text: "I need to first analyze the image, then fetch weather data using the weather tool.",
      signature: "abc123".repeat(1000),
      summary: [
        {
          type: "summary_text" as const,
          text: "Planning to analyze image and fetch weather",
        },
      ],
    });
    return (
      <StoryDebugWrapper debugLabel="block" debugData={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryDebugWrapper>
    );
  },
};

export const MultipleSummaryBlocks: Story = {
  args: {
    block: {
      text: "This is a complex multi-step reasoning process.",
      signature: "abc123".repeat(1000),
      summary: [
        {
          type: "summary_text",
          text: "Step 1: Analyze the user's input and extract key requirements",
        },
        {
          type: "summary_text",
          text: "Step 2: Identify the appropriate tools and data sources needed",
        },
        {
          type: "summary_text",
          text: "Step 3: Plan the execution sequence for optimal results",
        },
      ],
    },
    isEditing: false,
  },
  render: function MultipleSummaryBlocksStory(args) {
    return (
      <StoryDebugWrapper debugLabel="block" debugData={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const MultipleSummaryBlocksEditing: Story = {
  name: "Multiple Summary Blocks (Editing)",
  args: {
    block: {
      text: "This is a complex multi-step reasoning process.",
      signature: "abc123".repeat(1000),
      summary: [
        {
          type: "summary_text",
          text: "Step 1: Analyze the user's input and extract key requirements",
        },
        {
          type: "summary_text",
          text: "Step 2: Identify the appropriate tools and data sources needed",
        },
        {
          type: "summary_text",
          text: "Step 3: Plan the execution sequence for optimal results",
        },
      ],
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function MultipleSummaryBlocksEditingStory() {
    const [block, setBlock] = useState<Thought>({
      text: "This is a complex multi-step reasoning process.",
      signature: "abc123".repeat(1000),
      summary: [
        {
          type: "summary_text" as const,
          text: "Step 1: Analyze the user's input and extract key requirements",
        },
        {
          type: "summary_text" as const,
          text: "Step 2: Identify the appropriate tools and data sources needed",
        },
        {
          type: "summary_text" as const,
          text: "Step 3: Plan the execution sequence for optimal results",
        },
      ],
    });
    return (
      <StoryDebugWrapper debugLabel="block" debugData={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Empty: Story = {
  args: {
    block: {},
    isEditing: false,
  },
  render: function EmptyStory(args) {
    return (
      <StoryDebugWrapper debugLabel="block" debugData={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const EmptyEditing: Story = {
  name: "Empty (Editing)",
  args: {
    block: {},
    isEditing: true,
    onChange: () => {},
  },
  render: function EmptyEditingStory() {
    const [block, setBlock] = useState<Thought>({});
    return (
      <StoryDebugWrapper debugLabel="block" debugData={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Long: Story = {
  name: "Long",
  args: {
    block: {
      text: (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ".repeat(
          10,
        ) + "\n"
      ).repeat(10),
      signature: "abc123".repeat(1000),
      summary: Array.from({ length: 10 }, () => ({
        type: "summary_text",
        text: (
          "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ".repeat(
            10,
          ) + "\n"
        ).repeat(3),
      })),
    },
    isEditing: false,
  },
  render: function LongStory(args) {
    return (
      <StoryDebugWrapper debugLabel="block" debugData={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryDebugWrapper>
    );
  },
};
