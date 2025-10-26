import { type ReactNode, useState } from "react";
import ThoughtContentBlock from "./ThoughtContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { Thought } from "~/types/tensorzero";

function StoryWrapper({
  children,
  block,
}: {
  children: ReactNode;
  block: Thought;
}) {
  return (
    <div className="w-[80vw] bg-orange-100 p-8">
      <div className="bg-white p-4">{children}</div>
      <div className="mt-4 rounded border border-blue-300 bg-blue-50 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="font-semibold text-blue-900">
            Debug:{" "}
            <span className="text-md font-mono font-semibold">block</span>
          </h3>
        </div>
        <pre className="mt-2 overflow-auto rounded bg-white p-2 text-xs">
          {JSON.stringify(block, null, 2)}
        </pre>
      </div>
    </div>
  );
}

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
      <StoryWrapper block={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryWrapper>
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
      <StoryWrapper block={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryWrapper>
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
      <StoryWrapper block={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryWrapper>
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
      <StoryWrapper block={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryWrapper>
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
      <StoryWrapper block={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryWrapper>
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
      <StoryWrapper block={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryWrapper>
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
      <StoryWrapper block={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryWrapper>
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
      <StoryWrapper block={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryWrapper>
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
      <StoryWrapper block={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryWrapper>
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
      <StoryWrapper block={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryWrapper>
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
      <StoryWrapper block={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryWrapper>
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
      <StoryWrapper block={block}>
        <ThoughtContentBlock
          block={block}
          isEditing={true}
          onChange={setBlock}
        />
      </StoryWrapper>
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
      <StoryWrapper block={args.block}>
        <ThoughtContentBlock {...args} />
      </StoryWrapper>
    );
  },
};
