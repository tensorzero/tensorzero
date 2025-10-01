import ModelInferenceOutput from "./ModelInferenceOutput";
import type { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
  title: "ModelInferenceOutput",
  component: ModelInferenceOutput,
  render: (args) => (
    <div className="w-[80vw] p-4">
      <ModelInferenceOutput {...args} />
    </div>
  ),
} satisfies Meta<typeof ModelInferenceOutput>;

export default meta;
type Story = StoryObj<typeof meta>;

export const TextOutput: Story = {
  args: {
    output: [{ type: "text", text: "Hello, world!" }],
  },
};

export const MultipleTextBlocks: Story = {
  args: {
    output: [
      { type: "text", text: "Hello, world!" },
      { type: "text", text: "Hello, world!" },
      { type: "text", text: "Hello, world!" },
    ],
  },
};

const shortToolCallArgumentsFixture = JSON.stringify({
  location: "Paris",
  unit: "celsius",
});

const shortToolCallArgumentsFixture2 = JSON.stringify({
  location: "Paris",
});

export const ToolCallOutput: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tc-1234567890",
        name: "get_temperature",
        arguments: shortToolCallArgumentsFixture,
      },
    ],
  },
};

export const ParallelToolCallsAndText: Story = {
  args: {
    output: [
      {
        type: "text",
        text: "Some text before the tool calls",
      },
      {
        type: "tool_call",
        id: "tc-1234567890",
        name: "get_temperature",
        arguments: shortToolCallArgumentsFixture,
      },
      {
        type: "tool_call",
        id: "tc-1234567890",
        name: "get_humidity",
        arguments: shortToolCallArgumentsFixture2,
      },
      {
        type: "text",
        text: "Some text after the tool calls",
      },
    ],
  },
};

const massiveTextOutputFixture =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent quis orci turpis. Phasellus tempor metus sed enim congue consectetur. Donec commodo sollicitudin libero, quis mollis sapien pulvinar sit amet. Suspendisse potenti. Morbi vestibulum, justo id iaculis imperdiet, sem ipsum dignissim metus, ac viverra massa ex sit amet velit. Maecenas lobortis velit diam, nec finibus lacus blandit in. Morbi sed ullamcorper lectus, id maximus magna. ".repeat(
    100,
  );

export const MassiveTextOutput: Story = {
  args: {
    output: [{ type: "text", text: massiveTextOutputFixture }],
  },
};

const massiveToolCallOutputFixture = JSON.stringify({
  text: massiveTextOutputFixture,
});

export const MassiveToolCallOutput: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tc-1234567890",
        name: "summarize_text",
        arguments: massiveToolCallOutputFixture,
      },
    ],
  },
};

export const EmptyOutput: Story = {
  args: {
    output: [],
  },
};
