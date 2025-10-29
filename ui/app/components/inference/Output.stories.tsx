import { Output } from "./Output";
import type { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
  title: "Output",
  component: Output,
  render: (args) => (
    <div className="w-[80vw] p-4">
      <Output {...args} />
    </div>
  ),
} satisfies Meta<typeof Output>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ChatFunction: Story = {
  args: {
    output: [{ type: "text", text: "Hello, world!\n\nHow are you?" }],
  },
};

export const ChatFunctionMultipleBlocks: Story = {
  args: {
    output: [
      { type: "text", text: "Hello, world!" },
      { type: "text", text: "Hello, world!" },
      { type: "text", text: "Hello, world!" },
    ],
  },
};

const shortToolCallArgumentsFixture = {
  location: "Paris",
  unit: "celsius",
};

const shortToolCallArgumentsFixture2 = {
  location: "Paris",
};

export const ChatFunctionWithToolCall: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tc-1234567890",
        raw_name: "get_temperature",
        raw_arguments: JSON.stringify(shortToolCallArgumentsFixture),
        name: "get_temperature",
        arguments: shortToolCallArgumentsFixture,
      },
    ],
  },
};

export const ChatFunctionWithParallelToolCallsAndText: Story = {
  args: {
    output: [
      {
        type: "text",
        text: "Some text before the tool calls",
      },
      {
        type: "tool_call",
        id: "tc-1234567890",
        raw_name: "get_temperature",
        raw_arguments: JSON.stringify(shortToolCallArgumentsFixture),
        name: "get_temperature",
        arguments: shortToolCallArgumentsFixture,
      },
      {
        type: "tool_call",
        id: "tc-1234567890",
        raw_name: "get_humidity",
        raw_arguments: JSON.stringify(shortToolCallArgumentsFixture2),
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

// TODO: we must fallback to raw_name if name is null (make it clear to user the name is problematic)
// TODO: we must show the raw_arguments if arguments is null (make it clear to user the arguments are problematic)
export const ChatFunctionWithBadToolCallName: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tc-1234567890",
        raw_name: "get_temperatu",
        raw_arguments: JSON.stringify(shortToolCallArgumentsFixture),
        name: null,
        arguments: shortToolCallArgumentsFixture,
      },
    ],
  },
};

export const ChatFunctionWithBadToolCallArguments: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tc-1234567890",
        raw_name: "get_temperature",
        raw_arguments: JSON.stringify(shortToolCallArgumentsFixture).slice(
          0,
          -10,
        ),
        name: "get_temperature",
        arguments: null,
      },
    ],
  },
};

export const ChatFunctionWithBadToolCallBoth: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tc-1234567890",
        raw_name: "get_temperatu",
        raw_arguments: JSON.stringify(shortToolCallArgumentsFixture).slice(
          0,
          -10,
        ),
        name: null,
        arguments: null,
      },
    ],
  },
};
const massiveTextOutputFixture =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent quis orci turpis. Phasellus tempor metus sed enim congue consectetur. Donec commodo sollicitudin libero, quis mollis sapien pulvinar sit amet. Suspendisse potenti. Morbi vestibulum, justo id iaculis imperdiet, sem ipsum dignissim metus, ac viverra massa ex sit amet velit. Maecenas lobortis velit diam, nec finibus lacus blandit in. Morbi sed ullamcorper lectus, id maximus magna. ".repeat(
    100,
  );

export const ChatFunctionWithMassiveText: Story = {
  args: {
    output: [{ type: "text", text: massiveTextOutputFixture }],
  },
};

const massiveInferenceResponseToolCallFixture = {
  text: massiveTextOutputFixture,
};

export const ChatFunctionWithMassiveToolCall: Story = {
  args: {
    output: [
      {
        type: "tool_call",
        id: "tc-1234567890",
        raw_name: "summarize_text",
        raw_arguments: JSON.stringify(massiveInferenceResponseToolCallFixture),
        name: "summarize_text",
        arguments: massiveInferenceResponseToolCallFixture,
      },
    ],
  },
};

// TODO: we must show a message to the user if the output is empty
export const ChatFunctionEmpty: Story = {
  args: {
    output: [],
  },
};

const shortJSONOutputFixture = {
  answer: "Paris",
  source: "https://en.wikipedia.org/wiki/Paris",
};

const shortJSONOutputSchemaFixture = {
  $schema: "http://json-schema.org/draft-07/schema#",
  type: "object",
  properties: {
    answer: {
      type: "string",
    },
    source: {
      type: "string",
    },
  },
  required: ["answer", "source"],
  additionalProperties: false,
};

export const JSONFunction: Story = {
  args: {
    output: {
      raw: JSON.stringify(shortJSONOutputFixture),
      parsed: shortJSONOutputFixture,
      schema: shortJSONOutputSchemaFixture,
    },
  },
};

export const JSONFunctionNotParsed: Story = {
  args: {
    output: {
      raw: JSON.stringify(shortJSONOutputFixture).slice(0, -10), // imagine this JSON got truncated somehow
      parsed: null,
      schema: shortJSONOutputSchemaFixture,
    },
  },
};

export const JSONFunctionWithoutSchema: Story = {
  args: {
    output: {
      raw: JSON.stringify(shortJSONOutputFixture),
      parsed: shortJSONOutputFixture,
    },
  },
};

export const JSONFunctionNotParsedWithoutSchema: Story = {
  args: {
    output: {
      raw: JSON.stringify(shortJSONOutputFixture).slice(0, -10), // imagine this JSON got truncated somehow
      parsed: null,
    },
  },
};

export const ChatFunctionWithThinking: Story = {
  args: {
    output: [
      {
        type: "thought",
        text: "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
        signature: undefined,
        _internal_provider_type: undefined,
      },
      { type: "text", text: "Hello, world!" },
    ],
  },
};

export const ChatFunctionWithThinkingAndSignature: Story = {
  args: {
    output: [
      {
        type: "thought",
        text: "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
        signature: "b1e302e1-c3d3-4ed5-950b-1060d005a82d",
        _internal_provider_type: undefined,
      },
      { type: "text", text: "Hello, world!" },
    ],
  },
};

export const ChatFunctionWithUnknownContent: Story = {
  args: {
    output: [
      {
        type: "unknown",
        data: null,
        model_provider_name: null,
      },
      {
        type: "unknown",
        data: {
          some: "arbitrary",
          data: 123,
          structure: ["is", "not", "validated"],
        },
        model_provider_name: "openai",
      },
      { type: "text", text: "This text comes after unknown content." },
    ],
  },
};
