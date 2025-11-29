import { useState } from "react";
import { JsonOutputElement } from "./JsonOutputElement";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { JsonInferenceOutput } from "~/types/tensorzero";
import { StoryDebugWrapper } from "~/components/.storybook/StoryDebugWrapper";
import { GlobalToastProvider } from "~/providers/global-toast-provider";

const meta = {
  title: "Input Output/JsonOutputElement",
  component: JsonOutputElement,
  decorators: [
    (Story) => (
      <GlobalToastProvider>
        <Story />
      </GlobalToastProvider>
    ),
  ],
} satisfies Meta<typeof JsonOutputElement>;

export default meta;
type Story = StoryObj<typeof meta>;

export const None: Story = {
  args: {
    output: undefined,
    isEditing: false,
  },
  render: function Component(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <JsonOutputElement {...args} />
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
  render: function Component(args) {
    const [output, setOutput] = useState<JsonInferenceOutput | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="output" debugData={output}>
        <JsonOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Trivial: Story = {
  args: {
    output: {
      parsed: {},
      raw: "{}",
    },
    isEditing: false,
  },
  render: function Component(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <JsonOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const TrivialEditing: Story = {
  name: "Trivial (Editing)",
  args: {
    output: {
      parsed: {},
      raw: "{}",
    },
    isEditing: true,
  },
  render: function Component(args) {
    const [output, setOutput] = useState<JsonInferenceOutput | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="output" debugData={output}>
        <JsonOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};

const COMPLEX_JSON = {
  key1: "value1",
  key2: {
    subkey1: "subvalue1",
    subkey2: "subvalue2",
  },
  key3: ["array1", "array2", "array3"],
};

export const Complex: Story = {
  args: {
    output: {
      parsed: COMPLEX_JSON,
      raw: JSON.stringify(COMPLEX_JSON),
    },
    isEditing: false,
  },
  render: function Component(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <JsonOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const ComplexEditing: Story = {
  name: "Complex (Editing)",
  args: {
    output: {
      parsed: COMPLEX_JSON,
      raw: JSON.stringify(COMPLEX_JSON),
    },
    isEditing: true,
  },
  render: function Component(args) {
    const [output, setOutput] = useState<JsonInferenceOutput | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="output" debugData={output}>
        <JsonOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};

export const Invalid: Story = {
  args: {
    output: {
      parsed: null,
      raw: JSON.stringify(COMPLEX_JSON).slice(0, -1),
    },
    isEditing: false,
  },
  render: function Component(args) {
    return (
      <StoryDebugWrapper debugLabel="output" debugData={args.output}>
        <JsonOutputElement {...args} />
      </StoryDebugWrapper>
    );
  },
};

export const InvalidEditing: Story = {
  name: "Invalid (Editing)",
  args: {
    output: {
      parsed: null,
      raw: JSON.stringify(COMPLEX_JSON).slice(0, -1),
    },
    isEditing: true,
  },
  render: function Component(args) {
    const [output, setOutput] = useState<JsonInferenceOutput | undefined>(
      args.output,
    );
    return (
      <StoryDebugWrapper debugLabel="output" debugData={output}>
        <JsonOutputElement
          {...args}
          output={output}
          onOutputChange={setOutput}
        />
      </StoryDebugWrapper>
    );
  },
};
