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

const TRIVIAL_OUTPUT_SCHEMA = {
  $schema: "http://json-schema.org/draft-07/schema#",
  type: "object",
};

export const None: Story = {
  args: {
    output: undefined,
    outputSchema: TRIVIAL_OUTPUT_SCHEMA,
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
    outputSchema: TRIVIAL_OUTPUT_SCHEMA,
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
    outputSchema: TRIVIAL_OUTPUT_SCHEMA,
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
    outputSchema: TRIVIAL_OUTPUT_SCHEMA,
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
  person: ["Gabriel Bianconi", "Megumin"],
  organization: ["TensorZero", "Crunchyroll"],
  location: ["San Francisco", "Seattle", "New York"],
  miscellaneous: ["AI", "Technology", "Anime"],
};

const COMPLEX_OUTPUT_SCHEMA = {
  $schema: "http://json-schema.org/draft-07/schema#",
  type: "object",
  properties: {
    person: {
      type: "array",
      items: {
        type: "string",
      },
    },
    organization: {
      type: "array",
      items: {
        type: "string",
      },
    },
    location: {
      type: "array",
      items: {
        type: "string",
      },
    },
    miscellaneous: {
      type: "array",
      items: {
        type: "string",
      },
    },
  },
  required: ["person", "organization", "location", "miscellaneous"],
  additionalProperties: false,
};

export const Complex: Story = {
  args: {
    output: {
      parsed: COMPLEX_JSON,
      raw: JSON.stringify(COMPLEX_JSON),
    },
    outputSchema: COMPLEX_OUTPUT_SCHEMA,
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
    outputSchema: COMPLEX_OUTPUT_SCHEMA,
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
    outputSchema: COMPLEX_OUTPUT_SCHEMA,
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
    outputSchema: COMPLEX_OUTPUT_SCHEMA,
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
