import { useState } from "react";
import {
  UnknownContentBlock,
  type InvalidJsonMarker,
} from "./UnknownContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { JsonValue } from "~/types/tensorzero";

const meta = {
  title: "Input Output/Content Blocks/UnknownContentBlock",
  component: UnknownContentBlock,
  decorators: [
    (Story) => (
      <div className="w-[80vw] bg-orange-100 p-8">
        <div className="bg-white p-4">
          <Story />
        </div>
      </div>
    ),
  ],
} satisfies Meta<typeof UnknownContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SimpleObject: Story = {
  name: "Simple Object",
  args: {
    data: {
      custom_field: "This is a custom content block",
      provider_specific: true,
    },
    isEditing: false,
  },
};

export const NestedObject: Story = {
  name: "Nested Object",
  args: {
    data: {
      type: "custom_block",
      metadata: {
        source: "external_api",
        timestamp: "2024-01-15T10:30:00Z",
      },
      content: {
        items: ["item1", "item2", "item3"],
        count: 3,
      },
    },
    isEditing: false,
  },
};

export const Editing: Story = {
  name: "Editing",
  args: {
    data: { custom_field: "Edit me!", provider_specific: true },
    isEditing: true,
    onChange: () => {},
  },
  render: function EditingStory() {
    const [data, setData] = useState<JsonValue | InvalidJsonMarker>({
      custom_field: "Edit me!",
      provider_specific: true,
    });
    return (
      <UnknownContentBlock data={data} isEditing={true} onChange={setData} />
    );
  },
};

export const InvalidJson: Story = {
  name: "Invalid JSON (Error State)",
  args: {
    data: { __invalid_json__: true, raw: '{ "broken": json' },
    isEditing: true,
    onChange: () => {},
  },
};

export const ArrayData: Story = {
  name: "Array Data",
  args: {
    data: [
      { id: 1, name: "First" },
      { id: 2, name: "Second" },
      { id: 3, name: "Third" },
    ],
    isEditing: false,
  },
};

export const PrimitiveString: Story = {
  name: "Primitive String",
  args: {
    data: "Just a plain string value",
    isEditing: false,
  },
};

export const PrimitiveNumber: Story = {
  name: "Primitive Number",
  args: {
    data: 42,
    isEditing: false,
  },
};

export const NullValue: Story = {
  name: "Null Value",
  args: {
    data: null,
    isEditing: false,
  },
};
