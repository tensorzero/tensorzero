import { useState } from "react";
import { ToolCallContentBlock } from "./ToolCallContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";
import type { ToolCallWrapper } from "~/types/tensorzero";

const meta = {
  title: "Input Output/Content Blocks/ToolCallContentBlock",
  component: ToolCallContentBlock,
  decorators: [
    (Story) => (
      <div className="w-[80vw] bg-orange-100 p-8">
        <div className="bg-white p-4">
          <Story />
        </div>
      </div>
    ),
  ],
} satisfies Meta<typeof ToolCallContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  name: "Simple",
  args: {
    block: {
      id: "call_123",
      name: "get_weather",
      arguments: JSON.stringify({
        location: "San Francisco",
        unit: "celsius",
      }),
    },
    isEditing: false,
  },
};

export const SimpleEditing: Story = {
  name: "Simple (Editing)",
  args: {
    block: {
      id: "call_123",
      name: "get_weather",
      arguments: JSON.stringify({
        location: "San Francisco",
        unit: "celsius",
      }),
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function SimpleEditingStory() {
    const [block, setBlock] = useState<ToolCallWrapper>({
      id: "call_123",
      name: "get_weather",
      arguments: JSON.stringify({
        location: "San Francisco",
        unit: "celsius",
      }),
    });
    return (
      <ToolCallContentBlock
        block={block}
        isEditing={true}
        onChange={setBlock}
      />
    );
  },
};

export const ComplexArguments: Story = {
  name: "Complex Arguments",
  args: {
    block: {
      id: "call_789",
      name: "create_user",
      arguments: JSON.stringify({
        name: "John Doe",
        email: "john@example.com",
        age: 30,
        roles: ["admin", "user", "moderator"],
        preferences: {
          theme: "dark",
          notifications: {
            email: true,
            push: false,
            sms: true,
          },
          language: "en-US",
        },
        metadata: {
          created_at: "2025-10-25T12:00:00Z",
          source: "web",
        },
      }),
    },
    isEditing: false,
  },
};

export const LongValues: Story = {
  args: {
    block: {
      id: "call_" + "x".repeat(1000),
      name: "very_".repeat(1000) + "long_tool_name",
      arguments: JSON.stringify({
        text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        items: Array.from({ length: 50 }, (_, i) => `item_${i + 1}`),
      }),
    },
    isEditing: false,
  },
};

export const LongValuesEditing: Story = {
  name: "Long Values (Editing)",
  args: {
    block: {
      id: "call_" + "x".repeat(1000),
      name: "very_".repeat(1000) + "long_tool_name",
      arguments: JSON.stringify({
        text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        items: Array.from({ length: 50 }, (_, i) => `item_${i + 1}`),
      }),
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function LongValuesEditingStory() {
    const [block, setBlock] = useState<ToolCallWrapper>({
      id: "call_" + "x".repeat(1000),
      name: "very_".repeat(1000) + "long_tool_name",
      arguments: JSON.stringify({
        text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        items: Array.from({ length: 50 }, (_, i) => `item_${i + 1}`),
      }),
    });
    return (
      <ToolCallContentBlock
        block={block}
        isEditing={true}
        onChange={setBlock}
      />
    );
  },
};

export const Empty: Story = {
  args: {
    block: {
      id: "",
      name: "",
      arguments: "",
    },
    isEditing: false,
  },
};
