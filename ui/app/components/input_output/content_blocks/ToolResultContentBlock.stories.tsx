import { useState } from "react";
import { ToolResultContentBlock } from "./ToolResultContentBlock";
import type { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
  title: "Input Output/Content Blocks/ToolResultContentBlock",
  component: ToolResultContentBlock,
  decorators: [
    (Story) => (
      <div className="w-[80vw] bg-orange-100 p-8">
        <div className="bg-white p-4">
          <Story />
        </div>
      </div>
    ),
  ],
} satisfies Meta<typeof ToolResultContentBlock>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  name: "Simple",
  args: {
    block: {
      id: "call_123",
      name: "get_weather",
      result: "Weather in San Francisco: 72°F, Sunny",
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
      result: "Weather in San Francisco: 72°F, Sunny",
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function SimpleEditingStory() {
    const [block, setBlock] = useState({
      id: "call_123",
      name: "get_weather",
      result: "Weather in San Francisco: 72°F, Sunny",
    });
    return (
      <ToolResultContentBlock
        block={block}
        isEditing={true}
        onChange={setBlock}
      />
    );
  },
};

export const JSONResult: Story = {
  name: "JSON Result",
  args: {
    block: {
      id: "call_789",
      name: "create_user",
      result: JSON.stringify({
        success: true,
        user_id: "12345",
        name: "John Doe",
        email: "john@example.com",
        roles: ["admin", "user"],
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
      result: (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ".repeat(
          10,
        ) + "\n"
      ).repeat(100),
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
      result: (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ".repeat(
          10,
        ) + "\n"
      ).repeat(100),
    },
    isEditing: true,
    onChange: () => {},
  },
  render: function LongValuesEditingStory() {
    const [block, setBlock] = useState({
      id: "call_" + "x".repeat(1000),
      name: "very_".repeat(1000) + "long_tool_name",
      result: (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ".repeat(
          10,
        ) + "\n"
      ).repeat(100),
    });
    return (
      <ToolResultContentBlock
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
      result: "",
    },
    isEditing: false,
  },
};
