import type { Meta, StoryObj } from "@storybook/react-vite";
import { FunctionTypeBadge } from "./FunctionSelector";

const meta = {
  title: "UI/FunctionTypeBadge",
  component: FunctionTypeBadge,
  decorators: [
    (Story) => (
      <div className="p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof FunctionTypeBadge>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Chat: Story = {
  args: { type: "chat" },
};

export const Json: Story = {
  args: { type: "json" },
};

export const AllTypes: Story = {
  args: { type: "chat" },
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <FunctionTypeBadge type="chat" />
        <span className="text-sm text-gray-500">Chat function</span>
      </div>
      <div className="flex items-center gap-2">
        <FunctionTypeBadge type="json" />
        <span className="text-sm text-gray-500">JSON function</span>
      </div>
    </div>
  ),
};
