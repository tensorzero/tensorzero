import type { Meta, StoryObj } from "@storybook/react-vite";
import AuthTable from "./AuthTable";
import type { KeyInfo } from "tensorzero-node";

const meta = {
  title: "API Keys/AuthTable",
  component: AuthTable,
  render: (args) => (
    <div className="w-[80vw] p-4">
      <AuthTable apiKeys={args.apiKeys} />
    </div>
  ),
} satisfies Meta<typeof AuthTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    apiKeys: [],
  },
};

export const Simple: Story = {
  args: {
    apiKeys: [
      {
        public_id: "a1b2c3d4e5f6",
        organization: "default",
        workspace: "default",
        description: "Production API key for main service",
        created_at: "2024-01-15T10:30:00Z",
        disabled_at: null,
      },
      {
        public_id: "x9y8z7w6v5u4",
        organization: "default",
        workspace: "default",
        description: "Development environment key",
        created_at: "2024-02-20T14:15:00Z",
        disabled_at: null,
      },
      {
        public_id: "m3n4o5p6q7r8",
        organization: "default",
        workspace: "default",
        description: null,
        created_at: "2024-03-01T09:00:00Z",
        disabled_at: null,
      },
      {
        public_id: "k2l3m4n5o6p7",
        organization: "default",
        workspace: "default",
        description: "Testing key - can be rotated",
        created_at: "2024-03-05T11:20:00Z",
        disabled_at: "2024-03-15T14:30:00Z",
      },
      {
        public_id: "g8h9i0j1k2l3",
        organization: "default",
        workspace: "default",
        description: null,
        created_at: "2024-02-28T16:45:00Z",
        disabled_at: "2024-03-10T09:15:00Z",
      },
      {
        public_id: "s4t5u6v7w8x9",
        organization: "default",
        workspace: "default",
        description: "Staging environment - deprecated",
        created_at: "2024-03-10T13:00:00Z",
        disabled_at: null,
      },
    ] satisfies KeyInfo[],
  },
};
