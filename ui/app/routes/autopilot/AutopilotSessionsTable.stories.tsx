import type { Meta, StoryObj } from "@storybook/react-vite";
import AutopilotSessionsTable from "./AutopilotSessionsTable";
import type { Session } from "~/types/tensorzero";

const BASE_TIME = new Date("2024-08-15T16:30:00Z").getTime();

function buildSession(id: string, index: number, version: string): Session {
  return {
    id,
    organization_id: "00000000-0000-0000-0000-000000000000",
    workspace_id: "00000000-0000-0000-0000-000000000000",
    deployment_id: "00000000-0000-0000-0000-000000000000",
    tensorzero_version: version,
    created_at: new Date(BASE_TIME - index * 45 * 60 * 1000).toISOString(),
  };
}

const sessions: Session[] = [
  buildSession("d1a0b0c0-0000-0000-0000-000000000001", 0, "2026.1.0"),
  buildSession("d1a0b0c0-0000-0000-0000-000000000002", 1, "2026.1.0"),
  buildSession("d1a0b0c0-0000-0000-0000-000000000003", 2, "2026.2.7"),
  buildSession("d1a0b0c0-0000-0000-0000-000000000004", 3, "2026.3.4"),
  buildSession("d1a0b0c0-0000-0000-0000-000000000005", 4, "2026.1.0"),
];

const meta = {
  title: "Autopilot/AutopilotSessionsTable",
  component: AutopilotSessionsTable,
  render: (args) => (
    <div className="w-[80vw] p-4">
      <AutopilotSessionsTable {...args} />
    </div>
  ),
} satisfies Meta<typeof AutopilotSessionsTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    sessions: [],
  },
};

export const MatchingVersions: Story = {
  args: {
    sessions: sessions.map((session) => ({
      ...session,
      tensorzero_version: "2026.1.0",
    })),
    gatewayVersion: "2026.1.0",
    uiVersion: "2026.1.0",
  },
};

export const MixedVersions: Story = {
  args: {
    sessions,
    gatewayVersion: "2026.1.0",
    uiVersion: "2026.1.0",
  },
};

export const SessionMismatch: Story = {
  args: {
    sessions: [
      buildSession("d1a0b0c0-0000-0000-0000-000000000006", 0, "2026.4.2"),
    ],
    gatewayVersion: "2026.1.0",
    uiVersion: "2026.1.0",
  },
};

export const GatewayUiMismatch: Story = {
  args: {
    sessions: [
      buildSession("d1a0b0c0-0000-0000-0000-000000000007", 0, "2026.1.0"),
      buildSession("d1a0b0c0-0000-0000-0000-000000000008", 1, "2026.2.7"),
    ],
    gatewayVersion: "2026.1.0",
    uiVersion: "2026.2.7",
  },
};
