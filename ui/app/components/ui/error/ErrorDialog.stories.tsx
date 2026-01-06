import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { ErrorDialog } from "./ErrorDialog";
import { GatewayUnavailableErrorContent } from "./GatewayUnavailableErrorContent";
import { GatewayAuthErrorContent } from "./GatewayAuthErrorContent";
import { RouteNotFoundErrorContent } from "./RouteNotFoundErrorContent";
import { ServerErrorContent } from "./ServerErrorContent";
import { ClickHouseErrorContent } from "./ClickHouseErrorContent";

function InteractiveWrapper({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <ErrorDialog
      open={open}
      onDismiss={() => setOpen(false)}
      onReopen={() => setOpen(true)}
    >
      {children}
    </ErrorDialog>
  );
}

const meta = {
  title: "Error/ErrorDialog",
  component: InteractiveWrapper,
  parameters: {
    layout: "fullscreen",
  },
  decorators: [
    (Story) => (
      <div className="min-h-screen bg-gray-100 p-8">
        <div className="mb-4 rounded bg-white p-4 shadow">
          <p className="text-gray-600">
            Background content to show overlay effect
          </p>
        </div>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof InteractiveWrapper>;

export default meta;
type Story = StoryObj<typeof meta>;

export const GatewayUnavailable: Story = {
  args: {
    children: <GatewayUnavailableErrorContent />,
  },
};

export const AuthenticationFailed: Story = {
  args: {
    children: <GatewayAuthErrorContent />,
  },
};

export const RouteNotFound: Story = {
  args: {
    children: <RouteNotFoundErrorContent routeInfo="GET /api/v1/unknown" />,
  },
};

export const RouteNotFoundWithoutInfo: Story = {
  args: {
    children: <RouteNotFoundErrorContent />,
  },
};

export const ServerError: Story = {
  args: {
    children: (
      <ServerErrorContent status={500} message="Internal server error" />
    ),
  },
};

export const ServerErrorWithStack: Story = {
  args: {
    children: (
      <ServerErrorContent
        status={500}
        message="Internal server error"
        stack={`Error: Internal server error
    at loader (/app/routes/observability/inferences/route.tsx:42:11)
    at callRouteLoader (node_modules/react-router/dist/development/chunk.js:1234:22)
    at async Promise.all (index 0)`}
      />
    ),
  },
};

export const ServerErrorWithoutStatus: Story = {
  args: {
    children: <ServerErrorContent message="Something unexpected happened" />,
  },
};

export const ClickHouseError: Story = {
  args: {
    children: (
      <ClickHouseErrorContent message="Connection refused to ClickHouse at localhost:8123" />
    ),
  },
};

export const ClickHouseErrorDefault: Story = {
  args: {
    children: <ClickHouseErrorContent />,
  },
};
