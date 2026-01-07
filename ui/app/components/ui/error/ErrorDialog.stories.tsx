import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { ErrorDialog } from "./ErrorDialog";
import { ErrorContent } from "./ErrorContent";
import { BoundaryErrorType } from "~/utils/tensorzero/errors";

function InteractiveWrapper({
  type,
  message,
  routeInfo,
  status,
  stack,
  label,
}: {
  type: BoundaryErrorType;
  message?: string;
  routeInfo?: string;
  status?: number;
  stack?: string;
  label?: string;
}) {
  const [open, setOpen] = useState(true);
  return (
    <ErrorDialog
      open={open}
      onDismiss={() => setOpen(false)}
      onReopen={() => setOpen(true)}
      label={label}
    >
      <ErrorContent
        type={type}
        message={message}
        routeInfo={routeInfo}
        status={status}
        stack={stack}
      />
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
    type: BoundaryErrorType.GatewayUnavailable,
    label: "Connection Error",
  },
};

export const AuthenticationFailed: Story = {
  args: {
    type: BoundaryErrorType.GatewayAuthFailed,
    label: "Auth Error",
  },
};

export const RouteNotFound: Story = {
  args: {
    type: BoundaryErrorType.RouteNotFound,
    routeInfo: "GET /api/v1/unknown",
    label: "Route Error",
  },
};

export const RouteNotFoundWithoutInfo: Story = {
  args: {
    type: BoundaryErrorType.RouteNotFound,
    label: "Route Error",
  },
};

export const ServerError: Story = {
  args: {
    type: BoundaryErrorType.ServerError,
    status: 500,
    message: "Internal server error",
    label: "Server Error",
  },
};

export const ServerErrorWithStack: Story = {
  args: {
    type: BoundaryErrorType.ServerError,
    status: 500,
    message: "Internal server error",
    stack: `Error: Internal server error
    at loader (/app/routes/observability/inferences/route.tsx:42:11)
    at callRouteLoader (node_modules/react-router/dist/development/chunk.js:1234:22)
    at async Promise.all (index 0)`,
    label: "Server Error",
  },
};

export const ServerErrorWithoutStatus: Story = {
  args: {
    type: BoundaryErrorType.ServerError,
    message: "Something unexpected happened",
    label: "Server Error",
  },
};

export const ClickHouseError: Story = {
  args: {
    type: BoundaryErrorType.ClickHouseConnection,
    message: "Connection refused to ClickHouse at localhost:8123",
    label: "Database Error",
  },
};

export const ClickHouseErrorDefault: Story = {
  args: {
    type: BoundaryErrorType.ClickHouseConnection,
    label: "Database Error",
  },
};
