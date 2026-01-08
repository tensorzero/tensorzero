import * as React from "react";
import {
  data,
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "react-router";

import { ConfigProvider } from "./context/config";
import { ReadOnlyProvider } from "./context/read-only";
import { AutopilotAvailableProvider } from "./context/autopilot-available";
import type { Route } from "./+types/root";
import "./tailwind.css";
import {
  getConfig,
  checkAutopilotAvailable,
} from "./utils/config/index.server";
import { AlertTriangle } from "lucide-react";
import { AppSidebar } from "./components/layout/app.sidebar";
import {
  RootErrorBoundaryLayout,
  ErrorContent,
  ErrorDialog,
} from "./components/ui/error";
import {
  ErrorContentCard,
  ErrorContentHeader,
  ErrorContext,
} from "./components/ui/error/ErrorContentPrimitives";
import {
  BoundaryErrorType,
  isBoundaryErrorData,
  isAuthenticationError,
  isGatewayConnectionError,
  isRouteNotFoundError,
  isClickHouseError,
  type ClassifiedError,
} from "./utils/tensorzero/errors";
import { ContentLayout } from "./components/layout/ContentLayout";
import { startPeriodicCleanup } from "./utils/evaluations.server";
import { AppProviders } from "./providers/app-providers";
import { isReadOnlyMode, readOnlyMiddleware } from "./utils/read-only.server";

export const links: Route.LinksFunction = () => [
  { rel: "preconnect", href: "https://fonts.googleapis.com" },
  {
    rel: "preconnect",
    href: "https://fonts.gstatic.com",
    crossOrigin: "anonymous",
  },
  {
    rel: "stylesheet",
    href: "https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&family=Geist:wght@100..900&display=swap",
  },
  {
    rel: "icon",
    type: "image/svg+xml",
    href: "/favicon.svg",
  },
];

export const middleware: Route.MiddlewareFunction[] = [readOnlyMiddleware];

export async function loader() {
  // Initialize evaluation cleanup when the app loads
  startPeriodicCleanup();
  const isReadOnly = isReadOnlyMode();
  try {
    // Fetch config and autopilot availability in parallel
    const [config, autopilotAvailable] = await Promise.all([
      getConfig(),
      checkAutopilotAvailable(),
    ]);
    return { config, isReadOnly, autopilotAvailable };
  } catch (e) {
    if (isGatewayConnectionError(e)) {
      throw data(
        { errorType: BoundaryErrorType.GatewayUnavailable },
        { status: 503 },
      );
    }
    if (isAuthenticationError(e)) {
      throw data(
        { errorType: BoundaryErrorType.GatewayAuthFailed },
        { status: 401 },
      );
    }
    if (isClickHouseError(e)) {
      const message = e instanceof Error ? e.message : undefined;
      throw data(
        { errorType: BoundaryErrorType.ClickHouseConnection, message },
        { status: 503 },
      );
    }
    throw e;
  }
}

// Global Layout
export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App({ loaderData }: Route.ComponentProps) {
  const { config, isReadOnly, autopilotAvailable } = loaderData;

  return (
    <AppProviders>
      <ReadOnlyProvider value={isReadOnly}>
        <AutopilotAvailableProvider value={autopilotAvailable}>
          <ConfigProvider value={config}>
            <div className="fixed inset-0 flex">
              <AppSidebar />
              <ContentLayout>
                <Outlet />
              </ContentLayout>
            </div>
          </ConfigProvider>
        </AutopilotAvailableProvider>
      </ReadOnlyProvider>
    </AppProviders>
  );
}

function classifyError(error: unknown): ClassifiedError {
  // Check for serialized BoundaryErrorData (from data() throws)
  if (isRouteErrorResponse(error) && isBoundaryErrorData(error.data)) {
    const { errorType } = error.data;
    switch (errorType) {
      case BoundaryErrorType.GatewayUnavailable:
        return { type: BoundaryErrorType.GatewayUnavailable };
      case BoundaryErrorType.GatewayAuthFailed:
        return { type: BoundaryErrorType.GatewayAuthFailed };
      case BoundaryErrorType.RouteNotFound:
        return {
          type: BoundaryErrorType.RouteNotFound,
          routeInfo: error.data.routeInfo,
        };
      case BoundaryErrorType.ClickHouseConnection:
        return {
          type: BoundaryErrorType.ClickHouseConnection,
          message: "message" in error.data ? error.data.message : undefined,
        };
      case BoundaryErrorType.ServerError:
        return {
          type: BoundaryErrorType.ServerError,
          message: "message" in error.data ? error.data.message : undefined,
          status: error.status,
        };
      default: {
        const _exhaustiveCheck: never = errorType;
        return { type: BoundaryErrorType.ServerError, status: error.status };
      }
    }
  }

  if (isGatewayConnectionError(error)) {
    return { type: BoundaryErrorType.GatewayUnavailable };
  }

  if (isAuthenticationError(error)) {
    return { type: BoundaryErrorType.GatewayAuthFailed };
  }

  if (isRouteNotFoundError(error)) {
    const errorMessage = extractErrorMessage(error);
    const routeMatch = errorMessage.match(/Route not found: (\w+) (.+)/);
    const routeInfo = routeMatch
      ? `${routeMatch[1]} ${routeMatch[2]}`
      : errorMessage;
    return { type: BoundaryErrorType.RouteNotFound, routeInfo };
  }

  if (isClickHouseError(error)) {
    const message = error instanceof Error ? error.message : undefined;
    return { type: BoundaryErrorType.ClickHouseConnection, message };
  }

  const message = isRouteErrorResponse(error)
    ? error.statusText || undefined
    : error instanceof Error
      ? error.message
      : undefined;
  const status = isRouteErrorResponse(error) ? error.status : undefined;
  return { type: BoundaryErrorType.ServerError, message, status };
}

function extractErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof error.message === "string"
  ) {
    return error.message;
  }
  return "";
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  const [open, setOpen] = React.useState(true);

  // Client 404s (page not found in React Router) - show in content area with sidebar
  if (isRouteErrorResponse(error) && error.status === 404) {
    // Ensure this is actually a client 404, not a gateway route not found
    if (!isBoundaryErrorData(error.data)) {
      return (
        <RootErrorBoundaryLayout>
          <div className="fixed inset-0 z-50 flex items-center justify-center p-8 pb-20">
            <ErrorContentCard context={ErrorContext.Page}>
              <ErrorContentHeader
                icon={AlertTriangle}
                title="Error 404"
                description="The requested page could not be found."
                context={ErrorContext.Page}
              />
            </ErrorContentCard>
          </div>
        </RootErrorBoundaryLayout>
      );
    }
  }

  // All other errors use the dismissible modal pattern with sidebar visible
  const classified = classifyError(error);
  const label = getErrorLabel(classified.type);

  return (
    <RootErrorBoundaryLayout>
      <ErrorDialog
        open={open}
        onDismiss={() => setOpen(false)}
        onReopen={() => setOpen(true)}
        label={label}
      >
        <ErrorContent error={classified} />
      </ErrorDialog>
    </RootErrorBoundaryLayout>
  );
}

function getErrorLabel(type: BoundaryErrorType): string {
  switch (type) {
    case BoundaryErrorType.GatewayUnavailable:
      return "Connection Error";
    case BoundaryErrorType.GatewayAuthFailed:
      return "Auth Error";
    case BoundaryErrorType.RouteNotFound:
      return "Route Error";
    case BoundaryErrorType.ClickHouseConnection:
      return "Database Error";
    case BoundaryErrorType.ServerError:
      return "Server Error";
    default: {
      const _exhaustiveCheck: never = type;
      return "Server Error";
    }
  }
}
