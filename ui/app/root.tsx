import * as React from "react";
import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "react-router";

import { ConfigProvider, EMPTY_CONFIG } from "./context/config";
import type { UiConfig } from "./types/tensorzero";
import { ReadOnlyProvider } from "./context/read-only";
import { AutopilotAvailableProvider } from "./context/autopilot-available";
import type { Route } from "./+types/root";
import "./tailwind.css";
import {
  getConfig,
  checkAutopilotAvailable,
} from "./utils/config/index.server";
import { AppSidebar } from "./components/layout/app.sidebar";
import {
  ErrorAppShell,
  ErrorContent,
  ErrorDialog,
  PageNotFound,
} from "./components/ui/error";
import {
  InfraErrorType,
  isInfraErrorData,
  isAuthenticationError,
  isClickHouseError,
  isGatewayConnectionError,
  classifyError,
  isClickHouseError,
  getErrorLabel,
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

interface LoaderData {
  config: UiConfig;
  isReadOnly: boolean;
  autopilotAvailable: boolean;
  infraError: ClassifiedError | null;
}

export async function loader(): Promise<LoaderData> {
  // Initialize evaluation cleanup when the app loads
  startPeriodicCleanup();
  const isReadOnly = isReadOnlyMode();
  try {
    // Fetch config and autopilot availability in parallel
    const [config, autopilotAvailable] = await Promise.all([
      getConfig(),
      checkAutopilotAvailable(),
    ]);
    return { config, isReadOnly, autopilotAvailable, infraError: null };
  } catch (e) {
    // Graceful degradation for infrastructure errors:
    // Return fallback state so UI renders with dismissible error dialog.
    // Child routes will handle their own errors via their error boundaries.
    if (isGatewayConnectionError(e)) {
      return {
        config: EMPTY_CONFIG,
        isReadOnly,
        autopilotAvailable: false,
        infraError: { type: InfraErrorType.GatewayUnavailable },
      };
    }
    if (isAuthenticationError(e)) {
      return {
        config: EMPTY_CONFIG,
        isReadOnly,
        autopilotAvailable: false,
        infraError: { type: InfraErrorType.GatewayAuthFailed },
      };
    }
    if (isClickHouseError(e)) {
      const message = e instanceof Error ? e.message : undefined;
      return {
        config: EMPTY_CONFIG,
        isReadOnly,
        autopilotAvailable: false,
        infraError: {
          type: InfraErrorType.ClickHouseUnavailable,
          message,
        },
      };
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
  const { config, isReadOnly, autopilotAvailable, infraError } = loaderData;
  const [dialogOpen, setDialogOpen] = React.useState(true);

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
            {infraError && (
              <ErrorDialog
                open={dialogOpen}
                onDismiss={() => setDialogOpen(false)}
                onReopen={() => setDialogOpen(true)}
                label={getErrorLabel(infraError.type)}
              >
                <ErrorContent error={infraError} />
              </ErrorDialog>
            )}
          </ConfigProvider>
        </AutopilotAvailableProvider>
      </ReadOnlyProvider>
    </AppProviders>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  const [open, setOpen] = React.useState(true);

  // Client 404s (page not found in React Router) - show in content area with sidebar
  // Check that it's not an infrastructure error (those go through classifyError)
  if (isRouteErrorResponse(error) && error.status === 404) {
    if (!isInfraErrorData(error.data)) {
      return <ErrorAppShell content={<PageNotFound />} />;
    }
  }

  // All other errors use the dismissible modal pattern with sidebar visible
  const classified = classifyError(error);
  const label = getErrorLabel(classified.type);

  return (
    <ErrorAppShell
      overlay={
        <ErrorDialog
          open={open}
          onDismiss={() => setOpen(false)}
          onReopen={() => setOpen(true)}
          label={label}
        >
          <ErrorContent error={classified} />
        </ErrorDialog>
      }
    />
  );
}
