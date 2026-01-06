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
import { AppSidebar } from "./components/layout/app.sidebar";
import { GatewayAuthFailedState } from "./components/ui/error/GatewayAuthFailedState";
import { GatewayRequiredState } from "./components/ui/error/GatewayRequiredState";
import { RouteNotFoundState } from "./components/ui/error/RouteNotFoundState";
import {
  isAuthenticationError,
  isGatewayConnectionError,
  isRouteNotFoundError,
} from "./utils/tensorzero/errors";
import { SidebarProvider } from "./components/ui/sidebar";
import { ContentLayout } from "./components/layout/ContentLayout";
import { startPeriodicCleanup } from "./utils/evaluations.server";
import { ReactQueryProvider } from "./providers/react-query";
import { isReadOnlyMode, readOnlyMiddleware } from "./utils/read-only.server";
import { TooltipProvider } from "~/components/ui/tooltip";
import { GlobalToastProvider } from "~/providers/global-toast-provider";
import { Toaster } from "~/components/ui/toaster";

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

const GATEWAY_UNAVAILABLE_ERROR = "TensorZero Gateway Unavailable";
const GATEWAY_AUTH_FAILED_ERROR = "TensorZero Gateway Authentication Failed";

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
      throw data({ errorType: GATEWAY_UNAVAILABLE_ERROR }, { status: 503 });
    }
    if (isAuthenticationError(e)) {
      throw data({ errorType: GATEWAY_AUTH_FAILED_ERROR }, { status: 401 });
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
    <ReactQueryProvider>
      <GlobalToastProvider>
        <ReadOnlyProvider value={isReadOnly}>
          <AutopilotAvailableProvider value={autopilotAvailable}>
            <ConfigProvider value={config}>
              <SidebarProvider>
                <TooltipProvider>
                  <div className="fixed inset-0 flex">
                    <AppSidebar />
                    <ContentLayout>
                      <Outlet />
                    </ContentLayout>
                  </div>
                </TooltipProvider>
              </SidebarProvider>
            </ConfigProvider>
          </AutopilotAvailableProvider>
        </ReadOnlyProvider>
        <Toaster />
      </GlobalToastProvider>
    </ReactQueryProvider>
  );
}

// Fallback Error Boundary
//
// This is the first step in a larger overhaul of error handling in the UI.
// The goal is to provide a consistent, dismissible error experience that:
// - Keeps the sidebar visible so users can navigate even when errors occur
// - Shows errors in a modal overlay that can be dismissed
// - Provides actionable troubleshooting guidance
//
// This PR focuses specifically on route-not-found errors (#5504).
// Follow-up PRs will extend this pattern to other error types.
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  const [dismissed, setDismissed] = React.useState(false);

  // Check if this is a gateway connection error (wrapped with data() or raw)
  if (
    isRouteErrorResponse(error) &&
    error.data?.errorType === GATEWAY_UNAVAILABLE_ERROR
  ) {
    return <GatewayRequiredState />;
  }
  if (isGatewayConnectionError(error)) {
    return <GatewayRequiredState />;
  }

  // Check if this is a gateway authentication error (wrapped with data() or raw)
  if (
    isRouteErrorResponse(error) &&
    error.data?.errorType === GATEWAY_AUTH_FAILED_ERROR
  ) {
    return <GatewayAuthFailedState />;
  }
  if (isAuthenticationError(error)) {
    return <GatewayAuthFailedState />;
  }

  // Route not found errors (gateway API 404s) use a dismissible modal overlay.
  // This keeps the sidebar visible so users can still navigate the app.
  if (isRouteNotFoundError(error)) {
    // Extract route info from error message if available
    const errorMessage =
      error instanceof Error
        ? error.message
        : typeof error === "object" &&
            error !== null &&
            "message" in error &&
            typeof error.message === "string"
          ? error.message
          : "";
    const routeMatch = errorMessage.match(/Route not found: (\w+) (.+)/);
    const routeInfo = routeMatch ? `${routeMatch[1]} ${routeMatch[2]}` : null;

    return (
      <ReactQueryProvider>
        <GlobalToastProvider>
          <SidebarProvider>
            <TooltipProvider>
              <div className="fixed inset-0 flex">
                <AppSidebar />
                <ContentLayout />
              </div>
              {!dismissed && (
                <div
                  className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 pb-24"
                  onClick={() => setDismissed(true)}
                >
                  <div
                    className="max-h-[90vh] overflow-auto"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <RouteNotFoundState
                      routeInfo={routeInfo}
                      onDismiss={() => setDismissed(true)}
                    />
                  </div>
                </div>
              )}
              {dismissed && (
                <button
                  onClick={() => setDismissed(false)}
                  className="fixed bottom-4 left-4 z-50 flex items-center gap-2 rounded-lg bg-red-500 px-3 py-2 text-sm font-medium text-white shadow-lg hover:bg-red-600"
                >
                  <span className="h-2 w-2 rounded-full bg-white" />1 Error
                </button>
              )}
            </TooltipProvider>
          </SidebarProvider>
          <Toaster />
        </GlobalToastProvider>
      </ReactQueryProvider>
    );
  }

  // Generic error fallback (including client 404s)
  let message = "Oops!";
  let details = "An unexpected error occurred.";
  let stack: string | undefined;

  if (isRouteErrorResponse(error)) {
    message = error.status === 404 ? "404" : "Error";
    details =
      error.status === 404
        ? "The requested page could not be found."
        : error.statusText || details;
  } else if (import.meta.env.DEV && error && error instanceof Error) {
    details = error.message;
    stack = error.stack;
  }

  return (
    <main className="container mx-auto p-4 pt-16">
      <h1>{message}</h1>
      <p>{details}</p>
      {stack && (
        <pre className="w-full overflow-x-auto p-4">
          <code>{stack}</code>
        </pre>
      )}
    </main>
  );
}
