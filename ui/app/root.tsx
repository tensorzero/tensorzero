import * as React from "react";
import {
  data,
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useRouteLoaderData,
} from "react-router";

import { ConfigProvider } from "./context/config";
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
  isGatewayConnectionError,
  classifyError,
  getErrorLabel,
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
    // Throw typed errors that ErrorBoundary can handle
    if (isGatewayConnectionError(e)) {
      throw data(
        { errorType: InfraErrorType.GatewayUnavailable },
        { status: 503 },
      );
    }
    if (isAuthenticationError(e)) {
      throw data(
        { errorType: InfraErrorType.GatewayAuthFailed },
        { status: 401 },
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
  const { config } = loaderData;

  return (
    <AppProviders loaderData={loaderData}>
      <ConfigProvider value={config}>
        <div className="fixed inset-0 flex">
          <AppSidebar />
          <ContentLayout>
            <Outlet />
          </ContentLayout>
        </div>
      </ConfigProvider>
    </AppProviders>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  const [open, setOpen] = React.useState(true);
  const rootLoaderData = useRouteLoaderData<typeof loader>("root");

  // Reset dialog when error changes (component may re-render, not remount)
  React.useEffect(() => {
    setOpen(true);
  }, [error]);

  // Client 404s - show PageNotFound in content area
  if (isRouteErrorResponse(error) && error.status === 404) {
    if (!isInfraErrorData(error.data)) {
      return (
        <ErrorAppShell content={<PageNotFound />} loaderData={rootLoaderData} />
      );
    }
  }

  const classified = classifyError(error);
  const label = getErrorLabel(classified.type);

  return (
    <ErrorAppShell
      loaderData={rootLoaderData}
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
