import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "react-router";

import { ConfigProvider } from "./context/config";
import type { Route } from "./+types/root";
import "./tailwind.css";
import { getConfig } from "./utils/config/index.server";
import { AppSidebar } from "./components/layout/app.sidebar";
import { SidebarProvider } from "./components/ui/sidebar";
import { ContentLayout } from "./components/layout/ContentLayout";
import { startPeriodicCleanup } from "./utils/evaluations.server";
import { ReactQueryProvider } from "./providers/react-query";

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

export async function loader() {
  // Initialize evaluation cleanup when the app loads
  startPeriodicCleanup();
  return await getConfig();
}

// Global Layout
export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <script
          crossOrigin="anonymous"
          src="//unpkg.com/react-scan/dist/auto.global.js"
        />
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

export default function App({ loaderData: config }: Route.ComponentProps) {
  return (
    <ReactQueryProvider>
      <ConfigProvider value={config}>
        <SidebarProvider>
          <div className="fixed inset-0 flex">
            <AppSidebar />
            <ContentLayout>
              <Outlet />
            </ContentLayout>
          </div>
        </SidebarProvider>
      </ConfigProvider>
    </ReactQueryProvider>
  );
}

// Fallback Error Boundary
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
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
