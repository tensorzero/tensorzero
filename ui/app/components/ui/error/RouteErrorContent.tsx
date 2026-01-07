/**
 * Route-level error content for displaying errors within the content area.
 *
 * ERROR BOUNDARY ARCHITECTURE:
 * ---------------------------
 * This codebase uses a two-tier error boundary strategy:
 *
 * 1. ROOT ErrorBoundary (root.tsx):
 *    - Catches startup failures (gateway down, auth failed, ClickHouse unavailable)
 *    - Shows full-page dark modal - appropriate because no app shell exists yet
 *    - Uses ErrorContent + ErrorDialog components
 *
 * 2. SECTION LAYOUT ErrorBoundaries (e.g., datasets/layout.tsx):
 *    - Catches errors after app has loaded (API failures, validation, render errors)
 *    - Shows error WITHIN the content area - sidebar stays visible
 *    - Uses RouteErrorContent (this file) for consistent light-theme styling
 *    - Standalone routes without a parent layout define their own ErrorBoundary
 *
 * This separation ensures users see as much of the UI as possible during errors.
 */

import { AlertTriangle } from "lucide-react";
import { isRouteErrorResponse } from "react-router";
import {
  ErrorContentCard,
  ErrorContentHeader,
  ErrorStyle,
} from "./ErrorContentPrimitives";

interface RouteErrorContentProps {
  error: unknown;
}

export function RouteErrorContent({ error }: RouteErrorContentProps) {
  const { title, message, status } = extractErrorInfo(error);

  return (
    <div className="flex min-h-full items-center justify-center p-8 pb-20">
      <ErrorContentCard variant={ErrorStyle.Light}>
        <ErrorContentHeader
          icon={AlertTriangle}
          title={status ? `Error ${status}` : title}
          description={message}
          showBorder={false}
          variant={ErrorStyle.Light}
        />
      </ErrorContentCard>
    </div>
  );
}

function extractErrorInfo(error: unknown): {
  title: string;
  message: string;
  status?: number;
} {
  if (isRouteErrorResponse(error)) {
    return {
      title: error.statusText || "Error",
      message:
        typeof error.data === "string"
          ? error.data
          : "An unexpected error occurred.",
      status: error.status,
    };
  }

  if (error instanceof Error) {
    return {
      title: "Error",
      message: error.message,
    };
  }

  return {
    title: "Error",
    message: "An unexpected error occurred.",
  };
}
