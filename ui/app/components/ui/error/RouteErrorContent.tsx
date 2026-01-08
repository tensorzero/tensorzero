import { AlertTriangle } from "lucide-react";
import { isRouteErrorResponse } from "react-router";
import {
  ErrorContentCard,
  ErrorContentHeader,
  ErrorContext,
} from "./ErrorContentPrimitives";

interface RouteErrorContentProps {
  error: unknown;
}

export function RouteErrorContent({ error }: RouteErrorContentProps) {
  const { title, message, status } = extractErrorInfo(error);

  return (
    <div className="flex min-h-full items-center justify-center p-8 pb-20">
      <ErrorContentCard context={ErrorContext.Page}>
        <ErrorContentHeader
          icon={AlertTriangle}
          title={status ? `Error ${status}` : title}
          description={message}
          context={ErrorContext.Page}
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
