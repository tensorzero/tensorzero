import { AlertTriangle, FileQuestion } from "lucide-react";
import { isRouteErrorResponse } from "react-router";
import { ErrorScope, SimpleErrorDisplay } from "./ErrorContentPrimitives";

interface RouteErrorContentProps {
  error: unknown;
}

export function RouteErrorContent({ error }: RouteErrorContentProps) {
  // Special handling for 404s
  if (isRouteErrorResponse(error) && error.status === 404) {
    return (
      <div className="flex min-h-full items-center justify-center p-8 pb-20">
        <SimpleErrorDisplay
          icon={FileQuestion}
          title="Page Not Found"
          description="The page you're looking for doesn't exist."
          scope={ErrorScope.Page}
          muted
        />
      </div>
    );
  }

  const { title, message, status } = extractErrorInfo(error);

  return (
    <div className="flex min-h-full items-center justify-center p-8 pb-20">
      <SimpleErrorDisplay
        icon={AlertTriangle}
        title={status ? `Error ${status}` : title}
        description={message}
        scope={ErrorScope.Page}
      />
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
