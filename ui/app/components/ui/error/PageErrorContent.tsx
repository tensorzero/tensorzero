import { AlertTriangle } from "lucide-react";
import {
  ErrorScope,
  extractPageErrorInfo,
  NotFoundDisplay,
  PageErrorContainer,
  PageErrorStack,
} from "./ErrorContentPrimitives";
import { isRouteErrorResponse } from "react-router";

interface PageErrorContentProps {
  error: unknown;
}

/**
 * Renders page-scope errors inline.
 * For layout ErrorBoundaries, use LayoutErrorBoundary instead - it handles
 * both infra errors (dialog) and page errors (inline) automatically.
 */
export function PageErrorContent({ error }: PageErrorContentProps) {
  // Special handling for 404s
  if (isRouteErrorResponse(error) && error.status === 404) {
    return <NotFoundDisplay />;
  }

  const { title, message, status } = extractPageErrorInfo(error);

  return (
    <PageErrorContainer>
      <PageErrorStack
        icon={AlertTriangle}
        title={status ? `Error ${status}` : title}
        description={message}
        scope={ErrorScope.Page}
      />
    </PageErrorContainer>
  );
}
