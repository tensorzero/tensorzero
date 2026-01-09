import { AlertTriangle } from "lucide-react";
import {
  ErrorScope,
  NotFoundDisplay,
  PageErrorContainer,
  PageErrorStack,
} from "./ErrorContentPrimitives";
import { isRouteErrorResponse } from "react-router";
import { getPageErrorInfo } from "~/utils/tensorzero/errors";

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

  const { title, message, status } = getPageErrorInfo(error);

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
