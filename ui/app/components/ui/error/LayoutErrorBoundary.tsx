import * as React from "react";
import { isRouteErrorResponse } from "react-router";
import { ErrorDialog } from "./ErrorDialog";
import { ErrorContent } from "./ErrorContent";
import {
  ErrorScope,
  NotFoundDisplay,
  PageErrorContainer,
  PageErrorStack,
} from "./ErrorContentPrimitives";
import { AlertTriangle } from "lucide-react";
import {
  isInfraError,
  classifyError,
  getErrorLabel,
  getPageErrorInfo,
} from "~/utils/tensorzero/errors";

interface LayoutErrorBoundaryProps {
  error: unknown;
}

/**
 * Unified error display for layout ErrorBoundaries.
 * - Infra errors (gateway, auth, DB): Shows dismissible dialog
 * - Page errors (404, resource not found): Shows inline content
 */
export function LayoutErrorBoundary({ error }: LayoutErrorBoundaryProps) {
  const [dialogOpen, setDialogOpen] = React.useState(true);

  // Infra errors -> dismissible dialog
  if (isInfraError(error)) {
    const classified = classifyError(error);
    return (
      <ErrorDialog
        open={dialogOpen}
        onDismiss={() => setDialogOpen(false)}
        onReopen={() => setDialogOpen(true)}
        label={getErrorLabel(classified.type)}
      >
        <ErrorContent error={classified} />
      </ErrorDialog>
    );
  }

  // Page-scope: 404s get special muted treatment
  if (isRouteErrorResponse(error) && error.status === 404) {
    return <NotFoundDisplay />;
  }

  // Page-scope: other errors
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
