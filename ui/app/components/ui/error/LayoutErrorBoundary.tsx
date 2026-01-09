import * as React from "react";
import { ErrorDialog } from "./ErrorDialog";
import { ErrorContent, PageErrorContent } from "./ErrorContent";
import {
  isInfraError,
  classifyError,
  getErrorLabel,
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

  // Page errors -> inline display
  return <PageErrorContent error={error} />;
}
