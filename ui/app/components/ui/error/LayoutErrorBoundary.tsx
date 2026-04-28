import * as React from "react";
import { useRouteLoaderData } from "react-router";
import { ErrorDialog } from "./ErrorDialog";
import { ErrorContent, PageErrorContent } from "./ErrorContent";
import {
  isInfraError,
  classifyError,
  getErrorLabel,
} from "~/utils/tensorzero/errors";
import { logger } from "~/utils/logger";
import type { loader as rootLoader } from "~/root";

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
  const rootLoaderData = useRouteLoaderData<typeof rootLoader>("root");

  React.useEffect(() => {
    logger.error(error);
  }, [error]);

  // Infra errors -> dismissible dialog
  if (isInfraError(error)) {
    const classified = classifyError(error);
    // If the root layout already shows an infra dialog for the same error,
    // suppress this one to avoid duplicate stacked dialogs.
    if (rootLoaderData?.infraError?.type === classified.type) {
      return null;
    }
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
