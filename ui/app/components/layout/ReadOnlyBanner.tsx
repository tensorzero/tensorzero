/**
 * Read-Only Mode Banner
 *
 * Displays a prominent banner at the top of the application when in read-only mode.
 * Informs users that all write operations are disabled.
 */

import { AlertCircle } from "lucide-react";
import { useReadOnly } from "~/context/read-only";

export function ReadOnlyBanner() {
  const { isReadOnly } = useReadOnly();

  if (!isReadOnly) {
    return null;
  }

  return (
    <div className="flex items-center gap-2 border-b border-yellow-600 bg-yellow-50 px-4 py-2 text-sm text-yellow-900">
      <AlertCircle className="h-4 w-4 flex-shrink-0" />
      <p>
        <strong>Read-only mode is active.</strong> All write operations
        (inference, evaluations, dataset modifications, fine-tuning) are
        disabled.
      </p>
    </div>
  );
}
