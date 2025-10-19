import { useReadOnly } from "~/context/read-only";

/**
 * Displays a banner when the UI is running in read-only mode.
 */
export function ReadOnlyBanner() {
  const { isReadOnly } = useReadOnly();
  if (!isReadOnly) return null;

  return (
    <div
      className="bg-amber-50 border-b border-amber-200 text-amber-900 text-sm px-4 py-2"
      role="status"
      aria-live="polite"
    >
      Read-only mode is enabled. Write actions (create, edit, delete, launch jobs) are disabled.
    </div>
  );
