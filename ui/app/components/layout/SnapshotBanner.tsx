import { History } from "lucide-react";
import { useSnapshotHash } from "~/context/snapshot";

export function SnapshotBanner() {
  const snapshotHash = useSnapshotHash();
  if (!snapshotHash) return null;

  return (
    <div className="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-4 py-2 text-sm text-blue-800 dark:border-blue-700 dark:bg-blue-950/30 dark:text-blue-200">
      <History className="h-4 w-4 flex-shrink-0" />
      <span>Viewing historical configuration</span>
    </div>
  );
}
