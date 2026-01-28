import { Loader2 } from "lucide-react";
import { cn } from "~/utils/common";

type YoloModeIndicatorProps = {
  isEnabled: boolean;
  onToggle: (enabled: boolean) => void;
  currentToolName: string | null;
};

/**
 * A minimal indicator card shown when yolo mode is enabled and there are no pending tool calls.
 * Shows the current tool being auto-approved or just "Yolo mode enabled".
 */
export function YoloModeIndicator({
  isEnabled,
  onToggle,
  currentToolName,
}: YoloModeIndicatorProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-md border px-4 py-2",
        "border-amber-300 bg-amber-50 dark:border-amber-700 dark:bg-amber-950/30",
      )}
    >
      <div className="flex items-center gap-2">
        {currentToolName ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin text-amber-600 dark:text-amber-400" />
            <span className="text-sm text-amber-800 dark:text-amber-200">
              Auto-approving:{" "}
              <code className="rounded bg-amber-200 px-1 py-0.5 font-mono text-xs dark:bg-amber-800">
                {currentToolName}
              </code>
            </span>
          </>
        ) : (
          <span className="text-sm text-amber-700 dark:text-amber-300">
            Yolo mode enabled - auto-approving all tool calls
          </span>
        )}
      </div>
      <YoloToggle isEnabled={isEnabled} onToggle={onToggle} />
    </div>
  );
}

type YoloToggleProps = {
  isEnabled: boolean;
  onToggle: (enabled: boolean) => void;
  className?: string;
};

/**
 * A toggle switch for yolo mode with keyboard shortcut hint.
 */
export function YoloToggle({
  isEnabled,
  onToggle,
  className,
}: YoloToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={isEnabled}
      aria-label="Toggle yolo mode"
      title="Toggle yolo mode (Cmd/Ctrl+Shift+Y)"
      onClick={() => onToggle(!isEnabled)}
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors",
        isEnabled
          ? "bg-amber-600 text-white hover:bg-amber-700 dark:bg-amber-500 dark:hover:bg-amber-600"
          : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700",
        className,
      )}
    >
      <span>Yolo</span>
      <span
        className={cn(
          "inline-block h-3 w-3 rounded-full transition-colors",
          isEnabled
            ? "bg-white"
            : "border border-gray-400 bg-transparent dark:border-gray-500",
        )}
      />
    </button>
  );
}
