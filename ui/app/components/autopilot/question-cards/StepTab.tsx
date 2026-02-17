import { Check } from "lucide-react";
import { cn } from "~/utils/common";

export type StepState = "completed" | "active" | "upcoming";

export function StepTab({
  index,
  label,
  state,
  onClick,
}: {
  index: number;
  label: string;
  state: StepState;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs font-medium transition-all",
        state === "active" &&
          "bg-purple-200/70 text-purple-800 dark:bg-purple-800/50 dark:text-purple-200",
        state === "completed" &&
          "cursor-pointer text-green-700 hover:bg-green-50 dark:text-green-400 dark:hover:bg-green-900/20",
        state === "upcoming" &&
          "text-fg-muted cursor-pointer hover:bg-purple-100/50 dark:hover:bg-purple-900/20",
      )}
      aria-label={`Go to question ${index + 1}: ${label}`}
      aria-current={state === "active" ? "step" : undefined}
    >
      {state === "completed" ? (
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/40">
          <Check className="h-3 w-3" />
        </span>
      ) : (
        <span
          className={cn(
            "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold",
            state === "active" &&
              "bg-purple-600 text-white dark:bg-purple-400 dark:text-purple-950",
            state === "upcoming" && "border-fg-muted border bg-transparent",
          )}
        >
          {index + 1}
        </span>
      )}
      <span className="truncate">{label}</span>
    </button>
  );
}

export function HorizontalStepTab({
  index,
  label,
  state,
  onClick,
  icon,
}: {
  index: number;
  label: string;
  state: StepState;
  onClick: () => void;
  icon?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-all",
        state === "active" &&
          "bg-purple-200/70 text-purple-800 dark:bg-purple-800/50 dark:text-purple-200",
        state === "completed" &&
          "cursor-pointer text-green-700 hover:bg-green-50 dark:text-green-400 dark:hover:bg-green-900/20",
        state === "upcoming" &&
          "text-fg-muted cursor-pointer hover:bg-purple-100/50 dark:hover:bg-purple-900/20",
      )}
      aria-label={`Go to question ${index + 1}: ${label}`}
      aria-current={state === "active" ? "step" : undefined}
    >
      {state === "completed" ? (
        <span className="flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/40">
          <Check className="h-2.5 w-2.5" />
        </span>
      ) : (
        <span
          className={cn(
            "flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold",
            state === "active" &&
              "bg-purple-600 text-white dark:bg-purple-400 dark:text-purple-950",
            state === "upcoming" && "border-fg-muted border bg-transparent",
          )}
        >
          {icon ?? index + 1}
        </span>
      )}
      <span className="truncate">{label}</span>
    </button>
  );
}
