import { Check, Minus } from "lucide-react";
import { cn } from "~/utils/common";

type StepState = "completed" | "active" | "upcoming" | "skipped";

type StepTabProps = {
  index: number;
  label: string;
  state: StepState;
  disabled?: boolean;
  onClick: () => void;
};

export function StepTab({
  index,
  label,
  state,
  disabled,
  onClick,
}: StepTabProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex shrink-0 cursor-pointer items-center gap-1.5 rounded-full py-1 pr-2.5 pl-1 text-xs font-medium transition-all disabled:cursor-default disabled:opacity-50",
        state === "active" && "bg-purple-200/70 text-purple-800",
        state === "completed" && "text-green-700 hover:bg-green-50",
        state === "skipped" && "text-fg-muted hover:bg-bg-tertiary",
        state === "upcoming" && "text-fg-muted hover:bg-purple-100/50",
      )}
      aria-label={`Go to question ${index + 1}: ${label}`}
      aria-current={state === "active" ? "step" : undefined}
    >
      {state === "completed" ? (
        <span className="flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-green-100">
          <Check className="h-2.5 w-2.5" />
        </span>
      ) : state === "skipped" ? (
        <span className="flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-bg-tertiary">
          <Minus className="h-2.5 w-2.5" />
        </span>
      ) : (
        <span
          className={cn(
            "flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold",
            state === "active" && "bg-purple-600 text-white",
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
