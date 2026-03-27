import type { ReactNode } from "react";
import { X } from "lucide-react";
import { cn } from "~/utils/common";
import { StepTab } from "./StepTab";
import { useAnimatedHeight } from "~/hooks/useAnimatedHeight";

type StepTabState = "active" | "completed" | "skipped" | "upcoming";

type QuestionCardProps = {
  title: string;
  onDismiss: () => void;
  isLoading: boolean;
  children: ReactNode;
  className?: string;
  contentClassName?: string;
  dismissAriaLabel?: string;
  activeStep: number;
  steps?: {
    items: { id: string; label: string; state: StepTabState }[];
    onStepClick: (index: number) => void;
  };
  footer: ReactNode;
};

export function QuestionCard({
  title,
  onDismiss,
  isLoading,
  children,
  className,
  contentClassName,
  dismissAriaLabel = "Dismiss",
  activeStep,
  steps,
  footer,
}: QuestionCardProps) {
  const {
    ref: contentRef,
    height: contentHeight,
    onTransitionEnd,
  } = useAnimatedHeight(activeStep);

  const isAnimating = typeof contentHeight === "number";

  return (
    <div
      className={cn(
        "flex flex-col rounded-md border border-purple-200 bg-white dark:border-purple-800 dark:bg-purple-950/10",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-4 px-4 pt-3 pb-3">
        <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
          {title}
        </span>
        <button
          type="button"
          onClick={onDismiss}
          disabled={isLoading}
          className="-mr-1 cursor-pointer rounded-sm p-0.5 text-purple-300 transition-colors hover:text-purple-500 disabled:cursor-not-allowed disabled:opacity-50 dark:text-purple-700 dark:hover:text-purple-500"
          aria-label={dismissAriaLabel}
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {steps && steps.items.length > 1 && (
        <nav
          aria-label="Steps"
          className="flex gap-1 overflow-x-auto px-3 pb-3"
        >
          {steps.items.map((step, idx) => (
            <StepTab
              key={step.id}
              index={idx}
              label={step.label}
              state={step.state}
              disabled={isLoading}
              onClick={() => steps.onStepClick(idx)}
            />
          ))}
        </nav>
      )}

      <div className="px-4">
        <div
          ref={contentRef}
          className={cn(
            "transition-[height] duration-300 ease-in-out",
            contentClassName,
            isAnimating && "overflow-hidden",
          )}
          style={{ height: isAnimating ? contentHeight : "auto" }}
          onTransitionEnd={(e) => {
            if (e.propertyName === "height" && e.target === e.currentTarget) {
              onTransitionEnd();
            }
          }}
        >
          <div key={activeStep} className="animate-in fade-in duration-300">
            {children}
          </div>
        </div>
      </div>

      {footer}
    </div>
  );
}
