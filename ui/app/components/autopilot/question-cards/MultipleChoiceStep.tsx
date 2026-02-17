import { cn } from "~/utils/common";
import type { EventPayloadUserQuestion } from "~/types/tensorzero";

export function MultipleChoiceStep({
  question,
  selectedValues,
  onToggle,
}: {
  question: Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>;
  selectedValues: Set<string>;
  onToggle: (value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-0.5">
        <span className="text-fg-primary text-sm font-medium">
          {question.question}
        </span>
        <span className="text-fg-muted text-xs">
          {question.multi_select ? "Select all that apply" : "Select one"}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {question.options.map((option) => {
          const isSelected = selectedValues.has(option.id);
          return (
            <button
              key={option.id}
              type="button"
              onClick={() => onToggle(option.id)}
              className={cn(
                "group relative flex flex-col items-start rounded-lg border px-3 py-2 text-left transition-all",
                isSelected
                  ? "border-purple-500 bg-purple-50 ring-1 ring-purple-500 dark:border-purple-400 dark:bg-purple-950/40 dark:ring-purple-400"
                  : "border-border bg-bg-secondary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
              )}
            >
              <span
                className={cn(
                  "text-sm font-medium",
                  isSelected
                    ? "text-purple-700 dark:text-purple-300"
                    : "text-fg-primary",
                )}
              >
                {option.label}
              </span>
              <span
                className={cn(
                  "text-xs leading-snug",
                  isSelected
                    ? "text-purple-600 dark:text-purple-400"
                    : "text-fg-muted",
                )}
              >
                {option.description}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
