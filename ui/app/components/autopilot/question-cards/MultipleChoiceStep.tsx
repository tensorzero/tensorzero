import { MessageSquareMore } from "lucide-react";
import { Textarea } from "~/components/ui/textarea";
import { cn } from "~/utils/common";
import type { EventPayloadUserQuestion } from "~/types/tensorzero";
import { InlineMarkdown } from "./InlineMarkdown";

export function MultipleChoiceStep({
  question,
  selectedValues,
  onToggle,
  otherText,
  onOtherTextChange,
}: {
  question: Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>;
  selectedValues: Set<string>;
  onToggle: (value: string) => void;
  otherText: string;
  onOtherTextChange: (text: string) => void;
}) {
  const isOtherSelected = selectedValues.has("__other__");

  return (
    <div className="flex flex-col gap-3">
      <span className="text-fg-primary text-sm font-medium">
        <InlineMarkdown text={question.question} />
      </span>

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
                <InlineMarkdown text={option.label} />
              </span>
              <span
                className={cn(
                  "text-xs leading-snug",
                  isSelected
                    ? "text-purple-600 dark:text-purple-400"
                    : "text-fg-muted",
                )}
              >
                <InlineMarkdown text={option.description} />
              </span>
            </button>
          );
        })}

        <button
          type="button"
          onClick={() => onToggle("__other__")}
          className={cn(
            "group relative flex flex-col items-start rounded-lg border px-3 py-2 text-left transition-all",
            isOtherSelected
              ? "border-purple-500 bg-purple-50 ring-1 ring-purple-500 dark:border-purple-400 dark:bg-purple-950/40 dark:ring-purple-400"
              : "border-border bg-bg-secondary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
          )}
        >
          <span
            className={cn(
              "inline-flex items-center gap-1 text-sm font-medium",
              isOtherSelected
                ? "text-purple-700 dark:text-purple-300"
                : "text-fg-primary",
            )}
          >
            <MessageSquareMore className="h-3.5 w-3.5" />
            Other
          </span>
          <span
            className={cn(
              "text-xs leading-snug",
              isOtherSelected
                ? "text-purple-600 dark:text-purple-400"
                : "text-fg-muted",
            )}
          >
            Provide custom input
          </span>
        </button>
      </div>

      {isOtherSelected && (
        <Textarea
          value={otherText}
          onChange={(e) => onOtherTextChange(e.target.value)}
          placeholder="Type your response..."
          className="bg-bg-secondary min-h-[36px] resize-none text-sm"
          rows={1}
          autoFocus
        />
      )}
    </div>
  );
}
