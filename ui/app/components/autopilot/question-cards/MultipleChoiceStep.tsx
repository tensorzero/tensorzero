import { Markdown } from "~/components/ui/markdown";
import { Textarea } from "~/components/ui/textarea";
import { cn } from "~/utils/common";
import type { EventPayloadUserQuestion } from "~/types/tensorzero";

// Strip interactive elements inside buttons to avoid nested interactive HTML
const nonInteractiveComponents: React.ComponentProps<
  typeof Markdown
>["components"] = {
  a: ({ children }) => <span>{children}</span>,
};

function RadioIndicator({ selected }: { selected: boolean }) {
  return (
    <span
      className={cn(
        "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2 transition-colors",
        selected
          ? "border-purple-500 dark:border-purple-400"
          : "border-gray-300 dark:border-gray-600",
      )}
    >
      {selected && (
        <span className="h-2 w-2 rounded-full bg-purple-500 dark:bg-purple-400" />
      )}
    </span>
  );
}

function CheckboxIndicator({ selected }: { selected: boolean }) {
  return (
    <span
      className={cn(
        "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border-2 transition-colors",
        selected
          ? "border-purple-500 bg-purple-500 dark:border-purple-400 dark:bg-purple-400"
          : "border-gray-300 dark:border-gray-600",
      )}
    >
      {selected && (
        <svg
          className="h-3 w-3 text-white"
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M2.5 6l2.5 2.5 4.5-4.5" />
        </svg>
      )}
    </span>
  );
}

type MultipleChoiceStepProps = {
  question: Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>;
  selectedValues: Set<string>;
  onToggle: (value: string) => void;
  otherSelected: boolean;
  onOtherToggle: () => void;
  mcFreeText: string;
  onMcFreeTextChange: (text: string) => void;
};

export function MultipleChoiceStep({
  question,
  selectedValues,
  onToggle,
  otherSelected,
  onOtherToggle,
  mcFreeText,
  onMcFreeTextChange,
}: MultipleChoiceStepProps) {
  const Indicator = question.multi_select ? CheckboxIndicator : RadioIndicator;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-0.5">
        <Markdown className="text-fg-primary text-sm font-medium">
          {question.question}
        </Markdown>
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
                "group relative flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2 text-left transition-all",
                isSelected
                  ? "border-purple-500 bg-purple-50 ring-1 ring-purple-500 ring-inset dark:border-purple-400 dark:bg-purple-950/40 dark:ring-purple-400"
                  : "border-border bg-bg-secondary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
              )}
            >
              <Indicator selected={isSelected} />
              <div className="flex min-w-0 flex-col">
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
                <Markdown
                  className={cn(
                    "text-xs leading-snug",
                    isSelected
                      ? "text-purple-600 dark:text-purple-400"
                      : "text-fg-muted",
                  )}
                  components={nonInteractiveComponents}
                >
                  {option.description}
                </Markdown>
              </div>
            </button>
          );
        })}

        {!!question.include_free_response && (
          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={onOtherToggle}
              className={cn(
                "group relative flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2 text-left transition-all",
                otherSelected
                  ? "border-purple-500 bg-purple-50 ring-1 ring-purple-500 ring-inset dark:border-purple-400 dark:bg-purple-950/40 dark:ring-purple-400"
                  : "border-border bg-bg-secondary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
              )}
            >
              <Indicator selected={otherSelected} />
              <div className="flex min-w-0 flex-col">
                <span
                  className={cn(
                    "text-sm font-medium",
                    otherSelected
                      ? "text-purple-700 dark:text-purple-300"
                      : "text-fg-primary",
                  )}
                >
                  Other
                </span>
                <span
                  className={cn(
                    "text-xs leading-snug",
                    otherSelected
                      ? "text-purple-600 dark:text-purple-400"
                      : "text-fg-muted",
                  )}
                >
                  Provide your own answer
                </span>
              </div>
            </button>

            {otherSelected && (
              <Textarea
                value={mcFreeText}
                onChange={(e) => onMcFreeTextChange(e.target.value)}
                placeholder="Type your answer..."
                className="bg-bg-secondary min-h-[60px] resize-none text-sm"
                rows={2}
                autoFocus
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
