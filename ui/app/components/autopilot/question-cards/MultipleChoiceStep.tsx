import { Markdown } from "~/components/ui/markdown";
import { cn } from "~/utils/common";
import type { EventPayloadUserQuestion } from "~/types/tensorzero";
import { OptionButton } from "./OptionButton";

// Strip interactive elements inside buttons to avoid nested interactive HTML
const nonInteractiveComponents: React.ComponentProps<
  typeof Markdown
>["components"] = {
  a: ({ children }) => <span>{children}</span>,
};

type MultipleChoiceStepProps = {
  question: Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>;
  selectedValues: Set<string>;
  onToggle: (value: string) => void;
};

export function MultipleChoiceStep({
  question,
  selectedValues,
  onToggle,
}: MultipleChoiceStepProps) {
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
            <OptionButton
              key={option.id}
              isSelected={isSelected}
              onClick={() => onToggle(option.id)}
              className="group relative flex flex-col items-start"
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
            </OptionButton>
          );
        })}
      </div>
    </div>
  );
}
