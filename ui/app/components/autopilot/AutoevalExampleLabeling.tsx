import { type ReactNode, useEffect, useState } from "react";
import { Markdown, ReadOnlyCodeBlock } from "~/components/ui/markdown";
import { Textarea } from "~/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import type {
  AutoevalContentBlock,
  AutoevalLabelingExample,
  EventPayloadAutoevalExampleLabeling,
  UserQuestionAnswer,
} from "~/types/tensorzero";

function CollapsibleWrapper({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <details className="group rounded-md border border-purple-200 dark:border-purple-800">
      <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium">
        {label}
      </summary>
      <div
        className="border-t border-purple-200 px-3 py-2 dark:border-purple-800"
        style={{ maxHeight: "300px", overflowY: "auto" }}
      >
        {children}
      </div>
    </details>
  );
}

export function AutoevalContentBlockRenderer({
  block,
}: {
  block: AutoevalContentBlock;
}) {
  const { label } = block;
  let content: ReactNode;
  switch (block.type) {
    case "markdown":
      content = <Markdown className="text-sm">{block.text}</Markdown>;
      break;
    case "json":
      content = (
        <ReadOnlyCodeBlock
          code={JSON.stringify(block.data, null, 2)}
          language="json"
          maxHeight="150px"
        />
      );
      break;
  }

  if (label) {
    return <CollapsibleWrapper label={label}>{content}</CollapsibleWrapper>;
  }
  return content;
}

function LabelingExampleCard({
  example,
  selectedValue,
  onSelect,
  explanationText,
  onExplanationChange,
  readOnly = false,
}: {
  example: AutoevalLabelingExample;
  selectedValue: string | null;
  onSelect: (value: string) => void;
  explanationText: string;
  onExplanationChange: (text: string) => void;
  readOnly?: boolean;
}) {
  const { context, label_question, explanation_question } = example;

  return (
    <div className="flex flex-col gap-3 rounded-md border border-purple-300 bg-purple-50 p-4 dark:border-purple-700 dark:bg-purple-950/30">
      <span className="text-sm font-semibold">{label_question.header}</span>

      {/* Context blocks */}
      {context.map((block, i) => (
        <AutoevalContentBlockRenderer key={i} block={block} />
      ))}

      {/* Question */}
      <Markdown className="text-fg-primary text-sm font-medium">
        {label_question.question}
      </Markdown>

      {/* Options as compact radio-style buttons */}
      <div className="flex gap-2">
        {label_question.options.map((option) => {
          const isSelected = selectedValue === option.id;
          return (
            <Tooltip key={option.id}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  disabled={readOnly}
                  onClick={() => onSelect(option.id)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-all",
                    isSelected
                      ? "border-purple-500 bg-purple-100 text-purple-700 ring-1 ring-purple-500 ring-inset dark:border-purple-400 dark:bg-purple-950/40 dark:text-purple-300 dark:ring-purple-400"
                      : "border-border bg-bg-secondary text-fg-primary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
                    readOnly && "cursor-default opacity-75",
                  )}
                >
                  {option.label}
                </button>
              </TooltipTrigger>
              <TooltipContent>{option.description}</TooltipContent>
            </Tooltip>
          );
        })}
      </div>

      {/* Optional explanation */}
      {explanation_question && (
        <div className="flex flex-col gap-1">
          <span className="text-fg-muted text-xs">
            {explanation_question.question}
          </span>
          <Textarea
            value={explanationText}
            disabled={readOnly}
            onChange={(e) => onExplanationChange(e.target.value)}
            placeholder="Type your explanation..."
            className="bg-bg-secondary min-h-[60px] resize-none text-sm"
            rows={2}
          />
        </div>
      )}
    </div>
  );
}

/** Extract the initial selection/explanation state from saved answers. */
function initFromAnswers(
  examples: AutoevalLabelingExample[],
  answers: Record<string, UserQuestionAnswer>,
) {
  const selections: Record<string, string> = {};
  const explanations: Record<string, string> = {};
  for (const example of examples) {
    const labelAnswer = answers[example.label_question.id];
    if (
      labelAnswer?.type === "multiple_choice" &&
      labelAnswer.selected.length > 0
    ) {
      selections[example.label_question.id] = labelAnswer.selected[0];
    }
    if (example.explanation_question) {
      const expAnswer = answers[example.explanation_question.id];
      if (expAnswer?.type === "free_response") {
        explanations[example.explanation_question.id] = expAnswer.text;
      }
    }
  }
  return { selections, explanations };
}

type AutoevalExampleLabelingCardProps = {
  payload: EventPayloadAutoevalExampleLabeling;
  onSubmit?: (responses: Record<string, UserQuestionAnswer>) => void;
  isLoading?: boolean;
  /** Pre-populated answers (e.g. from a completed `user_questions_answers` event). */
  answers?: Record<string, UserQuestionAnswer>;
  /** When true, options and textarea are non-interactive and submit is hidden. */
  readOnly?: boolean;
};

export function AutoevalExampleLabelingCard({
  payload,
  onSubmit,
  isLoading = false,
  answers,
  readOnly = false,
}: AutoevalExampleLabelingCardProps) {
  const initial = answers
    ? initFromAnswers(payload.examples, answers)
    : undefined;
  const [selections, setSelections] = useState<Record<string, string>>(
    initial?.selections ?? {},
  );
  const [explanations, setExplanations] = useState<Record<string, string>>(
    initial?.explanations ?? {},
  );

  // Sync state when answers arrive after initial mount (e.g. SSE delivers
  // user_questions_answers while the card is already expanded).
  useEffect(() => {
    if (!answers) return;
    const updated = initFromAnswers(payload.examples, answers);
    setSelections(updated.selections);
    setExplanations(updated.explanations);
  }, [answers, payload.examples]);

  const handleSubmit = () => {
    if (!onSubmit) return;
    const responses: Record<string, UserQuestionAnswer> = {};
    for (const example of payload.examples) {
      const selectedId = selections[example.label_question.id];
      if (selectedId) {
        responses[example.label_question.id] = {
          type: "multiple_choice",
          selected: [selectedId],
        };
      } else {
        responses[example.label_question.id] = { type: "skipped" };
      }
      if (example.explanation_question) {
        const text = explanations[example.explanation_question.id];
        if (text) {
          responses[example.explanation_question.id] = {
            type: "free_response",
            text,
          };
        } else {
          responses[example.explanation_question.id] = { type: "skipped" };
        }
      }
    }
    onSubmit(responses);
  };

  return (
    <div className="flex flex-col gap-4">
      {payload.examples.map((example) => (
        <LabelingExampleCard
          key={example.label_question.id}
          example={example}
          readOnly={readOnly}
          selectedValue={selections[example.label_question.id] ?? null}
          onSelect={(value) =>
            setSelections((prev) => ({
              ...prev,
              [example.label_question.id]: value,
            }))
          }
          explanationText={
            example.explanation_question
              ? (explanations[example.explanation_question.id] ?? "")
              : ""
          }
          onExplanationChange={(text) => {
            if (example.explanation_question) {
              setExplanations((prev) => ({
                ...prev,
                [example.explanation_question!.id]: text,
              }));
            }
          }}
        />
      ))}

      {!readOnly && (
        <button
          type="button"
          disabled={isLoading}
          onClick={handleSubmit}
          className="self-end rounded-md bg-purple-600 px-4 py-2 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50"
        >
          {isLoading ? "Submitting..." : "Submit All"}
        </button>
      )}
    </div>
  );
}
