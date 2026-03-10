import { type ReactNode, useState } from "react";
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
  EventPayloadAutoevalExampleLabeling,
  EventPayloadUserQuestion,
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

function LabelingExample({
  context,
  question,
  explanationQuestion,
  selectedValue,
  onSelect,
  explanationText,
  onExplanationChange,
  readOnly = false,
}: {
  context: AutoevalContentBlock[];
  question: Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>;
  explanationQuestion?: Extract<
    EventPayloadUserQuestion,
    { type: "free_response" }
  >;
  selectedValue: string | null;
  onSelect: (value: string) => void;
  explanationText: string;
  onExplanationChange: (text: string) => void;
  readOnly?: boolean;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-md border border-purple-300 bg-purple-50 p-4 dark:border-purple-700 dark:bg-purple-950/30">
      <span className="text-sm font-semibold">{question.header}</span>

      {/* Context blocks */}
      {context.map((block, i) => (
        <AutoevalContentBlockRenderer key={i} block={block} />
      ))}

      {/* Question */}
      <Markdown className="text-fg-primary text-sm font-medium">
        {question.question}
      </Markdown>

      {/* Options as compact radio-style buttons */}
      <div className="flex gap-2">
        {question.options.map((option) => {
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
      {explanationQuestion && (
        <div className="flex flex-col gap-1">
          <span className="text-fg-muted text-xs">
            {explanationQuestion.question}
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
function initFromAnswers(answers: Record<string, UserQuestionAnswer>) {
  const selections: Record<string, string> = {};
  const explanations: Record<string, string> = {};
  for (const [id, answer] of Object.entries(answers)) {
    if (answer.type === "multiple_choice" && answer.selected.length > 0) {
      selections[id] = answer.selected[0];
    } else if (answer.type === "free_response") {
      explanations[id] = answer.text;
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
  const examples = payload.examples.map((example) => {
    const labelQ = example.questions.find(
      (q) => q.type === "multiple_choice",
    ) as
      | Extract<EventPayloadUserQuestion, { type: "multiple_choice" }>
      | undefined;
    const explanationQ = example.questions.find(
      (q) => q.type === "free_response",
    ) as
      | Extract<EventPayloadUserQuestion, { type: "free_response" }>
      | undefined;
    return { context: example.context, labelQ, explanationQ };
  });

  const initial = answers ? initFromAnswers(answers) : undefined;
  const [selections, setSelections] = useState<Record<string, string>>(
    initial?.selections ?? {},
  );
  const [explanations, setExplanations] = useState<Record<string, string>>(
    initial?.explanations ?? {},
  );

  const handleSubmit = () => {
    if (!onSubmit) return;
    const responses: Record<string, UserQuestionAnswer> = {};
    for (const { labelQ, explanationQ } of examples) {
      if (labelQ) {
        const selectedId = selections[labelQ.id];
        if (selectedId) {
          responses[labelQ.id] = {
            type: "multiple_choice",
            selected: [selectedId],
          };
        } else {
          responses[labelQ.id] = { type: "skipped" };
        }
      }
      if (explanationQ) {
        const text = explanations[explanationQ.id];
        if (text) {
          responses[explanationQ.id] = { type: "free_response", text };
        } else {
          responses[explanationQ.id] = { type: "skipped" };
        }
      }
    }
    onSubmit(responses);
  };

  return (
    <div className="flex flex-col gap-4">
      {examples.map(
        ({ context, labelQ, explanationQ }) =>
          labelQ && (
            <LabelingExample
              key={labelQ.id}
              context={context}
              question={labelQ}
              explanationQuestion={explanationQ}
              readOnly={readOnly}
              selectedValue={selections[labelQ.id] ?? null}
              onSelect={(value) =>
                setSelections((prev) => ({ ...prev, [labelQ.id]: value }))
              }
              explanationText={
                explanationQ ? (explanations[explanationQ.id] ?? "") : ""
              }
              onExplanationChange={(text) => {
                if (explanationQ) {
                  setExplanations((prev) => ({
                    ...prev,
                    [explanationQ.id]: text,
                  }));
                }
              }}
            />
          ),
      )}

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
