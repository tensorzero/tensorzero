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
                  onClick={() => onSelect(option.id)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-all",
                    isSelected
                      ? "border-purple-500 bg-purple-100 text-purple-700 ring-1 ring-purple-500 ring-inset dark:border-purple-400 dark:bg-purple-950/40 dark:text-purple-300 dark:ring-purple-400"
                      : "border-border bg-bg-secondary text-fg-primary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
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

export function AutoevalExampleLabelingCard({
  payload,
}: {
  payload: EventPayloadAutoevalExampleLabeling;
}) {
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

  const [selections, setSelections] = useState<Record<string, string>>({});
  const [explanations, setExplanations] = useState<Record<string, string>>({});

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
              selectedValue={selections[labelQ.id] ?? null}
              onSelect={(value) =>
                setSelections((prev) => ({ ...prev, [labelQ.id]: value }))
              }
              explanationText={explanations[labelQ.id] ?? ""}
              onExplanationChange={(text) =>
                setExplanations((prev) => ({ ...prev, [labelQ.id]: text }))
              }
            />
          ),
      )}

      <button
        type="button"
        className="self-end rounded-md bg-purple-600 px-4 py-2 text-sm font-medium text-white hover:bg-purple-700"
      >
        Submit All
      </button>
    </div>
  );
}
