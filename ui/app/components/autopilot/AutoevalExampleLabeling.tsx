import { useState } from "react";
import { Check, ChevronLeft, ChevronRight, X } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";
import { StepTab } from "~/components/autopilot/question-cards/StepTab";
import { InputElement } from "~/components/input_output/InputElement";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import type {
  AutoEvalContentBlock,
  EventPayloadAutoEvalExampleLabeling,
  UserQuestionAnswer,
  Input,
  ContentBlockChatOutput,
} from "~/types/tensorzero";

const CONTEXT_MAX_HEIGHT = 200;

function asInferenceInput(data: unknown): Input | undefined {
  if (
    typeof data === "object" &&
    data !== null &&
    "messages" in data &&
    Array.isArray((data as Record<string, unknown>).messages)
  ) {
    return data as Input;
  }
  return undefined;
}

function asChatOutput(data: unknown): ContentBlockChatOutput[] | undefined {
  if (Array.isArray(data)) {
    const first = data[0];
    if (
      first &&
      typeof first === "object" &&
      "type" in first &&
      typeof first.type === "string"
    ) {
      return data as ContentBlockChatOutput[];
    }
  }
  return undefined;
}

function ContextBlock({ block }: { block: AutoEvalContentBlock }) {
  switch (block.type) {
    case "json": {
      const inferenceInput = asInferenceInput(block.data);
      if (inferenceInput) {
        return (
          <div className="flex min-w-0 flex-col gap-1">
            {block.label && (
              <span className="text-fg-tertiary text-xs font-medium">
                {block.label}
              </span>
            )}
            <InputElement
              input={inferenceInput}
              maxHeight={CONTEXT_MAX_HEIGHT}
            />
          </div>
        );
      }

      const chatOutput = asChatOutput(block.data);
      if (chatOutput) {
        return (
          <div className="flex min-w-0 flex-col gap-1">
            {block.label && (
              <span className="text-fg-tertiary text-xs font-medium">
                {block.label}
              </span>
            )}
            <ChatOutputElement
              output={chatOutput}
              maxHeight={CONTEXT_MAX_HEIGHT}
            />
          </div>
        );
      }

      return (
        <div className="flex min-w-0 flex-col gap-1">
          {block.label && (
            <span className="text-fg-tertiary text-xs font-medium">
              {block.label}
            </span>
          )}
          <pre className="bg-bg-primary border-border overflow-auto rounded-lg border p-4 text-xs">
            {JSON.stringify(block.data, null, 2)}
          </pre>
        </div>
      );
    }
    case "markdown": {
      return (
        <div className="flex min-w-0 flex-col gap-1">
          {block.label && (
            <span className="text-fg-tertiary text-xs font-medium">
              {block.label}
            </span>
          )}
          <div className="bg-bg-primary border-border overflow-auto rounded-lg border p-4 text-sm whitespace-pre-wrap">
            {block.text}
          </div>
        </div>
      );
    }
  }
}

type AutoevalExampleLabelingCardProps = {
  payload: EventPayloadAutoEvalExampleLabeling;
  onSubmit: (responses: Record<string, UserQuestionAnswer>) => void;
  isLoading?: boolean;
};

export function AutoevalExampleLabelingCard({
  payload,
  onSubmit,
  isLoading = false,
}: AutoevalExampleLabelingCardProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [selections, setSelections] = useState<Record<number, string>>({});
  const [rationales, setRationales] = useState<Record<number, string>>({});

  const totalExamples = payload.examples.length;
  if (totalExamples === 0) return null;

  const isSingleExample = totalExamples === 1;
  const isFirst = activeIndex === 0;
  const isLast = activeIndex === totalExamples - 1;
  const example = payload.examples[activeIndex];

  const isStepAnswered = (idx: number) => Boolean(selections[idx]);
  const isCurrentComplete = isStepAnswered(activeIndex);

  const getStepTabState = (idx: number) => {
    if (idx === activeIndex) return "active" as const;
    if (isStepAnswered(idx)) return "completed" as const;
    return "upcoming" as const;
  };

  const handleDismiss = () => {
    const responses: Record<string, UserQuestionAnswer> = {};
    for (const ex of payload.examples) {
      responses[ex.label_question.id] = { type: "skipped" };
      if (ex.explanation_question) {
        responses[ex.explanation_question.id] = { type: "skipped" };
      }
    }
    onSubmit(responses);
  };

  const handleSubmit = () => {
    const responses: Record<string, UserQuestionAnswer> = {};
    for (let i = 0; i < payload.examples.length; i++) {
      const ex = payload.examples[i];
      const sel = selections[i];
      if (sel) {
        responses[ex.label_question.id] = {
          type: "multiple_choice",
          selected: [sel],
        };
      } else {
        responses[ex.label_question.id] = { type: "skipped" };
      }
      if (ex.explanation_question) {
        const text = rationales[i];
        if (text) {
          responses[ex.explanation_question.id] = {
            type: "free_response",
            text,
          };
        } else {
          responses[ex.explanation_question.id] = { type: "skipped" };
        }
      }
    }
    onSubmit(responses);
  };

  return (
    <div className="flex flex-col rounded-md border border-purple-300 bg-purple-50 dark:border-purple-700 dark:bg-purple-950/30">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 px-4 pt-3 pb-3">
        <span className="text-sm font-medium">
          Provide examples to improve evaluator accuracy
        </span>
        <button
          type="button"
          onClick={handleDismiss}
          disabled={isLoading}
          className="-mr-1 cursor-pointer rounded-sm p-0.5 text-purple-400 transition-colors hover:text-purple-600 disabled:cursor-not-allowed disabled:opacity-50 dark:text-purple-500 dark:hover:text-purple-300"
          aria-label="Dismiss examples"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Step tabs */}
      {!isSingleExample && (
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3">
          {payload.examples.map((ex, idx) => (
            <StepTab
              key={ex.label_question.id}
              index={idx}
              label={ex.label_question.header}
              state={getStepTabState(idx)}
              disabled={isLoading}
              onClick={() => setActiveIndex(idx)}
            />
          ))}
        </nav>
      )}

      {/* Content */}
      <div className="px-4">
        <div
          key={activeIndex}
          className="animate-in fade-in flex flex-col gap-3 duration-200"
        >
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2 md:gap-4">
            {example.context.map((block, i) => (
              <ContextBlock key={i} block={block} />
            ))}
          </div>

          <div className="flex flex-col gap-1.5">
            <span className="text-fg-secondary text-sm">
              {example.label_question.question}
            </span>
            <div className="flex flex-wrap gap-2">
              {example.label_question.options.map((opt) => {
                const isSelected = selections[activeIndex] === opt.id;
                const button = (
                  <button
                    key={opt.id}
                    type="button"
                    disabled={isLoading}
                    onClick={() =>
                      setSelections((prev) => ({
                        ...prev,
                        [activeIndex]: opt.id,
                      }))
                    }
                    className={cn(
                      "cursor-pointer rounded-lg border px-3 py-2 text-left text-sm font-medium transition-all disabled:cursor-not-allowed disabled:opacity-50",
                      isSelected
                        ? "border-purple-500 bg-purple-50 ring-1 ring-purple-500 ring-inset dark:border-purple-400 dark:bg-purple-950/40 dark:ring-purple-400"
                        : "border-border bg-bg-secondary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
                      isSelected
                        ? "text-purple-700 dark:text-purple-300"
                        : "text-fg-primary",
                    )}
                  >
                    {opt.label}
                  </button>
                );

                if (opt.description) {
                  return (
                    <Tooltip key={opt.id}>
                      <TooltipTrigger asChild>{button}</TooltipTrigger>
                      <TooltipContent
                        className="border-border bg-fg-primary text-bg-primary border text-xs shadow-lg"
                        sideOffset={5}
                      >
                        {opt.description}
                      </TooltipContent>
                    </Tooltip>
                  );
                }

                return button;
              })}
            </div>
          </div>

          {example.explanation_question && (
            <div className="flex flex-col gap-1.5">
              <span className="text-fg-secondary text-sm">
                {example.explanation_question.question}
              </span>
              <Textarea
                value={rationales[activeIndex] ?? ""}
                disabled={isLoading}
                onChange={(e) =>
                  setRationales((prev) => ({
                    ...prev,
                    [activeIndex]: e.target.value,
                  }))
                }
                placeholder="Rationale (optional)..."
                className="bg-bg-secondary min-h-[60px] resize-none text-sm"
                rows={2}
              />
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-4 pt-3 pb-3">
        <div>
          {!isSingleExample && !isFirst && (
            <Button
              variant="ghost"
              size="xs"
              disabled={isLoading}
              onClick={() => setActiveIndex((s) => s - 1)}
              className="gap-0.5 pl-1 text-purple-700 hover:text-purple-800 dark:text-purple-300 dark:hover:text-purple-200"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isSingleExample || isLast ? (
            <Button
              size="xs"
              disabled={!isCurrentComplete || isLoading}
              onClick={handleSubmit}
              className="gap-1 bg-purple-600 text-white hover:bg-purple-700 dark:bg-purple-600 dark:hover:bg-purple-500"
            >
              {isLoading ? "Submitting..." : "Submit"}
              <Check className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Button
              size="xs"
              disabled={!isCurrentComplete || isLoading}
              onClick={() => setActiveIndex((s) => s + 1)}
              className="gap-0.5 bg-purple-600 pr-1 text-white hover:bg-purple-700 dark:bg-purple-600 dark:hover:bg-purple-500"
            >
              Next
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
