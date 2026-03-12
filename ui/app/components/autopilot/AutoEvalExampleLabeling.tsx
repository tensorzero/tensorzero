import { useCallback, useState } from "react";
import { Check, ChevronLeft, ChevronRight, X } from "lucide-react";
import { Button } from "~/components/ui/button";
import { StepTab } from "~/components/autopilot/question-cards/StepTab";
import { CodeEditor, useFormattedJson } from "~/components/ui/code-editor";
import { ExpandableElement } from "~/components/input_output/ExpandableElement";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { cn } from "~/utils/common";
import { OptionButton } from "~/components/autopilot/question-cards/OptionButton";
import type {
  AutoEvalContentBlock,
  EventPayloadAutoEvalExampleLabeling,
  UserQuestionAnswer,
} from "~/types/tensorzero";

const MAX_TEXTAREA_ROWS = 3;

function JsonBlock({ data }: { data: unknown }) {
  const formatted = useFormattedJson(
    data as Parameters<typeof useFormattedJson>[0],
  );
  return (
    <CodeEditor
      value={formatted}
      readOnly
      maxHeight="none"
      allowedLanguages={["json"]}
      autoDetectLanguage={false}
    />
  );
}

const CONTEXT_MAX_HEIGHT = 200;

function ContextBlock({ block }: { block: AutoEvalContentBlock }) {
  const content = (() => {
    switch (block.type) {
      case "json":
        return <JsonBlock data={block.data} />;
      case "markdown":
        return (
          <CodeEditor
            value={block.text}
            readOnly
            showLineNumbers={false}
            maxHeight="none"
            allowedLanguages={["markdown"]}
            autoDetectLanguage={false}
          />
        );
      default: {
        const _exhaustiveCheck: never = block;
        return _exhaustiveCheck;
      }
    }
  })();

  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1">
      {block.label && (
        <span className="text-sm font-medium text-purple-500">
          {block.label}
        </span>
      )}
      <div className="border-l-2 border-purple-200 pl-2">
        <ExpandableElement maxHeight={CONTEXT_MAX_HEIGHT}>
          {content}
        </ExpandableElement>
      </div>
    </div>
  );
}

function ContextGrid({ blocks }: { blocks: AutoEvalContentBlock[] }) {
  if (blocks.length === 0) return null;
  const isSideBySide = blocks.length >= 2;
  return (
    <div
      className={cn(
        "grid grid-cols-1 gap-2 md:gap-4",
        isSideBySide && "md:grid-cols-2",
      )}
    >
      {blocks.map((block, i) => (
        <ContextBlock key={i} block={block} />
      ))}
    </div>
  );
}

type AutoEvalExampleLabelingCardProps = {
  payload: EventPayloadAutoEvalExampleLabeling;
  onSubmit: (responses: Record<string, UserQuestionAnswer>) => void;
  isLoading?: boolean;
};

export function AutoEvalExampleLabelingCard({
  payload,
  onSubmit,
  isLoading = false,
}: AutoEvalExampleLabelingCardProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [selections, setSelections] = useState<Record<number, string>>({});
  const [rationales, setRationales] = useState<Record<number, string>>({});

  const autoResize = useCallback((el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 20;
    const padding =
      parseFloat(getComputedStyle(el).paddingTop) +
      parseFloat(getComputedStyle(el).paddingBottom);
    const maxHeight = lineHeight * MAX_TEXTAREA_ROWS + padding;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
  }, []);

  const autoResizeRef = useCallback(
    (el: HTMLTextAreaElement | null) => {
      if (el) autoResize(el);
    },
    [autoResize],
  );

  const totalExamples = payload.examples.length;

  if (totalExamples === 0) return null;

  const isSingleExample = totalExamples === 1;
  const isFirst = activeIndex === 0;
  const isLast = activeIndex === totalExamples - 1;
  const example = payload.examples[activeIndex];

  const isStepAnswered = (idx: number) => Boolean(selections[idx]);
  const answeredCount = payload.examples.filter((_, i) =>
    isStepAnswered(i),
  ).length;
  const allAnswered = answeredCount === totalExamples;

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
    <div className="mt-4 flex flex-col rounded-md border border-purple-200 bg-white dark:border-purple-800 dark:bg-purple-950/10">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 px-4 pt-3 pb-3">
        <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
          Label examples to improve evaluator accuracy
        </span>
        <button
          type="button"
          onClick={handleDismiss}
          disabled={isLoading}
          className="-mr-1 cursor-pointer rounded-sm p-0.5 text-purple-300 transition-colors hover:text-purple-500 disabled:cursor-not-allowed disabled:opacity-50 dark:text-purple-700 dark:hover:text-purple-500"
          aria-label="Dismiss examples"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Step tabs */}
      {!isSingleExample && (
        <nav
          aria-label="Example steps"
          className="flex gap-1 overflow-x-auto px-3 pb-3"
        >
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
      <div className="flex flex-col gap-4 px-4">
        <ContextGrid blocks={example.context} />

        <div className="flex flex-col gap-1.5">
          <span className="text-fg-primary text-sm font-medium">
            {example.label_question.question}
          </span>
          <div className="flex gap-2">
            {example.label_question.options.map((opt) => {
              const isSelected = selections[activeIndex] === opt.id;
              const button = (
                <OptionButton
                  isSelected={isSelected}
                  disabled={isLoading}
                  onClick={() =>
                    setSelections((prev) => ({
                      ...prev,
                      [activeIndex]: opt.id,
                    }))
                  }
                  className={cn(
                    "flex-1 text-center text-sm font-medium",
                    isSelected
                      ? "text-purple-700 dark:text-purple-300"
                      : "text-fg-primary",
                  )}
                >
                  {opt.label}
                </OptionButton>
              );

              if (opt.description) {
                return (
                  <Tooltip key={opt.id}>
                    <TooltipTrigger asChild className="flex-1">
                      {button}
                    </TooltipTrigger>
                    <TooltipContent
                      className="border-border bg-fg-primary text-bg-primary border text-xs shadow-lg"
                      sideOffset={5}
                    >
                      {opt.description}
                    </TooltipContent>
                  </Tooltip>
                );
              }

              return (
                <div key={opt.id} className="flex flex-1">
                  {button}
                </div>
              );
            })}
          </div>
        </div>

        {example.explanation_question && (
          <div className="flex flex-col gap-1.5">
            <span className="text-fg-primary text-sm font-medium">
              {example.explanation_question.question}{" "}
              <span className="text-fg-tertiary text-xs">(optional)</span>
            </span>
            <textarea
              ref={autoResizeRef}
              value={rationales[activeIndex] ?? ""}
              disabled={isLoading}
              onChange={(e) => {
                setRationales((prev) => ({
                  ...prev,
                  [activeIndex]: e.target.value,
                }));
                autoResize(e.target);
              }}
              aria-label={example.explanation_question.question}
              className="border-input focus-visible:border-border-accent bg-bg-primary w-full resize-none overscroll-none rounded-md border px-3 py-2 text-sm transition-colors focus-visible:ring-0 focus-visible:outline-hidden disabled:cursor-not-allowed disabled:opacity-50"
              rows={1}
            />
          </div>
        )}
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
              className="gap-0.5 pl-1 text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          )}
        </div>
        <div className="flex items-center gap-4">
          {!isSingleExample && (
            <span className="text-purple-600 dark:text-purple-400 text-xs">
              {answeredCount}/{totalExamples} labeled
            </span>
          )}
          {isSingleExample || isLast ? (
            <Button
              size="xs"
              disabled={!allAnswered || isLoading}
              onClick={handleSubmit}
              className="gap-1"
            >
              {isLoading ? "Submitting..." : "Submit"}
              <Check className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Button
              size="xs"
              disabled={!isStepAnswered(activeIndex) || isLoading}
              onClick={() => setActiveIndex((s) => s + 1)}
              className="gap-0.5 pr-1 bg-purple-600 text-white hover:bg-purple-700 dark:bg-purple-500 dark:hover:bg-purple-600"
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
