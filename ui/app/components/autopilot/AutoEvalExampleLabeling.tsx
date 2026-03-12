import { useCallback, useState } from "react";
import { Check, ChevronLeft, ChevronRight, X } from "lucide-react";
import { Button } from "~/components/ui/button";
import { StepTab } from "~/components/autopilot/question-cards/StepTab";
import { CodeEditor, useFormattedJson } from "~/components/ui/code-editor";
import { ScrollFadeContainer } from "~/components/input_output/ScrollFadeContainer";
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
    <ScrollFadeContainer maxHeight={9999} contentClassName="-mt-4 -mb-4">
      <CodeEditor
        value={formatted}
        readOnly
        allowedLanguages={["json"]}
        autoDetectLanguage={false}
      />
    </ScrollFadeContainer>
  );
}

function ContextBlock({
  block,
  index,
}: {
  block: AutoEvalContentBlock;
  index: number;
}) {
  const isFirst = index === 0;
  const labelColor = isFirst ? "text-blue-500" : "text-emerald-500";
  const borderColor = isFirst ? "border-blue-200" : "border-emerald-200";

  const content = (() => {
    switch (block.type) {
      case "json":
        return <JsonBlock data={block.data} />;
      case "markdown":
        return (
          <ScrollFadeContainer maxHeight={9999} contentClassName="-mt-4 -mb-4">
            <CodeEditor
              value={block.text}
              readOnly
              showLineNumbers={false}
              allowedLanguages={["markdown"]}
              autoDetectLanguage={false}
            />
          </ScrollFadeContainer>
        );
      default: {
        const _exhaustiveCheck: never = block;
        return _exhaustiveCheck;
      }
    }
  })();

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-1">
      {block.label && (
        <span className={cn("shrink-0 text-sm font-medium", labelColor)}>
          {block.label}
        </span>
      )}
      <div
        className={cn(
          "flex min-h-0 flex-1 flex-col border-l-2 pl-2",
          borderColor,
        )}
      >
        {content}
      </div>
    </div>
  );
}

function ContextGrid({ blocks }: { blocks: AutoEvalContentBlock[] }) {
  const isSideBySide = blocks.length >= 2;
  return (
    <div
      className={cn(
        "grid min-h-0 flex-1 grid-cols-1 gap-2 md:gap-4",
        isSideBySide && "md:grid-cols-2",
      )}
    >
      {blocks.map((block, i) => (
        <ContextBlock key={i} block={block} index={i} />
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
    <div className="flex max-h-[70vh] flex-col rounded-md border border-border bg-bg-primary dark:bg-bg-primary">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between gap-4 px-4 pt-3 pb-3">
        <span className="text-sm font-medium text-fg-primary">
          Label examples to improve evaluator accuracy
        </span>
        <button
          type="button"
          onClick={handleDismiss}
          disabled={isLoading}
          className="-mr-1 cursor-pointer rounded-sm p-0.5 text-fg-tertiary transition-colors hover:text-fg-secondary disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Dismiss examples"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Step tabs */}
      {!isSingleExample && (
        <nav className="flex shrink-0 gap-1 overflow-x-auto px-3 pb-3">
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

      {/* Content — code blocks flex to fill, controls are fixed at bottom */}
      <div className="flex min-h-0 flex-1 flex-col gap-3 px-4">
        <ContextGrid blocks={example.context} />

        <div className="flex shrink-0 flex-col gap-1.5">
          <span className="text-fg-primary text-sm">
            {example.label_question.question}
          </span>
          <div className="flex gap-2">
            {example.label_question.options.map((opt) => {
              const isSelected = selections[activeIndex] === opt.id;
              const button = (
                <OptionButton
                  key={opt.id}
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
                      ? "text-orange-700 dark:text-orange-300"
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

              return button;
            })}
          </div>
        </div>

        {example.explanation_question && selections[activeIndex] && (
          <div className="flex shrink-0 flex-col gap-1.5">
            <span className="text-fg-primary text-sm">
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
              className="border-input focus-visible:border-border-accent bg-bg-primary w-full resize-none overscroll-none rounded-md border px-3 py-2 text-sm transition-colors focus-visible:ring-0 focus-visible:outline-hidden disabled:cursor-not-allowed disabled:opacity-50"
              rows={1}
            />
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex shrink-0 items-center justify-between px-4 pt-3 pb-3">
        <div>
          {!isSingleExample && !isFirst && (
            <Button
              variant="ghost"
              size="xs"
              disabled={isLoading}
              onClick={() => setActiveIndex((s) => s - 1)}
              className="gap-0.5 pl-1 text-neutral-700 hover:text-neutral-800 dark:text-neutral-300 dark:hover:text-neutral-200"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!isSingleExample && (
            <span className="text-fg-primary text-xs">
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
              className="gap-0.5 pr-1"
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
