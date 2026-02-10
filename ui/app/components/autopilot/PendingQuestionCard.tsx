import {
  Check,
  ChevronLeft,
  ChevronRight,
  Flag,
  MessageSquareMore,
  X,
} from "lucide-react";
import { useLayoutEffect, useRef, useState } from "react";
import { Button } from "~/components/ui/button";
import { Textarea } from "~/components/ui/textarea";
import { DotSeparator } from "~/components/ui/DotSeparator";
import { TableItemTime } from "~/components/ui/TableItems";
import { ToolEventId } from "~/components/autopilot/EventStream";
import { cn } from "~/utils/common";

// Renders a string with inline markdown (bold, italic, code).
// Strips all block-level formatting — only inline runs are rendered.
function InlineMarkdown({ text }: { text: string }) {
  // Tokenize the text into segments: code, bold, italic, and plain text.
  // Order matters — code first (so backtick content isn't processed for bold/italic).
  const TOKEN_RE = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|_[^_]+_)/g;

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = TOKEN_RE.exec(text)) !== null) {
    // Plain text before this token
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const token = match[0];
    if (token.startsWith("`")) {
      parts.push(
        <code
          key={match.index}
          className="bg-muted rounded px-1 py-0.5 font-mono text-xs font-medium"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**")) {
      parts.push(
        <strong key={match.index} className="font-semibold">
          {token.slice(2, -2)}
        </strong>,
      );
    } else if (token.startsWith("*") || token.startsWith("_")) {
      parts.push(
        <em key={match.index} className="italic">
          {token.slice(1, -1)}
        </em>,
      );
    }

    lastIndex = match.index + token.length;
  }

  // Remaining plain text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return <>{parts}</>;
}

/**
 * PROTOTYPE: Types for the AskUserQuestion event payload.
 * Replace these with Rust bindings once the backend defines the `ask_user_question`
 * event schema. The type structure (discriminated union on `type` field) is intentional
 * and should be preserved in the Rust types.
 */
export type QuestionOption = {
  label: string;
  description: string;
};

export type MultipleChoiceQuestion = {
  type: "multiple_choice";
  question: string;
  header: string;
  options: QuestionOption[];
  multiSelect: boolean;
};

export type FreeResponseQuestion = {
  type: "free_response";
  question: string;
  header: string;
  placeholder?: string;
};

export type RatingQuestion = {
  type: "rating";
  question: string;
  header: string;
  min: number;
  max: number;
  minLabel?: string;
  maxLabel?: string;
};

export type AskUserQuestionItem =
  | MultipleChoiceQuestion
  | FreeResponseQuestion
  | RatingQuestion;

export type AskUserQuestionPayload = {
  questions: AskUserQuestionItem[];
};

type PendingQuestionCardProps = {
  eventId: string;
  payload: AskUserQuestionPayload;
  isLoading: boolean;
  onSubmit: (eventId: string, answers: Record<string, string>) => void;
  onSkip?: () => void;
  className?: string;
  /** Tab layout: "vertical" (left sidebar, default) or "horizontal" (top row) */
  tabLayout?: "vertical" | "horizontal";
};

// --- Step renderers ---

function MultipleChoiceStep({
  question,
  selectedValues,
  onToggle,
  otherText,
  onOtherTextChange,
}: {
  question: MultipleChoiceQuestion;
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
          const isSelected = selectedValues.has(option.label);
          return (
            <button
              key={option.label}
              type="button"
              onClick={() => onToggle(option.label)}
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

function FreeResponseStep({
  question,
  text,
  onTextChange,
}: {
  question: FreeResponseQuestion;
  text: string;
  onTextChange: (text: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <span className="text-fg-primary text-sm font-medium">
        <InlineMarkdown text={question.question} />
      </span>
      <Textarea
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
        placeholder={question.placeholder ?? "Type your response..."}
        className="bg-bg-secondary min-h-[80px] resize-none text-sm"
        rows={3}
        autoFocus
      />
    </div>
  );
}

function RatingStep({
  question,
  value,
  onValueChange,
}: {
  question: RatingQuestion;
  value: number | null;
  onValueChange: (value: number) => void;
}) {
  const numbers = Array.from(
    { length: question.max - question.min + 1 },
    (_, i) => question.min + i,
  );

  return (
    <div className="flex flex-col gap-3">
      <span className="text-fg-primary text-sm font-medium">
        <InlineMarkdown text={question.question} />
      </span>
      <div className="flex flex-col gap-1.5">
        <div className="flex gap-1.5">
          {numbers.map((n) => {
            const isSelected = value === n;
            return (
              <button
                key={n}
                type="button"
                onClick={() => onValueChange(n)}
                className={cn(
                  "flex h-9 min-w-0 flex-1 items-center justify-center rounded-lg border text-sm font-medium transition-all",
                  isSelected
                    ? "border-purple-500 bg-purple-50 text-purple-700 ring-1 ring-purple-500 dark:border-purple-400 dark:bg-purple-950/40 dark:text-purple-300 dark:ring-purple-400"
                    : "border-border bg-bg-secondary text-fg-primary hover:border-purple-300 hover:bg-purple-50/50 dark:hover:border-purple-600 dark:hover:bg-purple-950/20",
                )}
              >
                {n}
              </button>
            );
          })}
        </div>
        {(question.minLabel || question.maxLabel) && (
          <div className="text-fg-muted flex justify-between px-1 text-xs">
            <span>{question.minLabel ?? ""}</span>
            <span>{question.maxLabel ?? ""}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Step Tab (left sidebar) ---

function StepTab({
  index,
  label,
  state,
  onClick,
}: {
  index: number;
  label: string;
  state: "completed" | "active" | "upcoming";
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs font-medium transition-all",
        state === "active" &&
          "bg-purple-200/70 text-purple-800 dark:bg-purple-800/50 dark:text-purple-200",
        state === "completed" &&
          "cursor-pointer text-green-700 hover:bg-green-50 dark:text-green-400 dark:hover:bg-green-900/20",
        state === "upcoming" &&
          "text-fg-muted cursor-pointer hover:bg-purple-100/50 dark:hover:bg-purple-900/20",
      )}
      aria-label={`Go to question ${index + 1}: ${label}`}
      aria-current={state === "active" ? "step" : undefined}
    >
      {state === "completed" ? (
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/40">
          <Check className="h-3 w-3" />
        </span>
      ) : (
        <span
          className={cn(
            "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold",
            state === "active" &&
              "bg-purple-600 text-white dark:bg-purple-400 dark:text-purple-950",
            state === "upcoming" && "border-fg-muted border bg-transparent",
          )}
        >
          {index + 1}
        </span>
      )}
      <span className="truncate">{label}</span>
    </button>
  );
}

// --- Horizontal Step Tab (top bar) ---

function HorizontalStepTab({
  index,
  label,
  state,
  onClick,
  icon,
}: {
  index: number;
  label: string;
  state: "completed" | "active" | "upcoming";
  onClick: () => void;
  icon?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-all",
        state === "active" &&
          "bg-purple-200/70 text-purple-800 dark:bg-purple-800/50 dark:text-purple-200",
        state === "completed" &&
          "cursor-pointer text-green-700 hover:bg-green-50 dark:text-green-400 dark:hover:bg-green-900/20",
        state === "upcoming" &&
          "text-fg-muted cursor-pointer hover:bg-purple-100/50 dark:hover:bg-purple-900/20",
      )}
      aria-label={`Go to question ${index + 1}: ${label}`}
      aria-current={state === "active" ? "step" : undefined}
    >
      {state === "completed" ? (
        <span className="flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/40">
          <Check className="h-2.5 w-2.5" />
        </span>
      ) : (
        <span
          className={cn(
            "flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold",
            state === "active" &&
              "bg-purple-600 text-white dark:bg-purple-400 dark:text-purple-950",
            state === "upcoming" && "border-fg-muted border bg-transparent",
          )}
        >
          {icon ?? index + 1}
        </span>
      )}
      <span className="truncate">{label}</span>
    </button>
  );
}

// --- Main component ---

export function PendingQuestionCard({
  eventId,
  payload,
  isLoading,
  onSubmit,
  onSkip,
  className,
  tabLayout = "vertical",
}: PendingQuestionCardProps) {
  const [activeStep, setActiveStep] = useState(0);

  // State per question type
  const [selections, setSelections] = useState<Map<number, Set<string>>>(
    () => new Map(),
  );
  const [otherTexts, setOtherTexts] = useState<Map<number, string>>(
    () => new Map(),
  );
  const [freeTexts, setFreeTexts] = useState<Map<number, string>>(
    () => new Map(),
  );
  const [ratingValues, setRatingValues] = useState<Map<number, number>>(
    () => new Map(),
  );

  // Animate content area height on step change.
  // To measure the *natural* height of the new content, temporarily set
  // height to auto (scrollHeight = max(content, cssHeight), so it can't
  // measure a shrink without this trick). Then restore and animate.
  const contentRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState<number | undefined>(
    undefined,
  );
  const isFirstRender = useRef(true);
  useLayoutEffect(() => {
    const el = contentRef.current;
    if (!el) return;

    // Temporarily remove explicit height so scrollHeight reflects content
    const prevHeight = el.style.height;
    el.style.height = "auto";
    const naturalHeight = el.scrollHeight;
    el.style.height = prevHeight; // restore before paint

    if (isFirstRender.current) {
      isFirstRender.current = false;
      setContentHeight(naturalHeight);
      return;
    }
    // Force a reflow at the old height so the transition has a start value
    el.getBoundingClientRect();
    setContentHeight(naturalHeight);
  }, [activeStep]);

  const questionCount = payload.questions.length;
  const isSingleQuestion = questionCount === 1;
  const isFirstStep = activeStep === 0;
  const isReviewStep = !isSingleQuestion && activeStep === questionCount;

  const handleMcToggle = (questionIndex: number, value: string) => {
    setSelections((prev) => {
      const next = new Map(prev);
      const question = payload.questions[questionIndex];
      if (question.type !== "multiple_choice") return prev;
      const current = next.get(questionIndex) ?? new Set<string>();

      if (question.multiSelect) {
        const updated = new Set(current);
        if (updated.has(value)) {
          updated.delete(value);
        } else {
          updated.add(value);
        }
        next.set(questionIndex, updated);
      } else {
        next.set(questionIndex, new Set([value]));
      }
      return next;
    });
  };

  const handleOtherTextChange = (questionIndex: number, text: string) => {
    setOtherTexts((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, text);
      return next;
    });
  };

  const handleFreeTextChange = (questionIndex: number, text: string) => {
    setFreeTexts((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, text);
      return next;
    });
  };

  const handleRatingChange = (questionIndex: number, value: number) => {
    setRatingValues((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, value);
      return next;
    });
  };

  const isStepValid = (idx: number): boolean => {
    const question = payload.questions[idx];
    switch (question.type) {
      case "multiple_choice": {
        const selected = selections.get(idx);
        if (!selected || selected.size === 0) return false;
        if (selected.has("__other__")) {
          return (otherTexts.get(idx) ?? "").trim().length > 0;
        }
        return true;
      }
      case "free_response":
        return (freeTexts.get(idx) ?? "").trim().length > 0;
      case "rating":
        return ratingValues.has(idx);
    }
  };

  const allStepsValid = payload.questions.every((_, idx) => isStepValid(idx));

  const handleSubmit = () => {
    const answers: Record<string, string> = {};
    payload.questions.forEach((question, idx) => {
      switch (question.type) {
        case "multiple_choice": {
          const selected = selections.get(idx);
          if (!selected || selected.size === 0) return;
          if (selected.has("__other__")) {
            answers[question.header] = otherTexts.get(idx) ?? "";
          } else {
            answers[question.header] = Array.from(selected).join(", ");
          }
          break;
        }
        case "free_response":
          answers[question.header] = freeTexts.get(idx) ?? "";
          break;
        case "rating":
          answers[question.header] = String(ratingValues.get(idx) ?? "");
          break;
      }
    });
    onSubmit(eventId, answers);
  };

  const getAnswerText = (idx: number): string => {
    const question = payload.questions[idx];
    switch (question.type) {
      case "multiple_choice": {
        const selected = selections.get(idx);
        if (!selected || selected.size === 0) return "";
        if (selected.has("__other__")) return otherTexts.get(idx) ?? "";
        return Array.from(selected).join(", ");
      }
      case "free_response":
        return freeTexts.get(idx) ?? "";
      case "rating":
        return ratingValues.has(idx) ? String(ratingValues.get(idx)) : "";
    }
  };

  const renderStep = (question: AskUserQuestionItem, idx: number) => {
    switch (question.type) {
      case "multiple_choice":
        return (
          <MultipleChoiceStep
            question={question}
            selectedValues={selections.get(idx) ?? new Set()}
            onToggle={(value) => handleMcToggle(idx, value)}
            otherText={otherTexts.get(idx) ?? ""}
            onOtherTextChange={(text) => handleOtherTextChange(idx, text)}
          />
        );
      case "free_response":
        return (
          <FreeResponseStep
            question={question}
            text={freeTexts.get(idx) ?? ""}
            onTextChange={(text) => handleFreeTextChange(idx, text)}
          />
        );
      case "rating":
        return (
          <RatingStep
            question={question}
            value={ratingValues.get(idx) ?? null}
            onValueChange={(value) => handleRatingChange(idx, value)}
          />
        );
    }
  };

  return (
    <div
      className={cn(
        "flex flex-col rounded-md border border-purple-300 bg-purple-50 dark:border-purple-700 dark:bg-purple-950/30",
        className,
      )}
    >
      {/* Header row */}
      <div className="flex items-center justify-between gap-4 px-4 py-3">
        <span className="text-sm font-medium">
          {isSingleQuestion ? "Question" : "Questions"}
        </span>
        {onSkip && (
          <button
            type="button"
            onClick={onSkip}
            className="text-purple-400 hover:text-purple-600 dark:text-purple-500 dark:hover:text-purple-300 -mr-1 cursor-pointer rounded-sm p-0.5 transition-colors"
            aria-label="Dismiss questions"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Horizontal tabs (top row) */}
      {!isSingleQuestion && tabLayout === "horizontal" && (
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3">
          {payload.questions.map((q, idx) => (
            <HorizontalStepTab
              key={idx}
              index={idx}
              label={q.header}
              state={
                idx === activeStep
                  ? "active"
                  : isStepValid(idx)
                    ? "completed"
                    : "upcoming"
              }
              onClick={() => setActiveStep(idx)}
            />
          ))}
          <HorizontalStepTab
            index={questionCount}
            label="Review"
            icon={<Flag className="h-2.5 w-2.5" />}
            state={
              isReviewStep ? "active" : allStepsValid ? "upcoming" : "upcoming"
            }
            onClick={() => {
              if (allStepsValid) setActiveStep(questionCount);
            }}
          />
        </nav>
      )}

      {/* Body */}
      <div className="flex">
        {/* Left sidebar tabs (only for multi-question + vertical layout) */}
        {!isSingleQuestion && tabLayout === "vertical" && (
          <nav className="flex w-32 shrink-0 flex-col gap-0.5 px-2 pb-3">
            {payload.questions.map((q, idx) => (
              <StepTab
                key={idx}
                index={idx}
                label={q.header}
                state={
                  idx === activeStep
                    ? "active"
                    : isStepValid(idx)
                      ? "completed"
                      : "upcoming"
                }
                onClick={() => setActiveStep(idx)}
              />
            ))}
            <StepTab
              index={questionCount}
              label="Review"
              state={
                isReviewStep
                  ? "active"
                  : allStepsValid
                    ? "upcoming"
                    : "upcoming"
              }
              onClick={() => {
                if (allStepsValid) setActiveStep(questionCount);
              }}
            />
          </nav>
        )}

        {/* Content area — min-h prevents jitter when stepping between question types */}
        <div
          className={cn(
            "flex min-w-0 flex-1 flex-col justify-between gap-3 pb-3",
            tabLayout === "vertical" ? "pr-4 pl-2" : "px-4",
          )}
        >
          {/* Only render the active step. A ref tracks the tallest
              content seen so the container never shrinks. */}
          <div
            ref={contentRef}
            className="overflow-hidden transition-[height] duration-200 ease-in-out"
            style={{ height: contentHeight }}
          >
            <div
              key={activeStep}
              className="animate-in fade-in duration-150"
            >
            {isReviewStep ? (
              <div className="flex flex-col gap-3">
                <span className="text-fg-primary text-sm font-medium">
                  Review your answers
                </span>
                <div className="flex flex-col gap-2">
                  {payload.questions.map((q, idx) => (
                    <button
                      key={q.header}
                      type="button"
                      onClick={() => setActiveStep(idx)}
                      className="border-border bg-bg-secondary flex flex-col items-start rounded-lg border px-3 py-2 text-left transition-all hover:border-purple-300 dark:hover:border-purple-600"
                    >
                      <span className="text-fg-muted text-xs font-medium">
                        {q.header}
                      </span>
                      <span className="text-fg-primary text-sm">
                        {getAnswerText(idx) || "—"}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              renderStep(payload.questions[activeStep], activeStep)
            )}
            </div>
          </div>

          {/* Footer: Back / Next / Submit — pinned to bottom */}
          <div className="flex items-center justify-between">
            <div>
              {!isSingleQuestion && !isFirstStep && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => setActiveStep((s) => s - 1)}
                  className="gap-0.5 pl-1 text-purple-700 hover:text-purple-800 dark:text-purple-300 dark:hover:text-purple-200"
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                  Back
                </Button>
              )}
            </div>
            <div>
              {isSingleQuestion || isReviewStep ? (
                <Button
                  size="xs"
                  disabled={!allStepsValid || isLoading}
                  onClick={handleSubmit}
                  className="gap-1 bg-purple-600 text-white hover:bg-purple-700 dark:bg-purple-600 dark:hover:bg-purple-500"
                >
                  {isLoading ? "Submitting..." : "Submit"}
                  <Check className="h-3.5 w-3.5" />
                </Button>
              ) : (
                <Button
                  size="xs"
                  disabled={!isStepValid(activeStep)}
                  onClick={() => setActiveStep((s) => s + 1)}
                  className="gap-0.5 bg-purple-600 pr-1 text-white hover:bg-purple-700 dark:bg-purple-600 dark:hover:bg-purple-500"
                >
                  Next
                  <ChevronRight className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Collapsed card shown in the event stream after a question has been answered.
 * Expandable to review the Q&A.
 */
export function AnsweredQuestionCard({
  payload,
  answers,
  eventId,
  timestamp,
  className,
}: {
  payload: AskUserQuestionPayload;
  answers: Record<string, string>;
  eventId: string;
  timestamp: string;
  className?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={cn(
        "border-border bg-bg-secondary flex flex-col rounded-md border",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-4 px-4 py-3">
        <button
          type="button"
          onClick={() => setIsExpanded((e) => !e)}
          className="inline-flex cursor-pointer items-center gap-2 text-left"
        >
          <span className="inline-flex items-center gap-2 text-sm font-medium">
            Question
            <DotSeparator />
            Answered
          </span>
          <span
            className={cn(
              "text-fg-muted inline-flex transition-transform duration-200",
              isExpanded ? "rotate-90" : "rotate-0",
            )}
          >
            <ChevronRight className="h-4 w-4" />
          </span>
        </button>
        <div className="text-fg-muted flex items-center gap-1.5 text-xs">
          <ToolEventId id={eventId} />
          <DotSeparator />
          <TableItemTime timestamp={timestamp} />
        </div>
      </div>

      {isExpanded && (
        <div className="flex flex-col gap-3 px-4 pb-3">
          {payload.questions.map((q) => (
            <div key={q.header} className="flex flex-col gap-0.5">
              <span className="text-fg-muted text-xs font-medium">
                {q.header}
              </span>
              <span className="text-fg-primary text-sm">
                {answers[q.header] ?? "—"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Collapsed card shown in the event stream after a question has been skipped.
 * Expandable to preview the unanswered questions.
 */
export function SkippedQuestionCard({
  payload,
  eventId,
  timestamp,
  className,
}: {
  payload: AskUserQuestionPayload;
  eventId: string;
  timestamp: string;
  className?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={cn(
        "border-border bg-bg-secondary flex flex-col rounded-md border",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-4 px-4 py-3">
        <button
          type="button"
          onClick={() => setIsExpanded((e) => !e)}
          className="inline-flex cursor-pointer items-center gap-2 text-left"
        >
          <span className="inline-flex items-center gap-2 text-sm font-medium">
            Question
            <DotSeparator />
            Skipped
          </span>
          <span
            className={cn(
              "text-fg-muted inline-flex transition-transform duration-200",
              isExpanded ? "rotate-90" : "rotate-0",
            )}
          >
            <ChevronRight className="h-4 w-4" />
          </span>
        </button>
        <div className="text-fg-muted flex items-center gap-1.5 text-xs">
          <ToolEventId id={eventId} />
          <DotSeparator />
          <TableItemTime timestamp={timestamp} />
        </div>
      </div>

      {isExpanded && (
        <div className="flex flex-col gap-3 px-4 pb-3">
          {payload.questions.map((q) => (
            <div key={q.header} className="flex flex-col gap-0.5">
              <span className="text-fg-muted text-xs font-medium">
                {q.header}
              </span>
              <span className="text-fg-muted text-sm italic">{q.question}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
