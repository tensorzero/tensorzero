import { Check, ChevronLeft, ChevronRight, X } from "lucide-react";
import { Button } from "~/components/ui/button";
import { cn } from "~/utils/common";
import type {
  EventPayloadUserQuestions,
  UserQuestionAnswer,
} from "~/types/tensorzero";
import { MultipleChoiceStep } from "./MultipleChoiceStep";
import { FreeResponseStep } from "./FreeResponseStep";
import { StepTab } from "./StepTab";
import { useQuestionCardState } from "./useQuestionCardState";
import { useAnimatedHeight } from "~/hooks/useAnimatedHeight";

type PendingQuestionCardProps = {
  eventId: string;
  payload: EventPayloadUserQuestions;
  isLoading: boolean;
  onSubmit: (
    eventId: string,
    answers: Record<string, UserQuestionAnswer>,
  ) => void;
  onSkip?: () => void;
  className?: string;
};

export function PendingQuestionCard({
  eventId,
  payload,
  isLoading,
  onSubmit,
  onSkip,
  className,
}: PendingQuestionCardProps) {
  const state = useQuestionCardState(payload, eventId, onSubmit);
  const { ref: contentRef, height: contentHeight } = useAnimatedHeight(
    state.activeStep,
  );

  const renderActiveStep = () => {
    const data = state.getStepData(state.activeStep);
    switch (data.question.type) {
      case "multiple_choice":
        return (
          <MultipleChoiceStep
            question={data.question}
            selectedValues={data.selectedValues}
            onToggle={data.onToggle}
          />
        );
      case "free_response":
        return (
          <FreeResponseStep
            question={data.question}
            text={data.freeText}
            onTextChange={data.onFreeTextChange}
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
          {state.isSingleQuestion ? "Question" : "Questions"}
        </span>
        {onSkip && (
          <button
            type="button"
            onClick={onSkip}
            className="-mr-1 cursor-pointer rounded-sm p-0.5 text-purple-400 transition-colors hover:text-purple-600 dark:text-purple-500 dark:hover:text-purple-300"
            aria-label="Dismiss questions"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Step tabs (only for multi-question) */}
      {!state.isSingleQuestion && (
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3">
          {payload.questions.map((q, idx) => (
            <StepTab
              key={idx}
              index={idx}
              label={q.header}
              state={
                idx === state.activeStep
                  ? "active"
                  : state.isStepValid(idx)
                    ? "completed"
                    : "upcoming"
              }
              onClick={() => state.setActiveStep(idx)}
            />
          ))}
        </nav>
      )}

      {/* Content area */}
      <div className="flex min-w-0 flex-1 flex-col justify-between gap-3 px-4 pb-3">
        <div
          ref={contentRef}
          className="overflow-hidden transition-[height] duration-300 ease-in-out"
          style={{ height: contentHeight }}
        >
          <div
            key={state.activeStep}
            className="animate-in fade-in duration-300"
          >
            {renderActiveStep()}
          </div>
        </div>

        {/* Footer: Back / Next / Submit */}
        <div className="flex items-center justify-between">
          <div>
            {!state.isSingleQuestion && !state.isFirstStep && (
              <Button
                variant="ghost"
                size="xs"
                onClick={() => state.setActiveStep((s) => s - 1)}
                className="gap-0.5 pl-1 text-purple-700 hover:text-purple-800 dark:text-purple-300 dark:hover:text-purple-200"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                Back
              </Button>
            )}
          </div>
          <div>
            {state.isSingleQuestion || state.isLastStep ? (
              <Button
                size="xs"
                disabled={!state.allStepsValid || isLoading}
                onClick={state.handleSubmit}
                className="gap-1 bg-purple-600 text-white hover:bg-purple-700 dark:bg-purple-600 dark:hover:bg-purple-500"
              >
                {isLoading ? "Submitting..." : "Submit"}
                <Check className="h-3.5 w-3.5" />
              </Button>
            ) : (
              <Button
                size="xs"
                disabled={!state.isStepValid(state.activeStep)}
                onClick={() => state.setActiveStep((s) => s + 1)}
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
  );
}
