import { Check, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "~/components/ui/button";
import type {
  EventPayloadUserQuestions,
  UserQuestionAnswer,
} from "~/types/tensorzero";
import { MultipleChoiceStep } from "./MultipleChoiceStep";
import { FreeResponseStep } from "./FreeResponseStep";
import { useQuestionCardState } from "./useQuestionCardState";
import { QuestionCard } from "./QuestionCard";

type PendingQuestionCardProps = {
  eventId: string;
  payload: EventPayloadUserQuestions;
  isLoading: boolean;
  onSubmit: (
    eventId: string,
    responses: Record<string, UserQuestionAnswer>,
  ) => void;
  className?: string;
};

export function PendingQuestionCard({
  eventId,
  payload,
  isLoading,
  onSubmit,
  className,
}: PendingQuestionCardProps) {
  const state = useQuestionCardState(payload, eventId, onSubmit);

  const handleDismiss = () => {
    const responses: Record<string, UserQuestionAnswer> = {};
    for (const q of payload.questions) {
      responses[q.id] = { type: "skipped" };
    }
    onSubmit(eventId, responses);
  };

  const renderActiveStep = () => {
    const data = state.getStepData(state.activeStep);
    switch (data.question.type) {
      case "multiple_choice":
        return (
          <MultipleChoiceStep
            question={data.question}
            selectedValues={data.selectedValues}
            onToggle={data.onToggle}
            otherSelected={data.otherSelected}
            onOtherToggle={data.onOtherToggle}
            mcFreeText={data.mcFreeText}
            onMcFreeTextChange={data.onMcFreeTextChange}
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
      default: {
        const _exhaustiveCheck: never = data.question;
        return _exhaustiveCheck;
      }
    }
  };

  const getStepTabState = (idx: number) => {
    if (idx === state.activeStep) return "active" as const;
    if (state.isStepAnswered(idx)) return "completed" as const;
    if (state.isStepSkipped(idx)) return "skipped" as const;
    return "upcoming" as const;
  };

  return (
    <QuestionCard
      title={state.isSingleQuestion ? "Question" : "Questions"}
      onDismiss={handleDismiss}
      isLoading={isLoading}
      className={className}
      activeStep={state.activeStep}
      steps={
        !state.isSingleQuestion
          ? {
              items: payload.questions.map((q, idx) => ({
                id: q.id,
                label: q.header,
                state: getStepTabState(idx),
              })),
              onStepClick: (idx) => state.setActiveStep(idx),
            }
          : undefined
      }
      footer={
        <div className="flex items-center justify-between px-4 pt-3 pb-3">
          <div>
            {!state.isSingleQuestion && !state.isFirstStep && (
              <Button
                variant="ghost"
                size="xs"
                disabled={isLoading}
                onClick={() => state.setActiveStep((s) => s - 1)}
                className="gap-0.5 pl-1 text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                Back
              </Button>
            )}
          </div>
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="xs"
              disabled={isLoading}
              onClick={state.handleSkipStep}
              className="text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300"
            >
              Skip
            </Button>
            {state.isSingleQuestion || state.isLastStep ? (
              <Button
                size="xs"
                disabled={!state.isStepComplete(state.activeStep) || isLoading}
                onClick={state.handleSubmit}
                className="gap-1"
              >
                {isLoading ? "Submitting..." : "Submit"}
                <Check className="h-3.5 w-3.5" />
              </Button>
            ) : (
              <Button
                size="xs"
                disabled={!state.isStepComplete(state.activeStep) || isLoading}
                onClick={() => state.setActiveStep((s) => s + 1)}
                className="gap-0.5 pr-1 bg-purple-600 text-white hover:bg-purple-700 dark:bg-purple-500 dark:hover:bg-purple-600"
              >
                Next
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        </div>
      }
    >
      {renderActiveStep()}
    </QuestionCard>
  );
}
