import { useState } from "react";
import type {
  EventPayloadUserQuestions,
  UserQuestionAnswer,
} from "~/types/tensorzero";

enum StepStatus {
  Unanswered = "unanswered",
  Skipped = "skipped",
  AnsweredMultipleChoice = "answered_multiple_choice",
  AnsweredFreeResponse = "answered_free_response",
}

type StepAnswer =
  | { status: StepStatus.Unanswered }
  | { status: StepStatus.Skipped }
  | { status: StepStatus.AnsweredMultipleChoice; selected: Set<string> }
  | { status: StepStatus.AnsweredFreeResponse; text: string };

type StepAnswers = Map<number, StepAnswer>;

function getStep(answers: StepAnswers, idx: number): StepAnswer {
  return answers.get(idx) ?? { status: StepStatus.Unanswered };
}

export function useQuestionCardState(
  payload: EventPayloadUserQuestions,
  eventId: string,
  onSubmit: (
    eventId: string,
    responses: Record<string, UserQuestionAnswer>,
  ) => void,
) {
  const [activeStep, setActiveStep] = useState(0);
  const [answers, setAnswers] = useState<StepAnswers>(() => new Map());

  const questionCount = payload.questions.length;
  const isSingleQuestion = questionCount === 1;
  const isFirstStep = activeStep === 0;
  const isLastStep = activeStep === questionCount - 1;

  const handleMcToggle = (questionIndex: number, value: string) => {
    setAnswers((prev) => {
      const question = payload.questions[questionIndex];
      if (question.type !== "multiple_choice") return prev;

      const step = getStep(prev, questionIndex);
      const current =
        step.status === StepStatus.AnsweredMultipleChoice
          ? step.selected
          : new Set<string>();

      let updated: Set<string>;
      if (question.multi_select) {
        updated = new Set(current);
        if (updated.has(value)) {
          updated.delete(value);
        } else {
          updated.add(value);
        }
      } else {
        updated = new Set([value]);
      }

      const next = new Map(prev);
      next.set(questionIndex, {
        status: StepStatus.AnsweredMultipleChoice,
        selected: updated,
      });
      return next;
    });
  };

  const handleFreeTextChange = (questionIndex: number, text: string) => {
    setAnswers((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, {
        status: StepStatus.AnsweredFreeResponse,
        text,
      });
      return next;
    });
  };

  const isStepAnswered = (idx: number): boolean => {
    const step = getStep(answers, idx);
    switch (step.status) {
      case StepStatus.AnsweredMultipleChoice:
        return step.selected.size > 0;
      case StepStatus.AnsweredFreeResponse:
        return step.text.trim().length > 0;
      case StepStatus.Unanswered:
      case StepStatus.Skipped:
        return false;
      default: {
        const _exhaustiveCheck: never = step;
        return _exhaustiveCheck;
      }
    }
  };

  const isStepSkipped = (idx: number): boolean =>
    getStep(answers, idx).status === StepStatus.Skipped;

  const isStepComplete = (idx: number): boolean =>
    isStepAnswered(idx) || isStepSkipped(idx);

  const buildResponses = (
    finalAnswers: StepAnswers,
  ): Record<string, UserQuestionAnswer> => {
    const responses: Record<string, UserQuestionAnswer> = {};
    payload.questions.forEach((question, idx) => {
      const step = getStep(finalAnswers, idx);
      switch (step.status) {
        case StepStatus.Skipped:
        case StepStatus.Unanswered:
          responses[question.id] = { type: "skipped" };
          break;
        case StepStatus.AnsweredMultipleChoice:
          responses[question.id] = {
            type: "multiple_choice",
            selected: Array.from(step.selected),
          };
          break;
        case StepStatus.AnsweredFreeResponse:
          responses[question.id] = {
            type: "free_response",
            text: step.text,
          };
          break;
        default: {
          const _exhaustiveCheck: never = step;
          return _exhaustiveCheck;
        }
      }
    });
    return responses;
  };

  const markUnansweredAsSkipped = (base: StepAnswers): StepAnswers => {
    const result = new Map(base);
    payload.questions.forEach((_, idx) => {
      const step = getStep(base, idx);
      const answered =
        (step.status === StepStatus.AnsweredMultipleChoice &&
          step.selected.size > 0) ||
        (step.status === StepStatus.AnsweredFreeResponse &&
          step.text.trim().length > 0);
      if (!answered && step.status !== StepStatus.Skipped) {
        result.set(idx, { status: StepStatus.Skipped });
      }
    });
    return result;
  };

  const handleSkipStep = () => {
    const updated = new Map(answers);
    updated.set(activeStep, { status: StepStatus.Skipped });
    setAnswers(updated);

    if (isLastStep) {
      const final = markUnansweredAsSkipped(updated);
      onSubmit(eventId, buildResponses(final));
    } else {
      setActiveStep((s) => s + 1);
    }
  };

  const handleSubmit = () => {
    const final = markUnansweredAsSkipped(answers);
    onSubmit(eventId, buildResponses(final));
  };

  const getStepData = (idx: number) => {
    const question = payload.questions[idx];
    const step = getStep(answers, idx);
    return {
      question,
      selectedValues:
        step.status === StepStatus.AnsweredMultipleChoice
          ? step.selected
          : new Set<string>(),
      freeText:
        step.status === StepStatus.AnsweredFreeResponse ? step.text : "",
      onToggle: (value: string) => handleMcToggle(idx, value),
      onFreeTextChange: (text: string) => handleFreeTextChange(idx, text),
    };
  };

  return {
    activeStep,
    setActiveStep,
    isSingleQuestion,
    isFirstStep,
    isLastStep,
    isStepAnswered,
    isStepSkipped,
    isStepComplete,
    handleSkipStep,
    handleSubmit,
    getStepData,
  };
}
