import { useState } from "react";
import type {
  EventPayloadUserQuestions,
  UserQuestionAnswer,
} from "~/types/tensorzero";

type StepAnswer =
  | { status: "unanswered" }
  | { status: "skipped" }
  | { status: "answered_mc"; selected: Set<string> }
  | { status: "answered_free"; text: string };

type StepAnswers = Map<number, StepAnswer>;

function getStep(answers: StepAnswers, idx: number): StepAnswer {
  return answers.get(idx) ?? { status: "unanswered" };
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
        step.status === "answered_mc" ? step.selected : new Set<string>();

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
      next.set(questionIndex, { status: "answered_mc", selected: updated });
      return next;
    });
  };

  const handleFreeTextChange = (questionIndex: number, text: string) => {
    setAnswers((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, { status: "answered_free", text });
      return next;
    });
  };

  const isStepAnswered = (idx: number): boolean => {
    const step = getStep(answers, idx);
    switch (step.status) {
      case "answered_mc":
        return step.selected.size > 0;
      case "answered_free":
        return step.text.trim().length > 0;
      case "unanswered":
      case "skipped":
        return false;
    }
  };

  const isStepSkipped = (idx: number): boolean =>
    getStep(answers, idx).status === "skipped";

  const isStepComplete = (idx: number): boolean =>
    isStepAnswered(idx) || isStepSkipped(idx);

  const buildResponses = (
    finalAnswers: StepAnswers,
  ): Record<string, UserQuestionAnswer> => {
    const responses: Record<string, UserQuestionAnswer> = {};
    payload.questions.forEach((question, idx) => {
      const step = getStep(finalAnswers, idx);
      switch (step.status) {
        case "skipped":
        case "unanswered":
          responses[question.id] = { type: "skipped" };
          break;
        case "answered_mc":
          responses[question.id] = {
            type: "multiple_choice",
            selected: Array.from(step.selected),
          };
          break;
        case "answered_free":
          responses[question.id] = {
            type: "free_response",
            text: step.text,
          };
          break;
      }
    });
    return responses;
  };

  const markUnansweredAsSkipped = (base: StepAnswers): StepAnswers => {
    const final = new Map(base);
    payload.questions.forEach((_, idx) => {
      if (!isStepAnswered(idx) && getStep(final, idx).status !== "skipped") {
        final.set(idx, { status: "skipped" });
      }
    });
    return final;
  };

  const handleSkipStep = () => {
    const updated = new Map(answers);
    updated.set(activeStep, { status: "skipped" });

    if (isLastStep) {
      const final = markUnansweredAsSkipped(updated);
      onSubmit(eventId, buildResponses(final));
    } else {
      setAnswers(updated);
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
        step.status === "answered_mc" ? step.selected : new Set<string>(),
      freeText: step.status === "answered_free" ? step.text : "",
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
