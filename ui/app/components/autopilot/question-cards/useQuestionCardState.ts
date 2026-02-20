import { useState } from "react";
import type {
  EventPayloadUserQuestions,
  UserQuestionAnswer,
} from "~/types/tensorzero";

export function useQuestionCardState(
  payload: EventPayloadUserQuestions,
  eventId: string,
  onSubmit: (
    eventId: string,
    responses: Record<string, UserQuestionAnswer>,
  ) => void,
) {
  const [activeStep, setActiveStep] = useState(0);
  const [selections, setSelections] = useState<Map<number, Set<string>>>(
    () => new Map(),
  );
  const [freeTexts, setFreeTexts] = useState<Map<number, string>>(
    () => new Map(),
  );
  const [skippedSteps, setSkippedSteps] = useState<Set<number>>(
    () => new Set(),
  );

  const questionCount = payload.questions.length;
  const isSingleQuestion = questionCount === 1;
  const isFirstStep = activeStep === 0;
  const isLastStep = activeStep === questionCount - 1;

  const handleMcToggle = (questionIndex: number, value: string) => {
    // Unskip if user starts answering
    setSkippedSteps((prev) => {
      if (!prev.has(questionIndex)) return prev;
      const next = new Set(prev);
      next.delete(questionIndex);
      return next;
    });
    setSelections((prev) => {
      const next = new Map(prev);
      const question = payload.questions[questionIndex];
      if (question.type !== "multiple_choice") return prev;
      const current = next.get(questionIndex) ?? new Set<string>();

      if (question.multi_select) {
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

  const handleFreeTextChange = (questionIndex: number, text: string) => {
    // Unskip if user starts typing
    setSkippedSteps((prev) => {
      if (!prev.has(questionIndex)) return prev;
      const next = new Set(prev);
      next.delete(questionIndex);
      return next;
    });
    setFreeTexts((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, text);
      return next;
    });
  };

  const isStepAnswered = (idx: number): boolean => {
    const question = payload.questions[idx];
    switch (question.type) {
      case "multiple_choice": {
        const selected = selections.get(idx);
        return Boolean(selected && selected.size > 0);
      }
      case "free_response":
        return (freeTexts.get(idx) ?? "").trim().length > 0;
      default: {
        const _exhaustiveCheck: never = question;
        return _exhaustiveCheck;
      }
    }
  };

  const isStepSkipped = (idx: number): boolean => skippedSteps.has(idx);

  const isStepComplete = (idx: number): boolean =>
    isStepAnswered(idx) || isStepSkipped(idx);

  const allStepsComplete = payload.questions.every((_, idx) =>
    isStepComplete(idx),
  );

  const handleSkipStep = () => {
    setSkippedSteps((prev) => {
      const next = new Set(prev);
      next.add(activeStep);
      return next;
    });

    if (isSingleQuestion || isLastStep) {
      // Build responses and submit immediately
      const responses: Record<string, UserQuestionAnswer> = {};
      payload.questions.forEach((question, idx) => {
        if (idx === activeStep || skippedSteps.has(idx)) {
          responses[question.id] = { type: "skipped" };
          return;
        }
        switch (question.type) {
          case "multiple_choice": {
            const selected = selections.get(idx);
            if (!selected || selected.size === 0) return;
            responses[question.id] = {
              type: "multiple_choice",
              selected: Array.from(selected),
            };
            break;
          }
          case "free_response":
            responses[question.id] = {
              type: "free_response",
              text: freeTexts.get(idx) ?? "",
            };
            break;
          default: {
            const _exhaustiveCheck: never = question;
            return _exhaustiveCheck;
          }
        }
      });
      onSubmit(eventId, responses);
    } else {
      setActiveStep((s) => s + 1);
    }
  };

  const handleSubmit = () => {
    const responses: Record<string, UserQuestionAnswer> = {};
    payload.questions.forEach((question, idx) => {
      if (skippedSteps.has(idx)) {
        responses[question.id] = { type: "skipped" };
        return;
      }
      switch (question.type) {
        case "multiple_choice": {
          const selected = selections.get(idx);
          if (!selected || selected.size === 0) return;
          responses[question.id] = {
            type: "multiple_choice",
            selected: Array.from(selected),
          };
          break;
        }
        case "free_response":
          responses[question.id] = {
            type: "free_response",
            text: freeTexts.get(idx) ?? "",
          };
          break;
        default: {
          const _exhaustiveCheck: never = question;
          return _exhaustiveCheck;
        }
      }
    });
    onSubmit(eventId, responses);
  };

  const getStepData = (idx: number) => {
    const question = payload.questions[idx];
    return {
      question,
      selectedValues: selections.get(idx) ?? new Set<string>(),
      freeText: freeTexts.get(idx) ?? "",
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
    allStepsComplete,
    handleSkipStep,
    handleSubmit,
    getStepData,
  };
}
