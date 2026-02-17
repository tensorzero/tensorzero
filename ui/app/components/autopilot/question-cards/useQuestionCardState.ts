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

  const questionCount = payload.questions.length;
  const isSingleQuestion = questionCount === 1;
  const isFirstStep = activeStep === 0;
  const isLastStep = activeStep === questionCount - 1;

  const handleMcToggle = (questionIndex: number, value: string) => {
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
    setFreeTexts((prev) => {
      const next = new Map(prev);
      next.set(questionIndex, text);
      return next;
    });
  };

  const isStepValid = (idx: number): boolean => {
    const question = payload.questions[idx];
    switch (question.type) {
      case "multiple_choice": {
        const selected = selections.get(idx);
        return Boolean(selected && selected.size > 0);
      }
      case "free_response":
        return (freeTexts.get(idx) ?? "").trim().length > 0;
    }
  };

  const allStepsValid = payload.questions.every((_, idx) => isStepValid(idx));

  const handleSubmit = () => {
    const responses: Record<string, UserQuestionAnswer> = {};
    payload.questions.forEach((question, idx) => {
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
      }
    });
    onSubmit(eventId, responses);
  };

  const renderStepData = (idx: number) => {
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
    questionCount,
    isSingleQuestion,
    isFirstStep,
    isLastStep,
    isStepValid,
    allStepsValid,
    handleSubmit,
    renderStepData,
  };
}
