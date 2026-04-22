import { useState } from "react";
import type {
  EventPayloadUserQuestions,
  UserQuestionAnswer,
} from "~/types/tensorzero";
import {
  StepStatus,
  getStep,
  applyMcToggle,
  applyOtherToggle,
  applyMcFreeTextChange,
  isStepAnswered,
  buildResponses,
  markUnansweredAsSkipped,
} from "./questionCardLogic";
import type { StepAnswers } from "./questionCardLogic";

export function useQuestionCardState(
  payload: EventPayloadUserQuestions,
  eventId: string,
  onSubmit: (
    eventId: string,
    responses: Record<string, UserQuestionAnswer>,
  ) => void,
) {
  const [activeStep, setActiveStep] = useState(0);
  const [answers, setAnswers] = useState<StepAnswers>(() => {
    const initial: StepAnswers = new Map();
    payload.questions.forEach((question, idx) => {
      if (question.type === "free_response" && question.default_value) {
        initial.set(idx, {
          status: StepStatus.AnsweredFreeResponse,
          text: question.default_value,
        });
      }
    });
    return initial;
  });

  const questionCount = payload.questions.length;
  const isSingleQuestion = questionCount === 1;
  const isFirstStep = activeStep === 0;
  const isLastStep = activeStep === questionCount - 1;

  const handleMcToggle = (questionIndex: number, value: string) => {
    setAnswers((prev) =>
      applyMcToggle(
        prev,
        questionIndex,
        value,
        payload.questions[questionIndex],
      ),
    );
  };

  const handleOtherToggle = (questionIndex: number) => {
    setAnswers((prev) =>
      applyOtherToggle(prev, questionIndex, payload.questions[questionIndex]),
    );
  };

  const handleMcFreeTextChange = (questionIndex: number, text: string) => {
    setAnswers((prev) => applyMcFreeTextChange(prev, questionIndex, text));
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

  const isStepSkipped = (idx: number): boolean =>
    getStep(answers, idx).status === StepStatus.Skipped;

  const isStepComplete = (idx: number): boolean =>
    isStepAnswered(answers, idx) || isStepSkipped(idx);

  const handleSkipStep = () => {
    const updated = new Map(answers);
    updated.set(activeStep, { status: StepStatus.Skipped });
    setAnswers(updated);

    if (isLastStep) {
      const final = markUnansweredAsSkipped(payload.questions, updated);
      onSubmit(eventId, buildResponses(payload.questions, final));
    } else {
      setActiveStep((s) => s + 1);
    }
  };

  const handleSubmit = () => {
    const final = markUnansweredAsSkipped(payload.questions, answers);
    onSubmit(eventId, buildResponses(payload.questions, final));
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
      otherSelected:
        step.status === StepStatus.AnsweredMultipleChoice
          ? step.otherSelected
          : false,
      freeText:
        step.status === StepStatus.AnsweredFreeResponse ? step.text : "",
      mcFreeText:
        step.status === StepStatus.AnsweredMultipleChoice
          ? step.freeResponseText
          : "",
      onToggle: (value: string) => handleMcToggle(idx, value),
      onOtherToggle: () => handleOtherToggle(idx),
      onFreeTextChange: (text: string) => handleFreeTextChange(idx, text),
      onMcFreeTextChange: (text: string) => handleMcFreeTextChange(idx, text),
    };
  };

  return {
    activeStep,
    setActiveStep,
    isSingleQuestion,
    isFirstStep,
    isLastStep,
    isStepAnswered: (idx: number) => isStepAnswered(answers, idx),
    isStepSkipped,
    isStepComplete,
    handleSkipStep,
    handleSubmit,
    getStepData,
  };
}
