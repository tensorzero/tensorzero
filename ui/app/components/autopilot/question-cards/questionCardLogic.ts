import type {
  EventPayloadUserQuestion,
  UserQuestionAnswer,
} from "~/types/tensorzero";

export enum StepStatus {
  Unanswered = "unanswered",
  Skipped = "skipped",
  AnsweredMultipleChoice = "answered_multiple_choice",
  AnsweredFreeResponse = "answered_free_response",
}

export type StepAnswer =
  | { status: StepStatus.Unanswered }
  | { status: StepStatus.Skipped }
  | {
      status: StepStatus.AnsweredMultipleChoice;
      selected: Set<string>;
      otherSelected: boolean;
      freeResponseText: string;
    }
  | { status: StepStatus.AnsweredFreeResponse; text: string };

export type StepAnswers = Map<number, StepAnswer>;

export function getStep(answers: StepAnswers, idx: number): StepAnswer {
  return answers.get(idx) ?? { status: StepStatus.Unanswered };
}

export function applyMcToggle(
  answers: StepAnswers,
  questionIndex: number,
  value: string,
  question: EventPayloadUserQuestion,
): StepAnswers {
  if (question.type !== "multiple_choice") return answers;

  const step = getStep(answers, questionIndex);
  const current =
    step.status === StepStatus.AnsweredMultipleChoice
      ? step.selected
      : new Set<string>();
  const currentFreeText =
    step.status === StepStatus.AnsweredMultipleChoice
      ? step.freeResponseText
      : "";

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

  const next = new Map(answers);
  if (question.multi_select) {
    const currentOther =
      step.status === StepStatus.AnsweredMultipleChoice
        ? step.otherSelected
        : false;
    next.set(questionIndex, {
      status: StepStatus.AnsweredMultipleChoice,
      selected: updated,
      otherSelected: currentOther,
      freeResponseText: currentFreeText,
    });
  } else {
    // Single-select: choosing a predefined option deselects "Other" and clears free text
    next.set(questionIndex, {
      status: StepStatus.AnsweredMultipleChoice,
      selected: updated,
      otherSelected: false,
      freeResponseText: "",
    });
  }
  return next;
}

export function applyOtherToggle(
  answers: StepAnswers,
  questionIndex: number,
  question: EventPayloadUserQuestion,
): StepAnswers {
  if (question.type !== "multiple_choice") return answers;

  const step = getStep(answers, questionIndex);
  const wasOtherSelected =
    step.status === StepStatus.AnsweredMultipleChoice
      ? step.otherSelected
      : false;
  const currentFreeText =
    step.status === StepStatus.AnsweredMultipleChoice
      ? step.freeResponseText
      : "";

  const next = new Map(answers);
  if (question.multi_select) {
    const current =
      step.status === StepStatus.AnsweredMultipleChoice
        ? step.selected
        : new Set<string>();
    next.set(questionIndex, {
      status: StepStatus.AnsweredMultipleChoice,
      selected: current,
      otherSelected: !wasOtherSelected,
      freeResponseText: wasOtherSelected ? "" : currentFreeText,
    });
  } else {
    // Single-select: selecting "Other" deselects predefined options
    next.set(questionIndex, {
      status: StepStatus.AnsweredMultipleChoice,
      selected: new Set<string>(),
      otherSelected: !wasOtherSelected,
      freeResponseText: wasOtherSelected ? "" : currentFreeText,
    });
  }
  return next;
}

export function applyMcFreeTextChange(
  answers: StepAnswers,
  questionIndex: number,
  text: string,
): StepAnswers {
  const step = getStep(answers, questionIndex);
  const current =
    step.status === StepStatus.AnsweredMultipleChoice
      ? step.selected
      : new Set<string>();
  const otherSelected =
    step.status === StepStatus.AnsweredMultipleChoice
      ? step.otherSelected
      : false;

  const next = new Map(answers);
  next.set(questionIndex, {
    status: StepStatus.AnsweredMultipleChoice,
    selected: current,
    otherSelected,
    freeResponseText: text,
  });
  return next;
}

export function isStepAnswered(answers: StepAnswers, idx: number): boolean {
  const step = getStep(answers, idx);
  switch (step.status) {
    case StepStatus.AnsweredMultipleChoice:
      return (
        step.selected.size > 0 ||
        (step.otherSelected && step.freeResponseText.trim().length > 0)
      );
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
}

export function buildResponses(
  questions: EventPayloadUserQuestion[],
  finalAnswers: StepAnswers,
): Record<string, UserQuestionAnswer> {
  const responses: Record<string, UserQuestionAnswer> = {};
  questions.forEach((question, idx) => {
    const step = getStep(finalAnswers, idx);
    switch (step.status) {
      case StepStatus.Skipped:
      case StepStatus.Unanswered:
        responses[question.id] = { type: "skipped" };
        break;
      case StepStatus.AnsweredMultipleChoice: {
        const mcAnswer: UserQuestionAnswer & { type: "multiple_choice" } = {
          type: "multiple_choice",
          selected: Array.from(step.selected),
        };
        if (step.freeResponseText.trim()) {
          mcAnswer.free_response_text = step.freeResponseText.trim();
        }
        responses[question.id] = mcAnswer;
        break;
      }
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
}

export function markUnansweredAsSkipped(
  questions: EventPayloadUserQuestion[],
  base: StepAnswers,
): StepAnswers {
  const result = new Map(base);
  questions.forEach((_, idx) => {
    const step = getStep(base, idx);
    const answered =
      (step.status === StepStatus.AnsweredMultipleChoice &&
        (step.selected.size > 0 ||
          (step.otherSelected && step.freeResponseText.trim().length > 0))) ||
      (step.status === StepStatus.AnsweredFreeResponse &&
        step.text.trim().length > 0);
    if (!answered && step.status !== StepStatus.Skipped) {
      result.set(idx, { status: StepStatus.Skipped });
    }
  });
  return result;
}
