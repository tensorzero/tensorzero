import type {
  EventPayloadUserQuestion,
  UserQuestionAnswer,
} from "~/types/tensorzero";

export function formatAnswer(
  answer: UserQuestionAnswer | undefined,
  question: EventPayloadUserQuestion,
): string {
  if (!answer) return "\u2014";
  switch (answer.type) {
    case "multiple_choice":
      return answer.selected
        .map((optId) => {
          if (question.type !== "multiple_choice") return optId;
          return question.options.find((o) => o.id === optId)?.label ?? optId;
        })
        .join(", ");
    case "free_response":
      return answer.text || "\u2014";
    case "skipped":
      return "Skipped";
  }
}
