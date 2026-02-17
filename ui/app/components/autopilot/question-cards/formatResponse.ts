import type {
  EventPayloadUserQuestion,
  UserQuestionAnswer,
} from "~/types/tensorzero";

/** Format a UserQuestionAnswer for display. */
export function formatResponse(
  response: UserQuestionAnswer | undefined,
  question: EventPayloadUserQuestion,
): string {
  if (!response) return "\u2014";
  switch (response.type) {
    case "multiple_choice":
      return response.selected
        .map((optId) => {
          if (question.type !== "multiple_choice") return optId;
          return question.options.find((o) => o.id === optId)?.label ?? optId;
        })
        .join(", ");
    case "free_response":
      return response.text || "\u2014";
    case "skipped":
      return "Skipped";
  }
}
