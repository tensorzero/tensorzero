import type { UserQuestionAnswer } from "~/types/tensorzero";

/**
 * Returns true if at least one response is not skipped.
 */
export function hasAnsweredResponse(
  responses: Record<string, UserQuestionAnswer> | undefined,
): boolean {
  if (!responses) return false;
  return Object.values(responses).some((r) => r.type !== "skipped");
}
