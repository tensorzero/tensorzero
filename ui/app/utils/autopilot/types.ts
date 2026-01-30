/**
 * Shared types for Autopilot tool call authorization.
 */

/**
 * Authorization status sent to the server.
 */
export type AuthorizationStatus =
  | { type: "approved" }
  | { type: "rejected"; reason: string };

/**
 * UI loading state for authorization actions.
 */
export type AuthorizationLoadingAction =
  | "approving"
  | "rejecting"
  | "approving_all";

/**
 * Helper to create an approved status.
 */
export function approvedStatus(): AuthorizationStatus {
  return { type: "approved" };
}

/**
 * Helper to create a rejected status with a reason.
 */
export function rejectedStatus(reason: string): AuthorizationStatus {
  return { type: "rejected", reason };
}
