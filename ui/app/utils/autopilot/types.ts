export type AuthorizationStatus =
  | { type: "approved" }
  | { type: "rejected"; reason: string };

export type AuthorizationLoadingAction =
  | "approving"
  | "rejecting"
  | "approving_all";

export function approvedStatus(): AuthorizationStatus {
  return { type: "approved" };
}

export function rejectedStatus(reason: string): AuthorizationStatus {
  return { type: "rejected", reason };
}
