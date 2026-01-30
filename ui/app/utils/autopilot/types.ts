import type { GatewayToolCallAuthorizationStatus } from "~/types/tensorzero";

export type { GatewayToolCallAuthorizationStatus as AuthorizationStatus };

export type AuthorizationLoadingAction =
  | "approving"
  | "rejecting"
  | "approving_all";

export function approvedStatus(): GatewayToolCallAuthorizationStatus {
  return { type: "approved" };
}

export function rejectedStatus(
  reason: string,
): GatewayToolCallAuthorizationStatus {
  return { type: "rejected", reason };
}
