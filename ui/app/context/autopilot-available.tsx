"use client";

/**
 * Autopilot Available Context Provider
 *
 * This module provides a React Context for checking if autopilot is available.
 * Autopilot is available when the gateway has TENSORZERO_AUTOPILOT_API_KEY configured.
 *
 * @module autopilot-available
 */

import { createContext, use } from "react";

const AutopilotAvailableContext = createContext(false);
AutopilotAvailableContext.displayName = "AutopilotAvailableContext";

/**
 * Hook to check if autopilot is available
 */
export function useAutopilotAvailable() {
  return use(AutopilotAvailableContext);
}

export function AutopilotAvailableProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: boolean;
}) {
  return (
    <AutopilotAvailableContext value={value}>
      {children}
    </AutopilotAvailableContext>
  );
}
