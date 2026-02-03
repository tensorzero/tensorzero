"use client";

/**
 * Feature Flags Context Provider
 *
 * This module provides a React Context for managing feature flags.
 * Feature flags are loaded server-side from environment variables and
 * passed to the client via this context.
 *
 * @module feature-flags
 */

import { createContext, use } from "react";

export interface FeatureFlags {
  /** When TENSORZERO_UI_FORCE_CACHE_ON=1, sets `cache_options.enabled = "on"` on all inference calls */
  FORCE_CACHE_ON: boolean;
  /** When TENSORZERO_UI_FF_INTERRUPT_SESSION=1, enables the interrupt session button in autopilot sessions */
  FF_INTERRUPT_SESSION: boolean;
}

export const DEFAULT_FEATURE_FLAGS: FeatureFlags = {
  FORCE_CACHE_ON: false,
  FF_INTERRUPT_SESSION: false,
};

const FeatureFlagsContext = createContext<FeatureFlags>(DEFAULT_FEATURE_FLAGS);
FeatureFlagsContext.displayName = "FeatureFlagsContext";

/**
 * Hook to access feature flags
 */
export function useFeatureFlags(): FeatureFlags {
  return use(FeatureFlagsContext);
}

export function FeatureFlagsProvider({
  children,
  value,
}: {
  children: React.ReactNode;
  value: FeatureFlags;
}) {
  return <FeatureFlagsContext value={value}>{children}</FeatureFlagsContext>;
}
