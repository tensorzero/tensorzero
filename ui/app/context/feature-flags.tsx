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

// eslint-disable-next-line @typescript-eslint/no-empty-object-type -- Will gain fields as new feature flags are added
export interface FeatureFlags {}

export const DEFAULT_FEATURE_FLAGS: FeatureFlags = {};

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
