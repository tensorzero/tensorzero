/**
 * Feature Flags Server Module
 *
 * This file is server-only (.server.ts convention in React Router).
 * It loads feature flags from environment variables.
 *
 * For client-side access, use `useFeatureFlags()` hook from `~/context/feature-flags`.
 */

import type { FeatureFlags } from "~/context/feature-flags";

export type { FeatureFlags } from "~/context/feature-flags";

/**
 * Load feature flags from environment variables.
 */
export function loadFeatureFlags(): FeatureFlags {
  return {};
}
