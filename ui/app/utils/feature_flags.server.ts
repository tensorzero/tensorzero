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
  return {
    FORCE_CACHE_ON: process.env.TENSORZERO_UI_FORCE_CACHE_ON === "1",
    FF_INTERRUPT_SESSION:
      process.env.TENSORZERO_UI_FF_INTERRUPT_SESSION === "1",
  };
}

interface ExtraInferenceOptions {
  cache_options?: {
    enabled: "on" | "off" | "write_only";
    max_age_s: number | null;
  };
}

/**
 * Returns extra parameters for inference calls.
 */
export function getExtraInferenceOptions(): ExtraInferenceOptions {
  if (loadFeatureFlags().FORCE_CACHE_ON) {
    return {
      cache_options: {
        enabled: "on",
        max_age_s: null,
      },
    };
  }
  return {};
}
