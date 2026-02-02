/**
 * Feature Flags Module
 *
 * Feature flags are loaded server-side from environment variables and passed
 * to the client via React Context. Use `useFeatureFlags()` hook in components.
 *
 * For server-side code (loaders, actions), use `loadFeatureFlags()`.
 */

// Re-export the hook and types for convenient access
export {
  useFeatureFlags,
  type FeatureFlags,
  DEFAULT_FEATURE_FLAGS,
} from "~/context/feature-flags";

import type { FeatureFlags } from "~/context/feature-flags";

/**
 * Load feature flags from environment variables.
 * This should only be called server-side (in loaders/actions).
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
 * Only call server-side.
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
