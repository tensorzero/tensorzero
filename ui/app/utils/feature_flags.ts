interface FeatureFlags {
  /// When set, sets `cache_options.enabled = "on"` on all inference calls
  /// Normally, we leave this unset, which uses the TensorZero default of 'write_only'
  /// This is used by e2e tests to allow us to populate the model inference cache
  /// from regen-fixtures without trampling existing entries, and then to use the cached
  /// entries from the normal ui e2e tests
  FORCE_CACHE_ON: boolean;
}

/**
 * Get feature flags for the application.
 * This can be accessed from the client.
 * @returns FeatureFlags
 */
export function getFeatureFlags(): FeatureFlags {
  const FORCE_CACHE_ON = import.meta.env.VITE_TENSORZERO_FORCE_CACHE_ON === "1";
  return {
    FORCE_CACHE_ON,
  };
}

/// Returns an object containing extra parameters that should be passed to
/// inference calls on our TensorZero client
export function getExtraInferenceOptions(): object {
  if (getFeatureFlags().FORCE_CACHE_ON) {
    return {
      dryrun: false,
      cache_options: {
        enabled: "on",
      },
    };
  }
  return {};
}
