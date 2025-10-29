import type { Config, FunctionConfig } from "~/types/tensorzero";
import { estimateTrackAndStopOptimalProbabilities } from "tensorzero-node";
import { getNativeDatabaseClient } from "~/utils/tensorzero/native_client.server";
import { logger } from "~/utils/logger";

/**
 * Computes optimal sampling probabilities for track_and_stop experimentation.
 *
 * This function implements a two-phase bandit algorithm:
 * 1. Nursery phase: Variants with insufficient samples receive equal probability (1/K)
 * 2. Bandit phase: Variants with sufficient samples are optimized using Thompson sampling
 *
 * The probabilities are scaled to ensure:
 * - All nursery variants together receive probability mass proportional to their count
 * - All bandit variants together receive probability mass proportional to their count
 * - Total probability sums to 1
 *
 * @param function_name - The name of the function being experimented on
 * @param function_config - The function configuration
 * @param config - The full TensorZero configuration (for metric definitions)
 * @returns A map of variant names to optimal probabilities, or undefined if computation fails
 *
 * @example
 * const optimalProbs = await computeTrackAndStopOptimalProbabilities(
 *   "my_function",
 *   functionConfig,
 *   config
 * );
 * // Returns: { "variant_a": 0.15, "variant_b": 0.55, "variant_c": 0.30 }
 */
export async function computeTrackAndStopOptimalProbabilities(
  function_name: string,
  function_config: FunctionConfig,
  config: Config,
): Promise<Record<string, number> | undefined> {
  const dbClient = await getNativeDatabaseClient();
  if (function_config.experimentation.type !== "track_and_stop") {
    return undefined;
  }

  try {
    const experimentationConfig = function_config.experimentation;
    const metric_config = config.metrics[experimentationConfig.metric];

    if (!metric_config) {
      return undefined;
    }

    const feedback = await dbClient.getFeedbackByVariant({
      metric_name: experimentationConfig.metric,
      function_name,
      variant_names: experimentationConfig.candidate_variants,
    });

    // Build feedback count map
    const feedbackCounts = new Map(
      feedback.map((f) => [f.variant_name, Number(f.count)]),
    );

    // Separate nursery and bandit variants
    const K = experimentationConfig.candidate_variants.length;
    const nurseryVariants = experimentationConfig.candidate_variants.filter(
      (v) =>
        (feedbackCounts.get(v) || 0) <
        Number(experimentationConfig.min_samples_per_variant),
    );
    const banditVariants = experimentationConfig.candidate_variants.filter(
      (v) =>
        (feedbackCounts.get(v) || 0) >=
        Number(experimentationConfig.min_samples_per_variant),
    );

    const num_bandit_variants = banditVariants.length;

    // Initialize probabilities
    const optimal_probabilities: Record<string, number> = {};

    // Assign 1/K to each nursery variant
    for (const variant of nurseryVariants) {
      optimal_probabilities[variant] = 1 / K;
    }

    // Compute and scale optimal probabilities for bandit variants
    if (num_bandit_variants > 0) {
      const banditFeedback = feedback.filter((f) =>
        banditVariants.includes(f.variant_name),
      );

      if (banditFeedback.length > 0) {
        const banditOptimalProbs = estimateTrackAndStopOptimalProbabilities({
          feedback: banditFeedback,
          epsilon: experimentationConfig.epsilon,
          metric_optimize: metric_config.optimize,
        });

        // Scale bandit probabilities by B/K
        const scale = num_bandit_variants / K;
        for (const [variant, prob] of Object.entries(banditOptimalProbs)) {
          optimal_probabilities[variant] = prob * scale;
        }
      }
    }

    return optimal_probabilities;
  } catch (error) {
    logger.error("Failed to compute optimal probabilities:", error);
    return undefined;
  }
}
