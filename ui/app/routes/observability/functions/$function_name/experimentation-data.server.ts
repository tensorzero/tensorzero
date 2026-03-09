import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type { FunctionConfig, TimeWindow } from "~/types/tensorzero";

export type ExperimentationSectionData = {
  feedback_timeseries:
    | Awaited<
        ReturnType<
          ReturnType<
            typeof getTensorZeroClient
          >["getCumulativeFeedbackTimeseries"]
        >
      >
    | undefined;
  variant_sampling_probabilities: Awaited<
    ReturnType<
      ReturnType<typeof getTensorZeroClient>["getVariantSamplingProbabilities"]
    >
  >["probabilities"];
};

export async function fetchExperimentationSectionData(params: {
  function_name: string;
  function_config: FunctionConfig;
  time_granularity: TimeWindow;
  namespace: string | undefined;
}): Promise<ExperimentationSectionData> {
  const { function_name, function_config, time_granularity, namespace } =
    params;

  const client = getTensorZeroClient();

  // Resolve the active experimentation config (namespace-specific or base)
  const namespaces = function_config.experimentation.namespaces;
  const activeConfig =
    namespace && Object.hasOwn(namespaces, namespace)
      ? namespaces[namespace]
      : function_config.experimentation.base;

  const feedbackParams =
    activeConfig.type === "adaptive"
      ? {
          metric_name: activeConfig.metric,
          variant_names: activeConfig.candidate_variants,
        }
      : null;

  const tag = namespace ? `tensorzero::namespace::${namespace}` : undefined;

  const [feedback_timeseries, variant_sampling_probabilities] =
    await Promise.all([
      feedbackParams
        ? client.getCumulativeFeedbackTimeseries({
            function_name,
            ...feedbackParams,
            time_window: time_granularity,
            max_periods: 10,
            tag,
          })
        : Promise.resolve(undefined),
      client
        .getVariantSamplingProbabilities(function_name, namespace)
        .then((response) => response.probabilities),
    ]);

  return { feedback_timeseries, variant_sampling_probabilities };
}
