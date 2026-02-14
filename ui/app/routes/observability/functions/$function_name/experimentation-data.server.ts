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
}): Promise<ExperimentationSectionData> {
  const { function_name, function_config, time_granularity } = params;

  const client = getTensorZeroClient();

  const feedbackParams =
    function_config.experimentation.base.type === "track_and_stop"
      ? {
          metric_name: function_config.experimentation.base.metric,
          variant_names: function_config.experimentation.base.candidate_variants,
        }
      : null;

  const [feedback_timeseries, variant_sampling_probabilities] =
    await Promise.all([
      feedbackParams
        ? client.getCumulativeFeedbackTimeseries({
            function_name,
            ...feedbackParams,
            time_window: time_granularity,
            max_periods: 10,
          })
        : Promise.resolve(undefined),
      client
        .getVariantSamplingProbabilities(function_name)
        .then((response) => response.probabilities),
    ]);

  return { feedback_timeseries, variant_sampling_probabilities };
}
