import type { SFTFormValues } from "~/routes/optimization/supervised-fine-tuning/types";
import { OpenAISFTJob } from "./openai";
import { FireworksSFTJob } from "./fireworks";
import type { SFTJob } from "./common";
import { TensorZeroClient, type OptimizerJobHandle } from "tensorzero-node";

const configPath = process.env.TENSORZERO_UI_CONFIG_PATH;
if (!configPath) {
  throw new Error("TENSORZERO_UI_CONFIG_PATH is not set");
}
const clickhouseUrl = process.env.CLICKHOUSE_URL;
if (!clickhouseUrl) {
  throw new Error("CLICKHOUSE_URL is not set");
}
const client = await TensorZeroClient.build(configPath, clickhouseUrl);
const useNativeSFT = process.env.TENSORZERO_UI_FF_USE_NATIVE_SFT === "1";

export function launch_sft_job(data: SFTFormValues): Promise<SFTJob> {
  if (useNativeSFT) {
    return launch_sft_job_native(data);
  } else {
    return launch_sft_job_ts(data);
  }
}

function launch_sft_job_ts(data: SFTFormValues): Promise<SFTJob> {
  switch (data.model.provider) {
    case "openai":
      return OpenAISFTJob.from_form_data(data);
    case "fireworks":
      return FireworksSFTJob.from_form_data(data);
    default:
      throw new Error("Invalid provider");
  }
}

function launch_sft_job_native(
  data: SFTFormValues,
): Promise<OptimizerJobHandle> {
  const job = client.experimentalStartOptimization({
    function_name: data.function,
    template_variant_name: data.variant,
    query_variant_name: null,
    filters: null,
    output_source: "foo",
  });
  return job;
}
