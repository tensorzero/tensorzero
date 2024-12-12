import { ParsedInferenceRow } from "../clickhouse";
import { JsExposedEnv } from "../minijinja/pkg/minijinja_bindings";

// TODO: fill this out after we figure out the parameters
export async function start_sft_fireworks(
  modelName: string,
  accountId: string,
  trainInferences: ParsedInferenceRow[],
  valInferences: ParsedInferenceRow[],
  templateEnv: JsExposedEnv
) {}
