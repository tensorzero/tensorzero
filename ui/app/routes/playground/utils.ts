import type { DisplayInput } from "~/utils/clickhouse/common";
import type {
  Datapoint as TensorZeroDatapoint,
  InferenceResponse,
  FunctionConfig,
} from "tensorzero-node";
import { prepareInferenceActionRequest } from "../api/tensorzero/inference.utils";

export function refreshClientInference(
  setPromise: (
    outerKey: string,
    innerKey: string,
    promise: Promise<InferenceResponse>,
  ) => void,
  input: DisplayInput,
  datapoint: TensorZeroDatapoint,
  variantName: string,
  functionName: string,
  functionConfig: FunctionConfig,
) {
  const request = prepareInferenceActionRequest({
    source: "clickhouse_datapoint",
    input,
    functionName,
    variant: variantName,
    tool_params:
      datapoint?.type === "chat"
        ? (datapoint.tool_params ?? undefined)
        : undefined,
    output_schema: datapoint?.type === "json" ? datapoint.output_schema : null,
    cache_options: {
      max_age_s: null,
      enabled: "off",
    },
    dryrun: true,
    functionConfig,
  });
  // The API endpoint takes form data so we need to stringify it and send as data
  const formData = new FormData();
  formData.append("data", JSON.stringify(request));
  const responsePromise = async () => {
    const response = await fetch("/api/tensorzero/inference", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }
    return data;
  };
  setPromise(variantName, datapoint.id, responsePromise());
}
