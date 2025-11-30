import { getTensorZeroClient } from "~/utils/tensorzero.server";
import type {
  ContentBlockChatOutput,
  CreateDatapointRequest,
  Input,
  JsonInferenceOutput,
} from "~/types/tensorzero";

export interface CreateDatapointParams {
  datasetName: string;
  functionName: string;
  functionType: "chat" | "json";
  input: Input;
  output?: ContentBlockChatOutput[] | JsonInferenceOutput;
  tags?: Record<string, string>;
  name?: string;
}

/**
 * Creates a new datapoint in the specified dataset.
 *
 * @param params - The parameters for creating the datapoint
 * @returns The ID of the newly created datapoint
 */
export async function createDatapoint(
  params: CreateDatapointParams,
): Promise<{ id: string }> {
  const { datasetName, functionName, functionType, input, output, tags, name } =
    params;

  let datapointRequest: CreateDatapointRequest;

  if (functionType === "chat") {
    datapointRequest = {
      type: "chat",
      function_name: functionName,
      input,
      output: output as ContentBlockChatOutput[] | undefined,
      tags,
      name,
      // These fields are required but we use defaults for manual creation
      provider_tools: [],
    };
  } else {
    // functionType === "json"
    datapointRequest = {
      type: "json",
      function_name: functionName,
      input,
      output: output
        ? {
            raw: (output as JsonInferenceOutput).raw ?? null,
          }
        : undefined,
      tags,
      name,
    };
  }

  const response = await getTensorZeroClient().createDatapoints(datasetName, {
    datapoints: [datapointRequest],
  });

  const id = response.ids[0];
  if (!id) {
    throw new Error("Failed to create datapoint: no ID returned");
  }

  return { id };
}
