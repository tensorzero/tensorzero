import { useMutation, useQuery } from "@tanstack/react-query";
import {
  InferenceResponseSchema,
  type InferenceRequest,
} from "~/utils/tensorzero";
import type { DisplayInput } from "~/utils/clickhouse/common";
import { resolvedInputToTensorZeroInput } from "../api/tensorzero/inference";
import {
  ParsedDatasetRowSchema,
  type DatapointRow,
} from "~/utils/clickhouse/datasets";

export const useDatapoint = (datasetName: string, datapointId: string) => {
  return useQuery({
    queryKey: ["GET_DATAPOINT", datasetName, datapointId],
    queryFn: async () => {
      const response = await fetch(
        `/api/datasets/${datasetName}/datapoint/${datapointId}`,
      );

      if (!response.ok) {
        // TODO Figure out what to expose
        const errorData = await response.text();
        return Promise.reject({ status: response.status, errorData });
      }

      const data = await response.json();
      return ParsedDatasetRowSchema.parse(data.datapoint);
    },
    refetchOnWindowFocus: false,
  });
};

// TODO Temporarily using proxy to gateway versus the bespoke loaders

export const GATEWAY_PROXY_PATH = "/api/gateway";

export const useDataset = (datasetName?: string, functionName?: string) => {
  return useQuery({
    queryKey: [
      "GET_DATASET",
      "datasetName",
      datasetName,
      "functionName",
      functionName,
    ],
    queryFn: async ({ signal }) => {
      if (!datasetName) {
        return null;
      }

      const searchParams = new URLSearchParams();
      if (functionName) {
        searchParams.append("function", functionName);
      }

      const url = `${GATEWAY_PROXY_PATH}/datasets/${datasetName}/datapoints?${searchParams.toString()}`;
      const response = await fetch(url, {
        signal,
      });
      if (!response.ok) {
        // TODO Figure out what to expose
        const errorData = await response.text();
        return Promise.reject({ status: response.status, errorData });
      }

      const data = (await response.json()) as DatapointRow[]; // TODO Parse actual schema

      return data.filter((row) => row.function_name === functionName);
    },
    refetchOnWindowFocus: false,
  });
};

export const useRunVariantInference = (
  functionName: string,
  variantName: string,
) => {
  return useMutation({
    mutationKey: [
      "INFERENCE",
      "functionName",
      functionName,
      "variantName",
      variantName,
    ],
    mutationFn: async (input: DisplayInput) => {
      const response = await fetch(`${GATEWAY_PROXY_PATH}/inference`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: resolvedInputToTensorZeroInput(input),
          function_name: functionName,
          variant_name: variantName,
          dryrun: true,
        } satisfies InferenceRequest),
      });
      if (!response.ok) {
        // TODO Figure out what to expose
        const errorData = await response.text();
        return Promise.reject({ status: response.status, errorData });
      }

      return InferenceResponseSchema.parse(await response.json());
    },
  });
};
