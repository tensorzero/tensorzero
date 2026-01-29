import { useAsyncError } from "react-router";
import type { StoredInference } from "~/types/tensorzero";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import BasicInfo from "./InferenceBasicInfo";
import type { ModelInferencesData } from "./inference-data.server";

// Content
export function BasicInfoContent({
  inference,
  modelInferences,
}: {
  inference: StoredInference;
  modelInferences: ModelInferencesData;
}) {
  const inferenceUsage = getTotalInferenceUsage(modelInferences);
  return (
    <BasicInfo
      inference={inference}
      inferenceUsage={inferenceUsage}
      modelInferences={modelInferences}
    />
  );
}

// Error
export function BasicInfoError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load inference details";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  );
}
