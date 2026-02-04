import { Suspense, useMemo } from "react";
import { Await } from "react-router";
import type { StoredInference, Input } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { useConfig, useFunctionConfig } from "~/context/config";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import { Skeleton } from "~/components/ui/skeleton";
import { ActionBar } from "~/components/layout/ActionBar";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { TryWithVariantAction } from "./TryWithVariantAction";
import { HumanFeedbackAction } from "./HumanFeedbackAction";
import type {
  ActionBarData,
  ModelInferencesData,
} from "./inference-data.server";

interface InferenceActionBarProps {
  inference: StoredInference;
  actionBarDataPromise: Promise<ActionBarData>;
  inputPromise: Promise<Input>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
  locationKey: string;
}

export function InferenceActionBar({
  inference,
  actionBarDataPromise,
  inputPromise,
  modelInferencesPromise,
  onFeedbackAdded,
  locationKey,
}: InferenceActionBarProps) {
  return (
    <ActionBar>
      <TryWithVariantActionStreaming
        key={`try-${locationKey}`}
        inference={inference}
        actionBarDataPromise={actionBarDataPromise}
        inputPromise={inputPromise}
        modelInferencesPromise={modelInferencesPromise}
        onFeedbackAdded={onFeedbackAdded}
      />
      <AddToDatasetButtonStreaming
        key={`dataset-${locationKey}`}
        inference={inference}
        actionBarDataPromise={actionBarDataPromise}
      />
      <HumanFeedbackAction
        inference={inference}
        onFeedbackAdded={onFeedbackAdded}
      />
    </ActionBar>
  );
}

interface TryWithVariantActionStreamingProps {
  inference: StoredInference;
  actionBarDataPromise: Promise<ActionBarData>;
  inputPromise: Promise<Input>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
}

function TryWithVariantActionStreaming({
  inference,
  actionBarDataPromise,
  inputPromise,
  modelInferencesPromise,
  onFeedbackAdded,
}: TryWithVariantActionStreamingProps) {
  const config = useConfig();
  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});
  const isDefault = inference.function_name === DEFAULT_FUNCTION;

  const dataPromise = useMemo(
    () =>
      Promise.all([
        actionBarDataPromise,
        inputPromise,
        modelInferencesPromise,
      ]).then(([actionBarData, input, modelInferences]) => ({
        usedVariants: actionBarData.usedVariants,
        input,
        inferenceUsage: getTotalInferenceUsage(modelInferences),
      })),
    [actionBarDataPromise, inputPromise, modelInferencesPromise],
  );

  return (
    <Suspense fallback={<Skeleton className="h-8 w-36" />}>
      <Await
        resolve={dataPromise}
        errorElement={<Skeleton className="h-8 w-36" />}
      >
        {(data) => {
          const modelsSet = new Set([
            ...data.usedVariants,
            ...config.model_names,
          ]);
          const models = [...modelsSet].sort();
          const options = isDefault ? models : variants;

          return (
            <TryWithVariantAction
              inference={inference}
              options={options}
              isDefault={isDefault}
              input={data.input}
              inferenceUsage={data.inferenceUsage}
              onFeedbackAdded={onFeedbackAdded}
            />
          );
        }}
      </Await>
    </Suspense>
  );
}

interface AddToDatasetButtonStreamingProps {
  inference: StoredInference;
  actionBarDataPromise: Promise<ActionBarData>;
}

function AddToDatasetButtonStreaming({
  inference,
  actionBarDataPromise,
}: AddToDatasetButtonStreamingProps) {
  return (
    <Suspense fallback={<Skeleton className="h-8 w-36" />}>
      <Await
        resolve={actionBarDataPromise}
        errorElement={<Skeleton className="h-8 w-36" />}
      >
        {(actionBarData) => (
          <AddToDatasetButton
            inferenceId={inference.inference_id}
            functionName={inference.function_name}
            variantName={inference.variant_name}
            episodeId={inference.episode_id}
            hasDemonstration={actionBarData.hasDemonstration}
          />
        )}
      </Await>
    </Suspense>
  );
}
