import { Suspense, useMemo } from "react";
import { Await } from "react-router";
import type { StoredInference, Input } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { useConfig, useFunctionConfig } from "~/context/config";
import {
  getTotalInferenceUsage,
  type InferenceUsage,
} from "~/utils/clickhouse/helpers";
import { Skeleton } from "~/components/ui/skeleton";
import { ActionBar } from "~/components/layout/ActionBar";
import { ActionBarAsyncError } from "~/components/ui/error/ErrorContentPrimitives";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { TryWithSelect } from "~/components/inference/TryWithSelect";
import { TryWithVariantAction } from "./TryWithVariantAction";
import { HumanFeedbackAction } from "./HumanFeedbackAction";
import type {
  ActionBarData,
  ModelInferencesData,
} from "./inference-data.server";

interface InferenceActionBarProps {
  inference: StoredInference;
  actionBarData: Promise<ActionBarData>;
  inputPromise: Promise<Input>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
  locationKey: string;
}

export function InferenceActionBar({
  inference,
  actionBarData,
  inputPromise,
  modelInferencesPromise,
  onFeedbackAdded,
  locationKey,
}: InferenceActionBarProps) {
  return (
    <Suspense key={locationKey} fallback={<InferenceActionBarSkeleton />}>
      <Await resolve={actionBarData} errorElement={<ActionBarAsyncError />}>
        {(resolvedActionBarData) => (
          <InferenceActionBarContent
            inference={inference}
            actionBarData={resolvedActionBarData}
            inputPromise={inputPromise}
            modelInferencesPromise={modelInferencesPromise}
            onFeedbackAdded={onFeedbackAdded}
          />
        )}
      </Await>
    </Suspense>
  );
}

function InferenceActionBarSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-8" />
    </div>
  );
}

interface InferenceActionBarContentProps {
  inference: StoredInference;
  actionBarData: ActionBarData;
  inputPromise: Promise<Input>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
}

function InferenceActionBarContent({
  inference,
  actionBarData,
  inputPromise,
  modelInferencesPromise,
  onFeedbackAdded,
}: InferenceActionBarContentProps) {
  const { hasDemonstration, usedVariants } = actionBarData;
  const config = useConfig();
  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});
  const isDefault = inference.function_name === DEFAULT_FUNCTION;

  const modelsSet = new Set<string>([...usedVariants, ...config.model_names]);
  const models = [...modelsSet].sort();
  const options = isDefault ? models : variants;

  // Combine promises for TryWithVariant - resolves when both input and model inferences are ready
  const tryWithVariantDataPromise = useMemo(
    () =>
      Promise.all([inputPromise, modelInferencesPromise]).then(
        ([input, modelInferences]) => ({
          input,
          inferenceUsage: getTotalInferenceUsage(modelInferences),
        }),
      ),
    [inputPromise, modelInferencesPromise],
  );

  return (
    <ActionBar>
      <Suspense
        fallback={
          <TryWithSelect
            options={options}
            onSelect={() => {}}
            isLoading={false}
            isDefaultFunction={isDefault}
            disabled
          />
        }
      >
        <Await
          resolve={tryWithVariantDataPromise}
          errorElement={
            <TryWithSelect
              options={options}
              onSelect={() => {}}
              isLoading={false}
              isDefaultFunction={isDefault}
              disabled
            />
          }
        >
          {(data: { input: Input; inferenceUsage: InferenceUsage }) => (
            <TryWithVariantAction
              inference={inference}
              options={options}
              isDefault={isDefault}
              input={data.input}
              inferenceUsage={data.inferenceUsage}
              onFeedbackAdded={onFeedbackAdded}
            />
          )}
        </Await>
      </Suspense>
      <AddToDatasetButton
        inferenceId={inference.inference_id}
        functionName={inference.function_name}
        variantName={inference.variant_name}
        episodeId={inference.episode_id}
        hasDemonstration={hasDemonstration}
      />
      <HumanFeedbackAction
        inference={inference}
        onFeedbackAdded={onFeedbackAdded}
      />
    </ActionBar>
  );
}
