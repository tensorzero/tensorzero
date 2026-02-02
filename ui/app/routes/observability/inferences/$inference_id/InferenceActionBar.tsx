import { Suspense, useState, useEffect } from "react";
import { Await, useAsyncError } from "react-router";
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

// Main Export - Self-contained with Suspense/Await
interface InferenceActionBarProps {
  inference: StoredInference;
  actionBarData: Promise<ActionBarData>;
  inputPromise: Promise<Input>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
}

export function InferenceActionBar({
  inference,
  actionBarData,
  inputPromise,
  modelInferencesPromise,
  onFeedbackAdded,
}: InferenceActionBarProps) {
  return (
    <Suspense fallback={<InferenceActionBarSkeleton />}>
      <Await resolve={actionBarData} errorElement={<InferenceActionBarError />}>
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

// Skeleton
function InferenceActionBarSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-36" />
      <Skeleton className="h-8 w-8" />
    </div>
  );
}

// Error
function InferenceActionBarError() {
  const error = useAsyncError();
  const message =
    error instanceof Error ? error.message : "Failed to load actions";

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  );
}

// Content - Composes sub-actions
interface InferenceInferenceActionBarContentProps {
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
}: InferenceInferenceActionBarContentProps) {
  const { hasDemonstration, usedVariants } = actionBarData;
  const config = useConfig();
  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});
  const isDefault = inference.function_name === DEFAULT_FUNCTION;

  const modelsSet = new Set<string>([...usedVariants, ...config.model_names]);
  const models = [...modelsSet].sort();
  const options = isDefault ? models : variants;

  // Resolve promises for TryWithVariant
  const [resolvedInput, setResolvedInput] = useState<Input | null>(null);
  const [resolvedModelInferences, setResolvedModelInferences] =
    useState<ModelInferencesData | null>(null);

  useEffect(() => {
    let cancelled = false;
    inputPromise
      .then((input) => {
        if (!cancelled) setResolvedInput(input);
      })
      .catch(() => {
        if (!cancelled) setResolvedInput(null);
      });
    return () => {
      cancelled = true;
    };
  }, [inputPromise]);

  useEffect(() => {
    let cancelled = false;
    modelInferencesPromise
      .then((mi) => {
        if (!cancelled) setResolvedModelInferences(mi);
      })
      .catch(() => {
        if (!cancelled) setResolvedModelInferences(null);
      });
    return () => {
      cancelled = true;
    };
  }, [modelInferencesPromise]);

  const inferenceUsage = resolvedModelInferences
    ? getTotalInferenceUsage(resolvedModelInferences)
    : null;

  return (
    <ActionBar>
      <TryWithVariantAction
        inference={inference}
        options={options}
        isDefault={isDefault}
        resolvedInput={resolvedInput}
        inferenceUsage={inferenceUsage}
        onFeedbackAdded={onFeedbackAdded}
      />
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
