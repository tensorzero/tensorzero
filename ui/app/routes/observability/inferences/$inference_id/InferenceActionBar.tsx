import { Suspense, useMemo } from "react";
import { Await } from "react-router";
import type { StoredInference, Input } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { useConfig, useFunctionConfig } from "~/context/config";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import { Skeleton } from "~/components/ui/skeleton";
import { ActionBarAsyncError } from "~/components/ui/error/ErrorContentPrimitives";
import { ActionBar } from "~/components/layout/ActionBar";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { CopyMessagesButton } from "~/components/inference/CopyMessagesButton";
import { TryWithVariantAction } from "./TryWithVariantAction";
import { HumanFeedbackAction } from "./HumanFeedbackAction";
import type { ModelInferencesData } from "./inference-data.server";

interface InferenceActionBarProps {
  inference: StoredInference;
  usedVariantsPromise: Promise<string[]>;
  hasDemonstrationPromise: Promise<boolean>;
  inputPromise: Promise<Input | undefined>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
  locationKey: string;
}

export function InferenceActionBar({
  inference,
  usedVariantsPromise,
  hasDemonstrationPromise,
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
        usedVariantsPromise={usedVariantsPromise}
        inputPromise={inputPromise}
        modelInferencesPromise={modelInferencesPromise}
        onFeedbackAdded={onFeedbackAdded}
      />
      <AddToDatasetButtonStreaming
        key={`dataset-${locationKey}`}
        inference={inference}
        hasDemonstrationPromise={hasDemonstrationPromise}
      />
      <HumanFeedbackAction
        key={`human-${locationKey}`}
        inference={inference}
        onFeedbackAdded={onFeedbackAdded}
      />
      <AskAutopilotButton
        message={`Inference ID: ${inference.inference_id}\n\n`}
      />
      {/* Keep at end of row — conditionally hidden, so trailing position avoids jitter */}
      <CopyMessagesButtonStreaming
        key={`copy-${locationKey}`}
        inference={inference}
        inputPromise={inputPromise}
      />
    </ActionBar>
  );
}

function AddToDatasetButtonStreaming({
  inference,
  hasDemonstrationPromise,
}: {
  inference: StoredInference;
  hasDemonstrationPromise: Promise<boolean>;
}) {
  return (
    <Suspense fallback={<Skeleton className="h-8 w-36" />}>
      <Await
        resolve={hasDemonstrationPromise}
        errorElement={<ActionBarAsyncError />}
      >
        {(hasDemonstration) => (
          <AddToDatasetButton
            inferenceId={inference.inference_id}
            functionName={inference.function_name}
            variantName={inference.variant_name}
            episodeId={inference.episode_id}
            hasDemonstration={hasDemonstration}
          />
        )}
      </Await>
    </Suspense>
  );
}

function CopyMessagesButtonStreaming({
  inference,
  inputPromise,
}: {
  inference: StoredInference;
  inputPromise: Promise<Input | undefined>;
}) {
  return (
    <Suspense fallback={<Skeleton className="h-8 w-36" />}>
      <Await resolve={inputPromise} errorElement={<ActionBarAsyncError />}>
        {(input) => (
          <CopyMessagesButton input={input} output={inference.output} />
        )}
      </Await>
    </Suspense>
  );
}

interface TryWithVariantActionStreamingProps {
  inference: StoredInference;
  usedVariantsPromise: Promise<string[]>;
  inputPromise: Promise<Input | undefined>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
}

function TryWithVariantActionStreaming({
  inference,
  usedVariantsPromise,
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
        usedVariantsPromise,
        inputPromise,
        modelInferencesPromise,
      ]).then(([usedVariants, input, modelInferences]) => ({
        usedVariants,
        input,
        inferenceUsage: getTotalInferenceUsage(modelInferences),
      })),
    [usedVariantsPromise, inputPromise, modelInferencesPromise],
  );

  return (
    <Suspense fallback={<Skeleton className="h-8 w-36" />}>
      <Await resolve={dataPromise} errorElement={<ActionBarAsyncError />}>
        {(data) => {
          const modelsSet = new Set([
            ...data.usedVariants,
            ...config.model_names,
          ]);
          const models = [...modelsSet].sort();
          const allOptions = isDefault ? models : variants;
          const options = allOptions.filter(
            (o) => o !== inference.variant_name,
          );

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
