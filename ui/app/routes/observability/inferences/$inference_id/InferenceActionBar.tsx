import { Suspense, useEffect, useMemo } from "react";
import { Await } from "react-router";
import { ClipboardIcon } from "lucide-react";
import type { StoredInference, Input } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { useConfig, useFunctionConfig } from "~/context/config";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import { useCopy } from "~/hooks/use-copy";
import { useToast } from "~/hooks/use-toast";
import { Skeleton } from "~/components/ui/skeleton";
import { ActionBarAsyncError } from "~/components/ui/error/ErrorContentPrimitives";
import { ActionBar } from "~/components/layout/ActionBar";
import { Button, ButtonIcon } from "~/components/ui/button";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { AskAutopilotButton } from "~/components/autopilot/AskAutopilotButton";
import { serializeMessages } from "~/utils/conversation-serializer";
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
      <CopyMessagesButton
        key={`copy-${locationKey}`}
        inference={inference}
        inputPromise={inputPromise}
      />
      <AskAutopilotButton
        message={`Inference ID: ${inference.inference_id}\n\n`}
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

function CopyMessagesButton({
  inference,
  inputPromise,
}: {
  inference: StoredInference;
  inputPromise: Promise<Input | undefined>;
}) {
  const { copy, didCopy } = useCopy();
  const { toast } = useToast();

  useEffect(() => {
    if (didCopy) {
      toast.success({ title: "Copied messages to clipboard" });
    }
  }, [didCopy, toast]);

  const handleCopy = async () => {
    const input = await inputPromise;
    const serialized = serializeMessages(input, inference);
    await copy(serialized);
  };

  return (
    <Button variant="outline" size="sm" className="w-fit" onClick={handleCopy}>
      <ButtonIcon as={ClipboardIcon} variant="tertiary" />
      Copy Messages
    </Button>
  );
}
