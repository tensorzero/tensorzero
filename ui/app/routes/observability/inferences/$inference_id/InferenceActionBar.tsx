import { useState, useEffect, useCallback } from "react";
import type { StoredInference, Input } from "~/types/tensorzero";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { useToast } from "~/hooks/use-toast";
import { useConfig, useFunctionConfig } from "~/context/config";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { getTotalInferenceUsage } from "~/utils/clickhouse/helpers";
import { ActionBar } from "~/components/layout/ActionBar";
import { TryWithSelect } from "~/components/inference/TryWithSelect";
import { AddToDatasetButton } from "~/components/dataset/AddToDatasetButton";
import { HumanFeedbackButton } from "~/components/feedback/HumanFeedbackButton";
import { HumanFeedbackModal } from "~/components/feedback/HumanFeedbackModal";
import { HumanFeedbackForm } from "~/components/feedback/HumanFeedbackForm";
import { DemonstrationFeedbackButton } from "~/components/feedback/DemonstrationFeedbackButton";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
  prepareDemonstrationFromVariantOutput,
} from "~/routes/api/tensorzero/inference.utils";
import { logger } from "~/utils/logger";
import type {
  ActionBarData,
  ModelInferencesData,
} from "./inference-data.server";

type ActionData =
  | { redirectTo: string; error?: never }
  | { error: string; redirectTo?: never };

interface InferenceActionBarProps {
  inference: StoredInference;
  actionBarData: ActionBarData;
  inputPromise: Promise<Input>;
  modelInferencesPromise: Promise<ModelInferencesData>;
  onFeedbackAdded: (redirectUrl?: string) => void;
}

/**
 * ActionBar component that handles TryWithSelect and VariantResponseModal.
 * Waits for input promise to resolve before enabling TryWithSelect functionality.
 */
export function InferenceActionBar({
  inference,
  actionBarData,
  inputPromise,
  modelInferencesPromise,
  onFeedbackAdded,
}: InferenceActionBarProps) {
  const { hasDemonstration, usedVariants } = actionBarData;
  const config = useConfig();
  const functionConfig = useFunctionConfig(inference.function_name);
  const variants = Object.keys(functionConfig?.variants || {});
  const isDefault = inference.function_name === DEFAULT_FUNCTION;

  const modelsSet = new Set<string>([...usedVariants, ...config.model_names]);
  const models = [...modelsSet].sort();
  const options = isDefault ? models : variants;

  const { toast } = useToast();

  // Track resolved state of input and modelInferences
  const [resolvedInput, setResolvedInput] = useState<Input | null>(null);
  const [resolvedModelInferences, setResolvedModelInferences] =
    useState<ModelInferencesData | null>(null);

  // Resolve input promise with cleanup to avoid state update on unmounted component
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

  // Resolve modelInferences promise with cleanup to avoid state update on unmounted component
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

  // Modal state
  const [openModal, setOpenModal] = useState<
    "human-feedback" | "variant-response" | null
  >(null);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

  // Human feedback fetcher
  const humanFeedbackFetcher = useFetcherWithReset<ActionData>();
  const {
    data: humanFeedbackData,
    state: humanFeedbackState,
    reset: resetHumanFeedbackFetcher,
  } = humanFeedbackFetcher;
  const humanFeedbackFormError =
    humanFeedbackState === "idle" ? (humanFeedbackData?.error ?? null) : null;

  useEffect(() => {
    const currentState = humanFeedbackState;
    const fetcherData = humanFeedbackData;
    if (currentState === "idle" && fetcherData?.redirectTo) {
      onFeedbackAdded(fetcherData.redirectTo);
      setOpenModal(null);
      resetHumanFeedbackFetcher();
    }
  }, [
    humanFeedbackData,
    humanFeedbackState,
    onFeedbackAdded,
    resetHumanFeedbackFetcher,
  ]);

  // Demonstration feedback fetcher
  const demonstrationFeedbackFetcher = useFetcherWithReset<ActionData>();
  const {
    data: demonstrationFeedbackData,
    state: demonstrationFeedbackState,
    reset: resetDemonstrationFeedbackFetcher,
  } = demonstrationFeedbackFetcher;
  const demonstrationFeedbackFormError =
    demonstrationFeedbackState === "idle"
      ? (demonstrationFeedbackData?.error ?? null)
      : null;

  useEffect(() => {
    const currentState = demonstrationFeedbackState;
    const fetcherData = demonstrationFeedbackData;
    if (currentState === "idle" && fetcherData?.redirectTo) {
      onFeedbackAdded(fetcherData.redirectTo);
      setOpenModal(null);
      setSelectedVariant(null);
      resetDemonstrationFeedbackFetcher();
    }
  }, [
    demonstrationFeedbackData,
    demonstrationFeedbackState,
    onFeedbackAdded,
    resetDemonstrationFeedbackFetcher,
  ]);

  // Variant inference fetcher
  const variantInferenceFetcher = useInferenceActionFetcher();
  const [lastRequestArgs, setLastRequestArgs] = useState<
    Parameters<typeof prepareInferenceActionRequest>[0] | null
  >(null);
  const variantSource = "inference";
  const variantInferenceIsLoading =
    openModal === "variant-response" &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const { submit } = variantInferenceFetcher;
  const processRequest = useCallback(
    (
      option: string,
      args: Parameters<typeof prepareInferenceActionRequest>[0],
    ) => {
      try {
        const request = prepareInferenceActionRequest(args);

        setSelectedVariant(option);
        setOpenModal("variant-response");
        setLastRequestArgs(args);

        try {
          void submit({ data: JSON.stringify(request) });
        } catch (stringifyError) {
          logger.error("Failed to stringify request:", stringifyError);
          toast.error({
            title: "Request Error",
            description: "Failed to prepare the request. Please try again.",
          });
          setSelectedVariant(null);
          setOpenModal(null);
        }
      } catch (error) {
        logger.error("Failed to prepare inference request:", error);

        let errorMessage = "Failed to prepare the request. Please try again.";
        if (error instanceof Error) {
          if (error.message.includes("Extra body is not supported")) {
            errorMessage =
              "This inference contains extra body parameters which are not supported in the UI.";
          } else if (error.message) {
            errorMessage = error.message;
          }
        }

        toast.error({
          title: "Request Preparation Error",
          description: errorMessage,
        });
      }
    },
    [submit, toast],
  );

  const onVariantSelect = useCallback(
    (variant: string) => {
      if (!resolvedInput) return;
      processRequest(variant, {
        resource: inference,
        input: resolvedInput,
        source: variantSource,
        variant,
      });
    },
    [inference, resolvedInput, processRequest],
  );

  const onModelSelect = useCallback(
    (model: string) => {
      if (!resolvedInput) return;
      processRequest(model, {
        resource: inference,
        input: resolvedInput,
        source: variantSource,
        model_name: model,
      });
    },
    [inference, resolvedInput, processRequest],
  );

  const handleRefresh = useCallback(() => {
    if (!lastRequestArgs) {
      return;
    }

    try {
      const request = prepareInferenceActionRequest(lastRequestArgs);
      request.cache_options = {
        ...request.cache_options,
        enabled: "write_only",
      };
      submit({ data: JSON.stringify(request) });
    } catch (error) {
      logger.error("Failed to prepare inference request for refresh:", error);
      toast.error({
        title: "Request Preparation Error",
        description: "Failed to refresh inference. Please try again.",
      });
    }
  }, [lastRequestArgs, submit, toast]);

  const onSelect = isDefault ? onModelSelect : onVariantSelect;

  // Disable TryWithSelect until input is loaded (needed for making requests)
  const tryWithSelectDisabled = !resolvedInput;

  return (
    <>
      <ActionBar>
        <TryWithSelect
          options={options}
          onSelect={onSelect}
          isLoading={variantInferenceIsLoading}
          isDefaultFunction={isDefault}
          disabled={tryWithSelectDisabled}
        />
        <AddToDatasetButton
          inferenceId={inference.inference_id}
          functionName={inference.function_name}
          variantName={inference.variant_name}
          episodeId={inference.episode_id}
          hasDemonstration={hasDemonstration}
        />
        <HumanFeedbackModal
          onOpenChange={(isOpen) => {
            if (humanFeedbackState !== "idle") {
              return;
            }
            if (!isOpen) {
              resetHumanFeedbackFetcher();
            }
            setOpenModal(isOpen ? "human-feedback" : null);
          }}
          isOpen={openModal === "human-feedback"}
          trigger={<HumanFeedbackButton />}
        >
          <humanFeedbackFetcher.Form method="post" action="/api/feedback">
            <HumanFeedbackForm
              inferenceId={inference.inference_id}
              inferenceOutput={inference.output}
              formError={humanFeedbackFormError}
              isSubmitting={
                humanFeedbackState === "submitting" ||
                humanFeedbackState === "loading"
              }
            />
          </humanFeedbackFetcher.Form>
        </HumanFeedbackModal>
      </ActionBar>

      {selectedVariant && (
        <VariantResponseModal
          isOpen={openModal === "variant-response"}
          isLoading={variantInferenceIsLoading}
          error={variantInferenceFetcher.error?.message}
          variantResponse={variantInferenceFetcher.data?.info ?? null}
          rawResponse={variantInferenceFetcher.data?.raw ?? null}
          onClose={() => {
            setOpenModal(null);
            setSelectedVariant(null);
            setLastRequestArgs(null);
          }}
          item={inference}
          inferenceUsage={inferenceUsage ?? undefined}
          selectedVariant={selectedVariant}
          source={variantSource}
          onRefresh={lastRequestArgs ? handleRefresh : null}
        >
          {variantInferenceFetcher.data?.info && (
            <demonstrationFeedbackFetcher.Form
              method="post"
              action="/api/feedback"
            >
              <input type="hidden" name="metricName" value="demonstration" />
              <input
                type="hidden"
                name="inferenceId"
                value={inference.inference_id}
              />
              <input
                type="hidden"
                name="value"
                value={JSON.stringify(
                  prepareDemonstrationFromVariantOutput(
                    variantInferenceFetcher.data.info,
                  ),
                )}
              />
              <DemonstrationFeedbackButton
                isSubmitting={demonstrationFeedbackState === "submitting"}
                submissionError={demonstrationFeedbackFormError}
              />
            </demonstrationFeedbackFetcher.Form>
          )}
        </VariantResponseModal>
      )}
    </>
  );
}
