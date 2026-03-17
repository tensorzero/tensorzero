import { useState, useEffect, useCallback } from "react";
import type { StoredInference, Input } from "~/types/tensorzero";
import type { InferenceUsage } from "~/utils/clickhouse/helpers";
import type { FeedbackActionData } from "~/routes/api/feedback/route";
import { useToast } from "~/hooks/use-toast";
import { useFetcherWithReset } from "~/hooks/use-fetcher-with-reset";
import { TryWithSelect } from "~/components/inference/TryWithSelect";
import { DemonstrationFeedbackButton } from "~/components/feedback/DemonstrationFeedbackButton";
import { VariantResponseModal } from "~/components/inference/VariantResponseModal";
import {
  prepareInferenceActionRequest,
  useInferenceActionFetcher,
  prepareDemonstrationFromVariantOutput,
} from "~/routes/api/tensorzero/inference.utils";
import { logger } from "~/utils/logger";

interface TryWithVariantActionProps {
  inference: StoredInference;
  options: string[];
  isDefault: boolean;
  input: Input | undefined;
  inferenceUsage: InferenceUsage;
  onFeedbackAdded: (redirectUrl?: string) => void;
}

export function TryWithVariantAction({
  inference,
  options,
  isDefault,
  input,
  inferenceUsage,
  onFeedbackAdded,
}: TryWithVariantActionProps) {
  const { toast } = useToast();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);
  const [lastRequestArgs, setLastRequestArgs] = useState<
    Parameters<typeof prepareInferenceActionRequest>[0] | null
  >(null);

  const variantInferenceFetcher = useInferenceActionFetcher();
  const isLoading =
    isModalOpen &&
    (variantInferenceFetcher.state === "submitting" ||
      variantInferenceFetcher.state === "loading");

  const demonstrationFeedbackFetcher =
    useFetcherWithReset<FeedbackActionData>();
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
    if (
      demonstrationFeedbackState === "idle" &&
      demonstrationFeedbackData?.redirectTo
    ) {
      onFeedbackAdded(demonstrationFeedbackData.redirectTo);
      setIsModalOpen(false);
      setSelectedVariant(null);
      resetDemonstrationFeedbackFetcher();
    }
  }, [
    demonstrationFeedbackData,
    demonstrationFeedbackState,
    onFeedbackAdded,
    resetDemonstrationFeedbackFetcher,
  ]);

  const { submit } = variantInferenceFetcher;

  const processRequest = useCallback(
    (
      option: string,
      args: Parameters<typeof prepareInferenceActionRequest>[0],
    ) => {
      try {
        const request = prepareInferenceActionRequest(args);

        setSelectedVariant(option);
        setIsModalOpen(true);
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
          setIsModalOpen(false);
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

  const handleSelect = useCallback(
    (option: string) => {
      if (!input) {
        toast.error({
          title: "Input Unavailable",
          description: "Cannot retry inference without input data.",
        });
        return;
      }
      const args = isDefault
        ? {
            resource: inference,
            input,
            source: "inference" as const,
            model_name: option,
          }
        : {
            resource: inference,
            input,
            source: "inference" as const,
            variant: option,
          };

      processRequest(option, args);
    },
    [inference, input, isDefault, processRequest, toast],
  );

  const handleRefresh = useCallback(() => {
    if (!lastRequestArgs) return;

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

  const handleClose = useCallback(() => {
    setIsModalOpen(false);
    setSelectedVariant(null);
    setLastRequestArgs(null);
  }, []);

  return (
    <>
      <TryWithSelect
        options={options}
        onSelect={handleSelect}
        isLoading={isLoading}
        isDefaultFunction={isDefault}
      />

      {selectedVariant && (
        <VariantResponseModal
          isOpen={isModalOpen}
          isLoading={isLoading}
          error={variantInferenceFetcher.error?.message}
          variantResponse={variantInferenceFetcher.data?.info ?? null}
          rawResponse={variantInferenceFetcher.data?.raw ?? null}
          onClose={handleClose}
          item={inference}
          inferenceUsage={inferenceUsage}
          selectedVariant={selectedVariant}
          source="inference"
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
